from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from pedblock.core.annotate import draw_annotations
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.danger_zone import DangerZonePct, danger_zone_pct_to_px
from pedblock.core.events import EventRecorder, ObstructionSpan, spans_to_jsonable
from pedblock.core.roi import roi_pct_to_px
from pedblock.core.road_area import RoadAreaParams, danger_zone_pct_from_road_mask, estimate_road_mask
from pedblock.core.types import Box, FrameInfo
from pedblock.core.video_io import make_writer, open_video, probe_video
from pedblock.core.yolo_ultralytics import YoloUltralyticsDetector


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_info: FrameInfo
    frame_bgr: np.ndarray
    persons: list[Box]
    obstructing: bool


@dataclass(frozen=True, slots=True)
class ProcessorProgress:
    frame_index: int
    frame_count: int
    fps: float
    obstructing: bool


@dataclass(frozen=True, slots=True)
class ProcessorStatus:
    running: bool
    paused: bool
    realtime: bool
    speed: float


@dataclass(frozen=True, slots=True)
class _ControlCmd:
    kind: str
    value: object | None = None


class VideoProcessor:
    _SPEED_MIN = 0.1
    _SPEED_MAX = 8.0
    _PROGRESS_EMIT_PERIOD_S = 0.2
    _ROAD_DZ_ALPHA = 0.25

    _DZ_MODE_ROI = "roi"
    _DZ_MODE_ROAD = "road"

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._out_queue: "queue.Queue[FrameResult | ProcessorProgress | Exception]" = queue.Queue(maxsize=5)
        self._running_lock = threading.Lock()
        self._ctrl_queue: "queue.Queue[_ControlCmd]" = queue.Queue(maxsize=50)
        self._paused = False
        self._realtime = True
        self._speed = 1.0
        self._roi_pct = RoiPct()
        self._danger_zone_pct: DangerZonePct | None = None
        self._danger_zone_manual_override = False
        self._danger_zone_mode = self._DZ_MODE_ROI
        self._show_road_mask = False
        self._road_params = RoadAreaParams()

    def start(self, video_path: str, det_cfg: DetectionConfig, exp_cfg: ExportConfig) -> None:
        with self._running_lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Процессор уже запущен")
            self._stop_event.clear()
            self._paused = False
            self._realtime = True
            self._speed = 1.0
            self._thread = threading.Thread(
                target=self._run,
                args=(video_path, det_cfg, exp_cfg),
                daemon=True,
                name="VideoProcessorThread",
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def pause(self) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="pause"))

    def resume(self) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="resume"))

    def set_realtime(self, enabled: bool) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="realtime", value=bool(enabled)))

    def set_speed(self, speed: float) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="speed", value=float(speed)))

    def set_roi_pct(self, roi: RoiPct) -> None:
        # send as a tuple to keep control channel simple/pickle-free
        roi = RoiPct(x=float(roi.x), y=float(roi.y), w=float(roi.w), h=float(roi.h)).clamp()
        self._enqueue_ctrl(_ControlCmd(kind="roi", value=(roi.x, roi.y, roi.w, roi.h)))

    def set_danger_zone_mode(self, mode: str) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="danger_zone_mode", value=str(mode or "roi")))

    def set_show_road_mask(self, enabled: bool) -> None:
        self._enqueue_ctrl(_ControlCmd(kind="show_road_mask", value=bool(enabled)))

    def set_road_params(
        self,
        *,
        color_dist_thresh: float | None = None,
        use_perspective: bool | None = None,
        dz_near_bottom_frac: float | None = None,
        dz_edge_quantile: float | None = None,
        dz_method: str | None = None,
    ) -> None:
        # Keep the control message simple: send only a small dict of primitives.
        # (For backward compatibility, _drain_ctrl accepts older float-coded values too.)
        payload: dict[str, object] = {}
        if color_dist_thresh is not None:
            payload["color_dist_thresh"] = float(color_dist_thresh)
        if use_perspective is not None:
            payload["use_perspective"] = bool(use_perspective)
        if dz_near_bottom_frac is not None:
            payload["dz_near_bottom_frac"] = float(dz_near_bottom_frac)
        if dz_edge_quantile is not None:
            payload["dz_edge_quantile"] = float(dz_edge_quantile)
        if dz_method is not None:
            payload["dz_method"] = str(dz_method or "fit").strip().lower()
        self._enqueue_ctrl(_ControlCmd(kind="road_params", value=payload))

    def set_danger_zone_pct(self, dz: DangerZonePct) -> None:
        dz = DangerZonePct(points=[(float(x), float(y)) for (x, y) in dz.points]).clamp()
        flat: list[float] = []
        for x, y in dz.points:
            flat.append(float(x))
            flat.append(float(y))
        self._enqueue_ctrl(_ControlCmd(kind="danger_zone", value=tuple(flat)))

    def poll(self, max_items: int = 5) -> list[FrameResult | ProcessorProgress | Exception]:
        items: list[FrameResult | ProcessorProgress | Exception] = []
        for _ in range(max_items):
            try:
                items.append(self._out_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _emit(self, item: FrameResult | ProcessorProgress | Exception) -> None:
        # Drop oldest to keep UI responsive.
        try:
            self._out_queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self._out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._out_queue.put_nowait(item)
            except queue.Full:
                pass

    def _enqueue_ctrl(self, cmd: _ControlCmd) -> None:
        try:
            self._ctrl_queue.put_nowait(cmd)
        except queue.Full:
            # drop oldest command (rare)
            try:
                _ = self._ctrl_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._ctrl_queue.put_nowait(cmd)
            except queue.Full:
                pass

    def _drain_ctrl(self) -> None:
        while True:
            try:
                cmd = self._ctrl_queue.get_nowait()
            except queue.Empty:
                break

            if cmd.kind == "pause":
                self._paused = True
            elif cmd.kind == "resume":
                self._paused = False
            elif cmd.kind == "realtime":
                self._realtime = bool(cmd.value)
            elif cmd.kind == "speed":
                v = float(cmd.value) if cmd.value is not None else 1.0
                self._speed = max(self._SPEED_MIN, min(self._SPEED_MAX, v))
            elif cmd.kind == "roi":
                try:
                    x, y, w, h = cmd.value  # type: ignore[misc]
                    self._roi_pct = RoiPct(x=float(x), y=float(y), w=float(w), h=float(h)).clamp()
                    # If user sets a rectangle ROI manually, treat it as a quad (rectangle).
                    self._danger_zone_pct = DangerZonePct.from_quad(
                        self._roi_pct.x,
                        self._roi_pct.y,
                        self._roi_pct.x + self._roi_pct.w,
                        self._roi_pct.y,
                        self._roi_pct.x + self._roi_pct.w,
                        self._roi_pct.y + self._roi_pct.h,
                        self._roi_pct.x,
                        self._roi_pct.y + self._roi_pct.h,
                    ).clamp()
                    self._danger_zone_manual_override = False
                except Exception:
                    # ignore malformed roi commands
                    pass
            elif cmd.kind == "danger_zone":
                try:
                    flat = list(cmd.value)  # type: ignore[arg-type]
                    pts: list[tuple[float, float]] = []
                    for i in range(0, len(flat) - 1, 2):
                        pts.append((float(flat[i]), float(flat[i + 1])))
                    self._danger_zone_pct = DangerZonePct(points=pts).clamp()
                    # Если мы в режиме "road", полигон может приходить как подсказка/фоллбек,
                    # но не должен отключать road-based расчёт на каждом кадре.
                    self._danger_zone_manual_override = bool(self._danger_zone_mode != self._DZ_MODE_ROAD)
                except Exception:
                    pass
            elif cmd.kind == "danger_zone_mode":
                try:
                    m = str(cmd.value or "roi").strip().lower()
                    if m not in (self._DZ_MODE_ROI, self._DZ_MODE_ROAD):
                        m = self._DZ_MODE_ROI
                    self._danger_zone_mode = m
                    # Switching mode should clear manual override (user intent is mode-driven).
                    self._danger_zone_manual_override = False
                    if m == self._DZ_MODE_ROI:
                        # ensure ROI->quad is the active zone baseline
                        self._danger_zone_pct = DangerZonePct.from_quad(
                            self._roi_pct.x,
                            self._roi_pct.y,
                            self._roi_pct.x + self._roi_pct.w,
                            self._roi_pct.y,
                            self._roi_pct.x + self._roi_pct.w,
                            self._roi_pct.y + self._roi_pct.h,
                            self._roi_pct.x,
                            self._roi_pct.y + self._roi_pct.h,
                        ).clamp()
                except Exception:
                    pass
            elif cmd.kind == "show_road_mask":
                try:
                    self._show_road_mask = bool(cmd.value)
                except Exception:
                    pass
            elif cmd.kind == "road_params":
                try:
                    d = dict(cmd.value or {})  # type: ignore[arg-type]
                    updates: dict[str, object] = {}

                    def _to_bool(v: object) -> bool:
                        if isinstance(v, bool):
                            return v
                        if isinstance(v, (int, float)):
                            return bool(v)
                        s = str(v).strip().lower()
                        if s in ("1", "true", "yes", "y", "on"):
                            return True
                        if s in ("0", "false", "no", "n", "off", ""):
                            return False
                        return bool(v)

                    def _clamped_float(v: object, *, lo: float, hi: float) -> float:
                        try:
                            fv = float(v)  # type: ignore[arg-type]
                        except Exception:
                            return float(lo)
                        return max(float(lo), min(float(hi), fv))

                    def _parse_dz_method(v: object | None) -> str:
                        # Backward compatibility: historically encoded as float (0/1/2).
                        if v is None:
                            return str(self._road_params.dz_method or "fit")
                        if isinstance(v, (int, float)):
                            fv = float(v)
                            if fv >= 1.5:
                                return "hough"
                            if fv >= 0.5:
                                return "max_width"
                            return "fit"
                        s = str(v).strip().lower()
                        if s in ("max_width", "maxwidth", "max-width", "max"):
                            return "max_width"
                        if s in ("hough", "lines", "mask_hough", "mask-lines"):
                            return "hough"
                        return "fit"

                    if "color_dist_thresh" in d:
                        updates["color_dist_thresh"] = _clamped_float(d["color_dist_thresh"], lo=1.0, hi=60.0)
                    if "use_perspective" in d:
                        updates["use_perspective"] = _to_bool(d["use_perspective"])
                    if "dz_near_bottom_frac" in d:
                        updates["dz_near_bottom_frac"] = _clamped_float(d["dz_near_bottom_frac"], lo=0.05, hi=0.95)
                    if "dz_edge_quantile" in d:
                        updates["dz_edge_quantile"] = _clamped_float(d["dz_edge_quantile"], lo=0.0, hi=0.20)
                    if "dz_method" in d:
                        updates["dz_method"] = _parse_dz_method(d.get("dz_method"))

                    if updates:
                        self._road_params = replace(self._road_params, **updates)
                except Exception:
                    pass

    def _run(self, video_path: str, det_cfg: DetectionConfig, exp_cfg: ExportConfig) -> None:
        try:
            meta = probe_video(video_path)
            cap = open_video(video_path)
        except Exception as e:
            self._emit(e)
            return

        det = None
        try:
            det = YoloUltralyticsDetector(det_cfg)
        except Exception as e:
            cap.release()
            self._emit(e)
            return

        out_dir = exp_cfg.out_dir.strip() if exp_cfg.out_dir else ""
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(video_path), "pedblock_out")
        os.makedirs(out_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_video_path = os.path.join(out_dir, f"{base}_{ts}_annotated.mp4")
        out_json_path = os.path.join(out_dir, f"{base}_{ts}_events.json")

        writer = None
        if exp_cfg.export_video:
            try:
                writer = make_writer(out_video_path, meta.fps, meta.width, meta.height)
            except Exception as e:
                cap.release()
                self._emit(e)
                return

        recorder = EventRecorder(fps=meta.fps if meta.fps > 0 else 25.0)
        spans: list[ObstructionSpan] = []

        frame_index = -1
        last_progress_emit = 0.0
        last_used_dz_pct: DangerZonePct | None = None

        # initialize mutable runtime ROI from config
        self._roi_pct = RoiPct(
            x=float(det_cfg.roi.x),
            y=float(det_cfg.roi.y),
            w=float(det_cfg.roi.w),
            h=float(det_cfg.roi.h),
        ).clamp()
        self._danger_zone_pct = DangerZonePct.from_quad(
            self._roi_pct.x,
            self._roi_pct.y,
            self._roi_pct.x + self._roi_pct.w,
            self._roi_pct.y,
            self._roi_pct.x + self._roi_pct.w,
            self._roi_pct.y + self._roi_pct.h,
            self._roi_pct.x,
            self._roi_pct.y + self._roi_pct.h,
        ).clamp()
        self._danger_zone_manual_override = False
        self._danger_zone_mode = (det_cfg.danger_zone_mode or "roi").strip().lower()
        if self._danger_zone_mode not in (self._DZ_MODE_ROI, self._DZ_MODE_ROAD):
            self._danger_zone_mode = self._DZ_MODE_ROI
        self._show_road_mask = bool(det_cfg.show_road_mask)
        self._road_params = RoadAreaParams()

        # road-based danger zone smoothing (best-effort)
        road_dz_smoothed: DangerZonePct | None = None
        road_alpha = self._ROAD_DZ_ALPHA

        try:
            while not self._stop_event.is_set():
                self._drain_ctrl()
                if self._paused:
                    time.sleep(0.03)
                    continue

                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1

                h, w = frame.shape[:2]
                road_mask_u8: np.ndarray | None = None
                dz_pct: DangerZonePct | None = None

                # 1) choose danger zone source
                mode = (self._danger_zone_mode or "roi").strip().lower()
                if mode == self._DZ_MODE_ROAD and not self._danger_zone_manual_override:
                    # detect drivable area mask and derive danger zone from it
                    road_params = self._road_params
                    rm = estimate_road_mask(frame, road_params)
                    road_mask_u8 = rm.mask_u8
                    est = danger_zone_pct_from_road_mask(road_mask_u8, road_params)
                    if est is not None:
                        prev = road_dz_smoothed
                        if prev is None:
                            road_dz_smoothed = est
                        else:
                            n = min(len(prev.points), len(est.points))
                            pts: list[tuple[float, float]] = []
                            for i in range(n):
                                px, py = prev.points[i]
                                ex, ey = est.points[i]
                                pts.append(((1 - road_alpha) * px + road_alpha * ex, (1 - road_alpha) * py + road_alpha * ey))
                            road_dz_smoothed = DangerZonePct(points=pts).clamp()
                        dz_pct = road_dz_smoothed

                # 2) fallback / manual ROI-based zone
                if dz_pct is None:
                    dz_pct = self._danger_zone_pct
                if dz_pct is None:
                    _ = roi_pct_to_px(self._roi_pct.x, self._roi_pct.y, self._roi_pct.w, self._roi_pct.h, w, h)
                    dz_pct = DangerZonePct.from_quad(
                        self._roi_pct.x,
                        self._roi_pct.y,
                        self._roi_pct.x + self._roi_pct.w,
                        self._roi_pct.y,
                        self._roi_pct.x + self._roi_pct.w,
                        self._roi_pct.y + self._roi_pct.h,
                        self._roi_pct.x,
                        self._roi_pct.y + self._roi_pct.h,
                    ).clamp()

                last_used_dz_pct = dz_pct
                dz = danger_zone_pct_to_px(dz_pct, w, h)

                persons = [b.clamp(w, h) for b in det.detect_persons(frame)]
                frame_area = float(w * h) if w > 0 and h > 0 else 1.0
                min_area = (det_cfg.min_area_pct / 100.0) * frame_area

                obstructing = False
                for b in persons:
                    if b.area < min_area:
                        continue
                    if dz.contains_point(b.cx, b.cy):
                        obstructing = True
                        break

                spans.extend(recorder.update(frame_index, obstructing))

                annotated = draw_annotations(
                    frame,
                    dz,
                    persons,
                    obstructing,
                    road_mask_u8 if bool(self._show_road_mask) else None,
                )
                if writer is not None:
                    writer.write(annotated)

                info = FrameInfo(frame_index=frame_index, fps=meta.fps, width=w, height=h)
                self._emit(FrameResult(frame_info=info, frame_bgr=frame, persons=persons, obstructing=obstructing))

                now = time.time()
                if now - last_progress_emit > self._PROGRESS_EMIT_PERIOD_S:
                    last_progress_emit = now
                    self._emit(
                        ProcessorProgress(
                            frame_index=frame_index,
                            frame_count=meta.frame_count,
                            fps=meta.fps,
                            obstructing=obstructing,
                        )
                    )

                # Realtime playback pacing (best-effort).
                if self._realtime:
                    fps = meta.fps if meta.fps and meta.fps > 0 else 25.0
                    frame_dt = (1.0 / fps) / (self._speed if self._speed > 0 else 1.0)
                    proc_dt = time.perf_counter() - t0
                    if frame_dt > proc_dt:
                        time.sleep(frame_dt - proc_dt)
        except Exception as e:
            self._emit(e)
        finally:
            try:
                spans.extend(recorder.close(frame_index))
            except Exception:
                pass

            cap.release()
            if writer is not None:
                writer.release()

            if exp_cfg.export_json:
                payload = {
                    "video_path": video_path,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "roi_pct": {"x": self._roi_pct.x, "y": self._roi_pct.y, "w": self._roi_pct.w, "h": self._roi_pct.h},
                    "danger_zone_pct": (
                        None
                        if last_used_dz_pct is None
                        else {
                            "points": [{"x": float(x), "y": float(y)} for (x, y) in last_used_dz_pct.points],
                        }
                    ),
                    "danger_zone_mode": (self._danger_zone_mode or (det_cfg.danger_zone_mode or "roi")),
                    "show_road_mask": bool(self._show_road_mask),
                    "min_area_pct": det_cfg.min_area_pct,
                    "model_name": det_cfg.model_name,
                    "device": det_cfg.device,
                    "conf": det_cfg.conf,
                    "iou": det_cfg.iou,
                    "spans": spans_to_jsonable(spans),
                }
                try:
                    with open(out_json_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    self._emit(e)


