from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from pedblock.core.annotate import draw_annotations
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.events import EventRecorder, ObstructionSpan, spans_to_jsonable
from pedblock.core.roi import roi_pct_to_px
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
                self._speed = max(0.1, min(8.0, v))
            elif cmd.kind == "roi":
                try:
                    x, y, w, h = cmd.value  # type: ignore[misc]
                    self._roi_pct = RoiPct(x=float(x), y=float(y), w=float(w), h=float(h)).clamp()
                except Exception:
                    # ignore malformed roi commands
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

        # initialize mutable runtime ROI from config
        self._roi_pct = RoiPct(
            x=float(det_cfg.roi.x),
            y=float(det_cfg.roi.y),
            w=float(det_cfg.roi.w),
            h=float(det_cfg.roi.h),
        ).clamp()

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
                roi = roi_pct_to_px(self._roi_pct.x, self._roi_pct.y, self._roi_pct.w, self._roi_pct.h, w, h)

                persons = [b.clamp(w, h) for b in det.detect_persons(frame)]
                frame_area = float(w * h) if w > 0 and h > 0 else 1.0
                min_area = (det_cfg.min_area_pct / 100.0) * frame_area

                obstructing = False
                for b in persons:
                    if b.area < min_area:
                        continue
                    if roi.contains_point(b.cx, b.cy):
                        obstructing = True
                        break

                spans.extend(recorder.update(frame_index, obstructing))

                annotated = draw_annotations(frame, roi, persons, obstructing)
                if writer is not None:
                    writer.write(annotated)

                info = FrameInfo(frame_index=frame_index, fps=meta.fps, width=w, height=h)
                self._emit(FrameResult(frame_info=info, frame_bgr=frame, persons=persons, obstructing=obstructing))

                now = time.time()
                if now - last_progress_emit > 0.2:
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


