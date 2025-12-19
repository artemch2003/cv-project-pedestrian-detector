from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np

from pedblock.core.annotate import draw_annotations
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.danger_zone import DangerZonePct, danger_zone_pct_from_rect, danger_zone_pct_to_px
from pedblock.core.events import EventRecorder, ObstructionSpan, spans_to_jsonable
from pedblock.core.roi import roi_pct_to_px
from pedblock.core.road_area import RoadAreaParams, danger_zone_pct_from_road_mask, estimate_road_mask
from pedblock.core.types import FrameInfo
from pedblock.core.video_io import make_writer, open_video, probe_video
from pedblock.core.yolo_ultralytics import YoloUltralyticsDetector

from .models import FrameResult, ProcessorProgress
from .protocols import RunTarget


def run(vp: RunTarget, video_path: str, det_cfg: DetectionConfig, exp_cfg: ExportConfig) -> None:
    """
    Основной цикл обработки видео.
    Вынесено из `VideoProcessor._run()` ради читабельности.
    """
    try:
        meta = probe_video(video_path)
        cap = open_video(video_path)
    except Exception as e:
        vp._emit(e)
        return

    det = None
    try:
        det = YoloUltralyticsDetector(det_cfg)
    except Exception as e:
        cap.release()
        vp._emit(e)
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
            vp._emit(e)
            return

    recorder = EventRecorder(fps=meta.fps if meta.fps > 0 else 25.0)
    spans: list[ObstructionSpan] = []

    frame_index = -1
    last_progress_emit = 0.0
    last_used_dz_pct: DangerZonePct | None = None

    # initialize mutable runtime ROI from config
    vp._roi_pct = RoiPct(
        x=float(det_cfg.roi.x),
        y=float(det_cfg.roi.y),
        w=float(det_cfg.roi.w),
        h=float(det_cfg.roi.h),
    ).clamp()
    vp._danger_zone_pct = danger_zone_pct_from_rect(vp._roi_pct.x, vp._roi_pct.y, vp._roi_pct.w, vp._roi_pct.h)
    vp._danger_zone_manual_override = False
    vp._danger_zone_mode = (det_cfg.danger_zone_mode or "roi").strip().lower()
    if vp._danger_zone_mode not in (vp._DZ_MODE_ROI, vp._DZ_MODE_ROAD):
        vp._danger_zone_mode = vp._DZ_MODE_ROI
    vp._show_road_mask = bool(det_cfg.show_road_mask)
    vp._road_params = RoadAreaParams()

    # road-based danger zone smoothing (best-effort)
    road_dz_smoothed: DangerZonePct | None = None
    road_alpha = vp._ROAD_DZ_ALPHA

    try:
        while not vp._stop_event.is_set():
            vp._drain_ctrl()
            if vp._paused:
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
            mode = (vp._danger_zone_mode or "roi").strip().lower()
            if mode == vp._DZ_MODE_ROAD and not vp._danger_zone_manual_override:
                road_params = vp._road_params
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
                dz_pct = vp._danger_zone_pct
            if dz_pct is None:
                dz_pct = danger_zone_pct_from_rect(vp._roi_pct.x, vp._roi_pct.y, vp._roi_pct.w, vp._roi_pct.h)

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
                road_mask_u8 if bool(vp._show_road_mask) else None,
            )
            if writer is not None:
                writer.write(annotated)

            info = FrameInfo(frame_index=frame_index, fps=meta.fps, width=w, height=h)
            vp._emit(FrameResult(frame_info=info, frame_bgr=frame, persons=persons, obstructing=obstructing))

            now = time.time()
            if now - last_progress_emit > vp._PROGRESS_EMIT_PERIOD_S:
                last_progress_emit = now
                vp._emit(
                    ProcessorProgress(
                        frame_index=frame_index,
                        frame_count=meta.frame_count,
                        fps=meta.fps,
                        obstructing=obstructing,
                    )
                )

            # Realtime playback pacing (best-effort).
            if vp._realtime:
                fps = meta.fps if meta.fps and meta.fps > 0 else 25.0
                frame_dt = (1.0 / fps) / (vp._speed if vp._speed > 0 else 1.0)
                proc_dt = time.perf_counter() - t0
                if frame_dt > proc_dt:
                    time.sleep(frame_dt - proc_dt)
    except Exception as e:
        vp._emit(e)
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
                "roi_pct": {"x": vp._roi_pct.x, "y": vp._roi_pct.y, "w": vp._roi_pct.w, "h": vp._roi_pct.h},
                "danger_zone_pct": (
                    None
                    if last_used_dz_pct is None
                    else {
                        "points": [{"x": float(x), "y": float(y)} for (x, y) in last_used_dz_pct.points],
                    }
                ),
                "danger_zone_mode": (vp._danger_zone_mode or (det_cfg.danger_zone_mode or "roi")),
                "show_road_mask": bool(vp._show_road_mask),
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
                vp._emit(e)


