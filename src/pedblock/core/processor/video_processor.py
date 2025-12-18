from __future__ import annotations

import queue
import threading

from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.danger_zone import DangerZonePct
from pedblock.core.road_area import RoadAreaParams

from .control import drain_ctrl
from .models import FrameResult, ProcessorProgress, _ControlCmd
from .runner import run


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
        # (For backward compatibility, drain_ctrl accepts older float-coded values too.)
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
        drain_ctrl(self)

    def _run(self, video_path: str, det_cfg: DetectionConfig, exp_cfg: ExportConfig) -> None:
        run(self, video_path, det_cfg, exp_cfg)


