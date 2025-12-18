from __future__ import annotations

import queue
import threading
from typing import Protocol

from pedblock.core.config import RoiPct
from pedblock.core.danger_zone import DangerZonePct
from pedblock.core.road_area import RoadAreaParams

from .models import FrameResult, ProcessorProgress, _ControlCmd


class ControlTarget(Protocol):
    _ctrl_queue: "queue.Queue[_ControlCmd]"

    _paused: bool
    _realtime: bool
    _speed: float

    _SPEED_MIN: float
    _SPEED_MAX: float

    _roi_pct: RoiPct
    _danger_zone_pct: DangerZonePct | None
    _danger_zone_manual_override: bool
    _danger_zone_mode: str

    _DZ_MODE_ROI: str
    _DZ_MODE_ROAD: str

    _show_road_mask: bool
    _road_params: RoadAreaParams


class RunTarget(ControlTarget, Protocol):
    _stop_event: threading.Event
    _PROGRESS_EMIT_PERIOD_S: float
    _ROAD_DZ_ALPHA: float

    def _emit(self, item: FrameResult | ProcessorProgress | Exception) -> None: ...
    def _drain_ctrl(self) -> None: ...


