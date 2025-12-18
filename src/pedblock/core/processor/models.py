from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedblock.core.types import Box, FrameInfo


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



