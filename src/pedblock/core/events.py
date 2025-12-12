from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ObstructionSpan:
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float


@dataclass(slots=True)
class EventRecorder:
    """Turns per-frame boolean signal into time spans."""

    fps: float
    _open_start_frame: int | None = None

    def update(self, frame_index: int, obstructing: bool) -> list[ObstructionSpan]:
        emitted: list[ObstructionSpan] = []
        if obstructing and self._open_start_frame is None:
            self._open_start_frame = frame_index
            return emitted

        if (not obstructing) and self._open_start_frame is not None:
            s = self._open_start_frame
            e = frame_index - 1
            emitted.append(
                ObstructionSpan(
                    start_frame=s,
                    end_frame=e,
                    start_time_s=(s / self.fps) if self.fps > 0 else 0.0,
                    end_time_s=(e / self.fps) if self.fps > 0 else 0.0,
                )
            )
            self._open_start_frame = None
        return emitted

    def close(self, last_frame_index: int) -> list[ObstructionSpan]:
        if self._open_start_frame is None:
            return []
        s = self._open_start_frame
        e = last_frame_index
        self._open_start_frame = None
        return [
            ObstructionSpan(
                start_frame=s,
                end_frame=e,
                start_time_s=(s / self.fps) if self.fps > 0 else 0.0,
                end_time_s=(e / self.fps) if self.fps > 0 else 0.0,
            )
        ]


def spans_to_jsonable(spans: list[ObstructionSpan]) -> list[dict]:
    return [asdict(s) for s in spans]


