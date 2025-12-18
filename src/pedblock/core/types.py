from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FrameInfo",
    "Box",
]


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """Метаданные кадра (индекс/размер/FPS)."""

    frame_index: int
    fps: float
    width: int
    height: int

    @property
    def time_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_index / self.fps


@dataclass(frozen=True, slots=True)
class Box:
    """BBox детекции (xyxy + confidence)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def clamp(self, w: int, h: int) -> "Box":
        """Ограничивает координаты bbox границами кадра и нормализует порядок (x1<=x2, y1<=y2)."""
        x1 = max(0.0, min(float(w - 1), self.x1))
        x2 = max(0.0, min(float(w - 1), self.x2))
        y1 = max(0.0, min(float(h - 1), self.y1))
        y2 = max(0.0, min(float(h - 1), self.y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return Box(x1=x1, y1=y1, x2=x2, y2=y2, conf=self.conf)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


