from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "PointPct",
    "PointPx",
    "DangerZonePct",
    "DangerZonePx",
    "danger_zone_pct_to_px",
]


PointPct = tuple[float, float]
PointPx = tuple[int, int]


def _clamp_pct(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _ray_cast_contains_point(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    # General polygon point-in-polygon via ray casting.
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        denom = (yj - yi) if (yj - yi) != 0 else 1e-9
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / denom + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


@dataclass(slots=True)
class DangerZonePct:
    """
    Danger zone as a polygon in percents of frame size.

    For backward compatibility this can be constructed from 4 points (quad),
    but it can also represent any polygon (>= 3 points).
    """

    points: list[PointPct]

    @staticmethod
    def default_quad() -> "DangerZonePct":
        return DangerZonePct(points=[(25.0, 60.0), (75.0, 60.0), (95.0, 98.0), (5.0, 98.0)])

    @staticmethod
    def from_quad(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> "DangerZonePct":
        return DangerZonePct(points=[(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    def clamp(self) -> "DangerZonePct":
        self.points = [(_clamp_pct(x), _clamp_pct(y)) for (x, y) in self.points]
        return self


@dataclass(frozen=True, slots=True)
class DangerZonePx:
    points: tuple[PointPx, ...]

    def clamp(self, w: int, h: int) -> "DangerZonePx":
        def cx(v: int) -> int:
            return max(0, min(int(w - 1), int(v)))

        def cy(v: int) -> int:
            return max(0, min(int(h - 1), int(v)))

        return DangerZonePx(points=tuple((cx(x), cy(y)) for (x, y) in self.points))

    def contains_point(self, x: float, y: float) -> bool:
        poly = [(float(px), float(py)) for (px, py) in self.points]
        return _ray_cast_contains_point(poly, x, y)

    @property
    def as_int32_polyline(self) -> "list[list[int]]":
        # OpenCV expects Nx2 int32
        return [[int(x), int(y)] for (x, y) in self.points]


def danger_zone_pct_to_px(dz: DangerZonePct, w: int, h: int) -> DangerZonePx:
    dz = DangerZonePct(points=[(float(x), float(y)) for (x, y) in dz.points]).clamp()

    def px_x(pct: float) -> int:
        return int(round((pct / 100.0) * w))

    def px_y(pct: float) -> int:
        return int(round((pct / 100.0) * h))

    pts = tuple((px_x(x), px_y(y)) for (x, y) in dz.points)
    return DangerZonePx(points=pts).clamp(w, h)


