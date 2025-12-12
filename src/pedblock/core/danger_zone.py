from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DangerZonePct:
    """
    Danger zone as a convex quad in percents of frame size.

    Point order: p1 -> p2 -> p3 -> p4 (clockwise or counter-clockwise).
    """

    x1: float = 25.0
    y1: float = 60.0
    x2: float = 75.0
    y2: float = 60.0
    x3: float = 95.0
    y3: float = 98.0
    x4: float = 5.0
    y4: float = 98.0

    def clamp(self) -> "DangerZonePct":
        def c(v: float) -> float:
            return max(0.0, min(100.0, float(v)))

        self.x1, self.y1 = c(self.x1), c(self.y1)
        self.x2, self.y2 = c(self.x2), c(self.y2)
        self.x3, self.y3 = c(self.x3), c(self.y3)
        self.x4, self.y4 = c(self.x4), c(self.y4)
        return self


@dataclass(frozen=True, slots=True)
class DangerZonePx:
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int

    def clamp(self, w: int, h: int) -> "DangerZonePx":
        def cx(v: int) -> int:
            return max(0, min(int(w - 1), int(v)))

        def cy(v: int) -> int:
            return max(0, min(int(h - 1), int(v)))

        return DangerZonePx(
            x1=cx(self.x1),
            y1=cy(self.y1),
            x2=cx(self.x2),
            y2=cy(self.y2),
            x3=cx(self.x3),
            y3=cy(self.y3),
            x4=cx(self.x4),
            y4=cy(self.y4),
        )

    def contains_point(self, x: float, y: float) -> bool:
        # Convex quad point-in-polygon via ray casting (general polygon).
        pts = [(float(self.x1), float(self.y1)), (float(self.x2), float(self.y2)), (float(self.x3), float(self.y3)), (float(self.x4), float(self.y4))]
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside


def danger_zone_pct_to_px(dz: DangerZonePct, w: int, h: int) -> DangerZonePx:
    dz = DangerZonePct(
        x1=float(dz.x1),
        y1=float(dz.y1),
        x2=float(dz.x2),
        y2=float(dz.y2),
        x3=float(dz.x3),
        y3=float(dz.y3),
        x4=float(dz.x4),
        y4=float(dz.y4),
    ).clamp()

    def px_x(pct: float) -> int:
        return int(round((pct / 100.0) * w))

    def px_y(pct: float) -> int:
        return int(round((pct / 100.0) * h))

    return DangerZonePx(
        x1=px_x(dz.x1),
        y1=px_y(dz.y1),
        x2=px_x(dz.x2),
        y2=px_y(dz.y2),
        x3=px_x(dz.x3),
        y3=px_y(dz.y3),
        x4=px_x(dz.x4),
        y4=px_y(dz.y4),
    ).clamp(w, h)


