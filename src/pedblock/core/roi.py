from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoiPx:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "RoiPx":
        x1 = max(0, min(w - 1, self.x1))
        x2 = max(0, min(w - 1, self.x2))
        y1 = max(0, min(h - 1, self.y1))
        y2 = max(0, min(h - 1, self.y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return RoiPx(x1=x1, y1=y1, x2=x2, y2=y2)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def roi_pct_to_px(x_pct: float, y_pct: float, w_pct: float, h_pct: float, w: int, h: int) -> RoiPx:
    x1 = int(round((x_pct / 100.0) * w))
    y1 = int(round((y_pct / 100.0) * h))
    x2 = int(round(((x_pct + w_pct) / 100.0) * w))
    y2 = int(round(((y_pct + h_pct) / 100.0) * h))
    return RoiPx(x1=x1, y1=y1, x2=x2, y2=y2).clamp(w, h)


