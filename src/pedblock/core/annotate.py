from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from pedblock.core.danger_zone import DangerZonePx
from pedblock.core.types import Box

__all__ = [
    "draw_annotations",
]


def draw_annotations(
    bgr: np.ndarray,
    roi: DangerZonePx,
    boxes: Iterable[Box],
    obstructing: bool,
    road_mask_u8: np.ndarray | None = None,
    *,
    road_alpha: float = 0.25,
) -> np.ndarray:
    out = bgr.copy()

    # Road mask overlay (optional)
    if road_mask_u8 is not None and getattr(road_mask_u8, "size", 0) != 0:
        try:
            a = float(max(0.0, min(0.9, road_alpha)))
            if a > 0:
                mask = (road_mask_u8 > 0).astype(np.uint8)
                if mask.shape[:2] == out.shape[:2]:
                    overlay = out.copy()
                    # blue-ish fill
                    overlay[mask > 0] = (180, 80, 0)
                    out = cv2.addWeighted(overlay, a, out, 1.0 - a, 0.0)
        except Exception:
            # never fail annotation due to overlay
            pass

    # ROI
    cv2.polylines(out, [np.array(roi.as_int32_polyline, dtype=np.int32)], isClosed=True, color=(255, 200, 0), thickness=2)

    # Boxes
    for b in boxes:
        color = (0, 0, 255) if obstructing and roi.contains_point(b.cx, b.cy) else (0, 255, 0)
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(
            out,
            f"person {b.conf:.2f}",
            (int(b.x1), max(0, int(b.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Status overlay
    label = "МЕШАЕТ" if obstructing else "OK"
    lbl_color = (0, 0, 255) if obstructing else (0, 200, 0)
    cv2.putText(out, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, lbl_color, 2, cv2.LINE_AA)
    return out


