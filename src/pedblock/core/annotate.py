from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from pedblock.core.danger_zone import DangerZonePx
from pedblock.core.types import Box


def draw_annotations(
    bgr: np.ndarray,
    roi: DangerZonePx,
    boxes: Iterable[Box],
    obstructing: bool,
) -> np.ndarray:
    out = bgr.copy()

    # ROI
    pts = [(roi.x1, roi.y1), (roi.x2, roi.y2), (roi.x3, roi.y3), (roi.x4, roi.y4)]
    cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=True, color=(255, 200, 0), thickness=2)

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


