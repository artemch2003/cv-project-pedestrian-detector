from __future__ import annotations

import cv2
import numpy as np

from .models import RoadAreaParams, RoadMaskResult


def _fallback_mask(frame_bgr: np.ndarray, p: RoadAreaParams) -> RoadMaskResult:
    # Если кадра нет/не удалось — вернём маску по простой трапеции внизу (как «типичная дорога»).
    h = int(getattr(frame_bgr, "shape", (0, 0))[0] or 0)
    w = int(getattr(frame_bgr, "shape", (0, 0))[1] or 0)
    if h <= 0 or w <= 0:
        return RoadMaskResult(mask_u8=np.zeros((0, 0), dtype=np.uint8), polygon_px=[])

    cx = float(p.fallback_center_x) / 100.0 * float(w)
    top_y = float(p.fallback_top_y) / 100.0 * float(h)
    bot_y = float(p.fallback_bottom_y) / 100.0 * float(h)
    top_w = float(p.fallback_top_w) / 100.0 * float(w)
    bot_w = float(p.fallback_bottom_w) / 100.0 * float(w)
    poly = [
        (int(round(cx - top_w / 2.0)), int(round(top_y))),
        (int(round(cx + top_w / 2.0)), int(round(top_y))),
        (int(round(cx + bot_w / 2.0)), int(round(bot_y))),
        (int(round(cx - bot_w / 2.0)), int(round(bot_y))),
    ]
    poly = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for (x, y) in poly]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return RoadMaskResult(mask_u8=mask, polygon_px=poly)


