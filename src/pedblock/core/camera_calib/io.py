from __future__ import annotations

import json
import os
from pathlib import Path

from .models import CameraCalib
from .paths import DEFAULT_CALIB_PATH
from .transform import order_quad_tl_tr_br_bl


def load_camera_calib(path: str = DEFAULT_CALIB_PATH) -> CameraCalib | None:
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        pts = d.get("image_points_px", None)
        if not isinstance(pts, list) or len(pts) != 4:
            return None
        image_points_px = [(float(p[0]), float(p[1])) for p in pts]
        calib = CameraCalib(
            image_points_px=image_points_px,
            L1_m=float(d.get("L1_m", 0.0)),
            L2_m=float(d.get("L2_m", 0.0)),
            W1_m=float(d.get("W1_m", 0.0)),
            W2_m=float(d.get("W2_m", 0.0)),
            px_per_meter=float(d.get("px_per_meter", 60.0)),
        )
        calib.validate()
        return calib
    except Exception:
        return None


def save_camera_calib(calib: CameraCalib, path: str = DEFAULT_CALIB_PATH) -> None:
    calib.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pts = order_quad_tl_tr_br_bl(calib.image_points_px)
    payload = {
        "image_points_px": [[float(x), float(y)] for (x, y) in pts],
        "L1_m": float(calib.L1_m),
        "L2_m": float(calib.L2_m),
        "W1_m": float(calib.W1_m),
        "W2_m": float(calib.W2_m),
        "px_per_meter": float(calib.px_per_meter),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = [
    "load_camera_calib",
    "save_camera_calib",
]


