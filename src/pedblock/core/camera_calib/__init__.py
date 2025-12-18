"""
Калибровка перспективного преобразования (OpenADAS-style).

Это пакет-рефакторинг бывшего модуля `pedblock.core.camera_calib`.
Публичный API сохранён: DEFAULT_CALIB_PATH, CameraCalib, BirdeyeTransform,
load_camera_calib, save_camera_calib, order_quad_tl_tr_br_bl.
"""

from .paths import DEFAULT_CALIB_PATH
from .models import BirdeyeTransform, CameraCalib
from .io import load_camera_calib, save_camera_calib
from .transform import order_quad_tl_tr_br_bl

__all__ = [
    "DEFAULT_CALIB_PATH",
    "CameraCalib",
    "BirdeyeTransform",
    "load_camera_calib",
    "save_camera_calib",
    "order_quad_tl_tr_br_bl",
]


