from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .transform import build_dst_points, order_quad_tl_tr_br_bl


@dataclass(frozen=True, slots=True)
class CameraCalib:
    """
    Калибровка перспективного преобразования (OpenADAS-style).

    Параметры (как в документации OpenADAS):
    - L1, L2: расстояния вперёд (метры) до ближней/дальней линии
    - W1, W2: ширина (метры) между левым/правым краем дороги на этих дистанциях
    - image_points_px: 4 точки на изображении (в пикселях), в порядке:
        TL, TR, BR, BL
    """

    image_points_px: list[tuple[float, float]]  # TL, TR, BR, BL
    L1_m: float
    L2_m: float
    W1_m: float
    W2_m: float
    px_per_meter: float = 60.0

    def validate(self) -> "CameraCalib":
        pts = list(self.image_points_px or [])
        if len(pts) != 4:
            raise ValueError("Нужно ровно 4 точки (TL, TR, BR, BL)")
        for x, y in pts:
            if not np.isfinite([x, y]).all():
                raise ValueError("Точки должны быть конечными числами")
        if not (self.L2_m > self.L1_m >= 0.0):
            raise ValueError("Ожидается L2 > L1 >= 0")
        if not (self.W1_m > 0.0 and self.W2_m > 0.0):
            raise ValueError("Ожидается W1 > 0 и W2 > 0")
        if not (self.px_per_meter > 1.0):
            raise ValueError("px_per_meter должен быть > 1")
        return self

    def build_birdeye(self, frame_w: int, frame_h: int) -> "BirdeyeTransform":
        self.validate()
        src_pts = order_quad_tl_tr_br_bl(self.image_points_px)
        src = np.array(src_pts, dtype=np.float32)
        dst, bev_w, bev_h = build_dst_points(self.L1_m, self.L2_m, self.W1_m, self.W2_m, self.px_per_meter)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return BirdeyeTransform(
            M=M,
            Minv=Minv,
            bev_size=(int(bev_w), int(bev_h)),
            frame_size=(int(frame_w), int(frame_h)),
            px_per_meter=float(self.px_per_meter),
        )


@dataclass(frozen=True, slots=True)
class BirdeyeTransform:
    M: np.ndarray  # 3x3 float64
    Minv: np.ndarray  # 3x3 float64
    bev_size: tuple[int, int]  # (W, H)
    frame_size: tuple[int, int]  # (W, H)
    px_per_meter: float

    def warp_to_bev(self, frame_bgr: np.ndarray) -> np.ndarray:
        w, h = self.bev_size
        return cv2.warpPerspective(frame_bgr, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def warp_mask_to_frame(self, mask_u8_bev: np.ndarray) -> np.ndarray:
        fw, fh = self.frame_size
        out = cv2.warpPerspective(mask_u8_bev, self.Minv, (fw, fh), flags=cv2.INTER_NEAREST)
        if out.dtype != np.uint8:
            out = out.astype(np.uint8, copy=False)
        return out


__all__ = [
    "CameraCalib",
    "BirdeyeTransform",
]


