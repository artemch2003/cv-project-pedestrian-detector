from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _default_calib_path() -> str:
    # src/pedblock/core/camera_calib.py -> project root is parents[3]
    root = Path(__file__).resolve().parents[3]
    return str(root / "data" / "camera_calib.json")


DEFAULT_CALIB_PATH = _default_calib_path()


@dataclass(frozen=True, slots=True)
class CameraCalib:
    """
    Калибровка перспективного преобразования (OpenADAS-style).

    Смысл параметров (как в документации OpenADAS):
    - L1, L2: расстояния вперёд (метры) до ближней/дальней линии
    - W1, W2: ширина (метры) между левым/правым краем дороги на этих дистанциях
    - image_points_px: 4 точки на изображении (в пикселях), в порядке:
        TL, TR, BR, BL

    По этим данным строим матрицы гомографии:
    - M: image -> bird view
    - Minv: bird view -> image
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
        dst, bev_w, bev_h = _build_dst_points(self.L1_m, self.L2_m, self.W1_m, self.W2_m, self.px_per_meter)
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


def order_quad_tl_tr_br_bl(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
    """
    Нормализует порядок 4 точек (квад) в TL, TR, BR, BL в координатах изображения.

    Работает с любым порядком ввода и устойчиво для трапеции дороги:
    - TL: минимальная сумма (x+y)
    - BR: максимальная сумма (x+y)
    - TR: минимальная разность (x-y)
    - BL: максимальная разность (x-y)
    """
    pts = [(float(x), float(y)) for (x, y) in list(points or [])]
    if len(pts) != 4:
        raise ValueError("Нужно 4 точки для нормализации порядка")
    arr = np.array(pts, dtype=np.float32)
    s = arr[:, 0] + arr[:, 1]
    d = arr[:, 0] - arr[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(d))]
    bl = pts[int(np.argmax(d))]
    out = [tl, tr, br, bl]
    # sanity: avoid duplicates due to degenerate geometry
    if len({(round(x, 3), round(y, 3)) for (x, y) in out}) != 4:
        # fallback: sort by y then x and build TL/TR from top row, BL/BR from bottom row
        pts_sorted = sorted(pts, key=lambda p: (p[1], p[0]))
        top2 = sorted(pts_sorted[:2], key=lambda p: p[0])
        bot2 = sorted(pts_sorted[2:], key=lambda p: p[0])
        out = [top2[0], top2[1], bot2[1], bot2[0]]  # TL, TR, BR, BL
    return out


def _build_dst_points(L1_m: float, L2_m: float, W1_m: float, W2_m: float, px_per_meter: float) -> tuple[np.ndarray, int, int]:
    """
    Строим dst-точки в bird-view (в пикселях) по размерам в метрах.

    Принято:
    - верх bird-view (y=0) соответствует дистанции L2 (дальняя линия)
    - низ bird-view (y=H) соответствует дистанции L1 (ближняя линия)
    - ширина на верху равна W2, ширина на низу равна W1
    - трапеция центрируется по оси X.
    """
    H_m = float(L2_m - L1_m)
    maxW = float(max(W1_m, W2_m))
    bev_w = int(max(64, round(maxW * px_per_meter)))
    bev_h = int(max(64, round(H_m * px_per_meter)))

    def x_left(width_m: float) -> float:
        return float((maxW - float(width_m)) * 0.5 * px_per_meter)

    # far (top) uses W2, near (bottom) uses W1
    xl_f = x_left(W2_m)
    xr_f = xl_f + float(W2_m) * px_per_meter
    xl_n = x_left(W1_m)
    xr_n = xl_n + float(W1_m) * px_per_meter

    # Order: TL, TR, BR, BL in BEV coordinates
    dst = np.array(
        [
            [xl_f, 0.0],  # TL (far-left)
            [xr_f, 0.0],  # TR (far-right)
            [xr_n, float(bev_h - 1)],  # BR (near-right)
            [xl_n, float(bev_h - 1)],  # BL (near-left)
        ],
        dtype=np.float32,
    )
    return dst, bev_w, bev_h


