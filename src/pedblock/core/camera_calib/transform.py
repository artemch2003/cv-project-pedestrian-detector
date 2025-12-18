from __future__ import annotations

import numpy as np


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


def build_dst_points(L1_m: float, L2_m: float, W1_m: float, W2_m: float, px_per_meter: float) -> tuple[np.ndarray, int, int]:
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


__all__ = [
    "order_quad_tl_tr_br_bl",
    "build_dst_points",
]


