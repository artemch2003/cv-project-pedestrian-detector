from __future__ import annotations

import cv2
import numpy as np

from .models import RoadAreaParams


def _x_at_y(m: float, b: float, y: float) -> float:
    if abs(m) < 1e-6:
        return float("nan")
    return (y - b) / m


def _clamp_x(x: float, w: int) -> float:
    return float(max(0.0, min(float(w - 1), float(x))))


def _weighted_avg_line(items: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if not items:
        return None
    sw = sum(wt for _, _, wt in items)
    if sw <= 1e-6:
        return None
    m = sum(mv * wt for mv, _, wt in items) / sw
    b = sum(bv * wt for _, bv, wt in items) / sw
    return (float(m), float(b))


def _estimate_corridor_from_lines(
    frame_bgr: np.ndarray, p: RoadAreaParams, *, debug: dict | None
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray] | tuple[np.ndarray, list[tuple[int, int]]] | None:
    """
    Строит грубую маску-коридор (трапеция) между левой/правой «дорожными» линиями.
    Возвращает (mask_u8, polygon_px). Если debug != None, дополнительно возвращает edges_u8.
    """
    h, w = frame_bgr.shape[:2]

    # 1) edges
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, int(p.canny1), int(p.canny2))

    # 2) bottom ROI + margin crop
    y_cut = int(round(h * float(p.analyze_top_frac)))
    if 0 < y_cut < h:
        edges[:y_cut, :] = 0
    xm = int(round(w * float(max(0.0, min(0.45, p.analyze_x_margin_frac)))))
    if xm > 0 and xm * 2 < w:
        edges[:, :xm] = 0
        edges[:, w - xm :] = 0

    # 3) optional color mask (white/yellow markings)
    if p.use_lane_color_mask:
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(hls)
        white = cv2.inRange(L, 190, 255) & cv2.inRange(S, 0, 90)
        yellow = cv2.inRange(H, 12, 45) & cv2.inRange(S, 70, 255) & cv2.inRange(L, 80, 255)
        cmask = cv2.bitwise_or(white, yellow)
        if 0 < y_cut < h:
            cmask[:y_cut, :] = 0
        if xm > 0 and xm * 2 < w:
            cmask[:, :xm] = 0
            cmask[:, w - xm :] = 0
        cmask = cv2.dilate(cmask, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, cmask)

    # 4) Hough lines
    min_line_len = max(10, int(round(w * float(p.min_line_len_frac))))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(p.hough_threshold),
        minLineLength=int(min_line_len),
        maxLineGap=int(p.max_line_gap),
    )
    if lines is None or len(lines) == 0:
        return None

    # 5) choose left/right representative lines (weighted by length)
    left: list[tuple[float, float, float]] = []
    right: list[tuple[float, float, float]] = []
    for (x1, y1, x2, y2) in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        m = dy / dx
        if abs(m) < float(p.min_abs_slope):
            continue
        b = float(y1) - m * float(x1)
        wt = float(np.hypot(dx, dy))
        if wt <= 1e-3:
            continue
        if m < 0:
            left.append((m, b, wt))
        else:
            right.append((m, b, wt))

    abL = _weighted_avg_line(left)
    abR = _weighted_avg_line(right)
    if abL is None or abR is None:
        return None

    mL, bL = abL
    mR, bR = abR

    # 6) build corridor polygon between the two lines
    y_bottom = float(h - 1)
    y_top = float(int(round(h * float(p.corridor_top_frac))))
    y_top = max(0.0, min(float(h - 2), y_top))

    xL_t = _x_at_y(mL, bL, y_top)
    xR_t = _x_at_y(mR, bR, y_top)
    xL_b = _x_at_y(mL, bL, y_bottom)
    xR_b = _x_at_y(mR, bR, y_bottom)
    if not np.isfinite([xL_t, xR_t, xL_b, xR_b]).all():
        return None

    # normalize left/right ordering
    if xL_t > xR_t:
        xL_t, xR_t = xR_t, xL_t
    if xL_b > xR_b:
        xL_b, xR_b = xR_b, xL_b

    x_pad = float(w) * float(p.x_pad_frac)
    x1_t = _clamp_x(xL_t - x_pad, w)
    x2_t = _clamp_x(xR_t + x_pad, w)
    x1_b = _clamp_x(xL_b - x_pad, w)
    x2_b = _clamp_x(xR_b + x_pad, w)

    # sanity
    if (x2_t - x1_t) < 5 or (x2_b - x1_b) < 5:
        return None
    if (x2_b - x1_b) < int(round(float(w) * float(p.min_width_frac))):
        return None

    poly = [
        (int(round(x1_t)), int(round(y_top))),
        (int(round(x2_t)), int(round(y_top))),
        (int(round(x2_b)), int(round(y_bottom))),
        (int(round(x1_b)), int(round(y_bottom))),
    ]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    if debug is not None:
        return (mask, poly, edges)
    return (mask, poly)


