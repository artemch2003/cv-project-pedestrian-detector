from __future__ import annotations

import cv2
import numpy as np

from pedblock.core.danger_zone import DangerZonePct

from .models import AutoRoiParams


def _fallback_trapezoid(p: AutoRoiParams) -> DangerZonePct:
    cx = float(p.fallback_center_x)
    top_w = float(p.fallback_top_w)
    bot_w = float(p.fallback_bottom_w)
    top_y = float(p.fallback_top_y)
    bot_y = float(p.fallback_bottom_y)

    x1 = cx - top_w / 2.0
    x2 = cx + top_w / 2.0
    x4 = cx - bot_w / 2.0
    x3 = cx + bot_w / 2.0
    return DangerZonePct(points=[(x1, top_y), (x2, top_y), (x3, bot_y), (x4, bot_y)]).clamp()


def _weighted_avg(items: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if not items:
        return None
    sw = sum(wt for _, _, wt in items)
    if sw <= 1e-6:
        return None
    m = sum(mv * wt for mv, _, wt in items) / sw
    b = sum(bv * wt for _, bv, wt in items) / sw
    return (float(m), float(b))


def _x_at_y(m: float, b: float, y: float) -> float:
    # y = m*x + b  => x = (y - b) / m
    if abs(m) < 1e-6:
        return float("nan")
    return (y - b) / m


def estimate_danger_zone_pct(frame_bgr: np.ndarray, params: AutoRoiParams | None = None) -> DangerZonePct:
    """
    Estimate a danger-zone trapezoid (in percents) that roughly covers the road corridor.
    """
    p = params or AutoRoiParams()
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return _fallback_trapezoid(p)

    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return _fallback_trapezoid(p)

    # Preprocess
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, p.canny1, p.canny2)

    # Keep only bottom region
    y_cut = int(round(h * p.analyze_top_frac))
    if 0 < y_cut < h:
        edges[:y_cut, :] = 0
    # Also crop left/right margins to avoid sidewalks/skyline clutter
    xm = int(round(float(w) * float(max(0.0, min(0.45, p.analyze_x_margin_frac)))))
    if xm > 0 and xm * 2 < w:
        edges[:, :xm] = 0
        edges[:, w - xm :] = 0

    # Optional: emphasize typical lane/edge markings (white/yellow) in the analyzed region.
    if p.use_lane_color_mask:
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(hls)
        # White-ish (high L, low S)
        white = cv2.inRange(L, 190, 255) & cv2.inRange(S, 0, 90)
        # Yellow-ish (H in ~[12..45], reasonably saturated and bright)
        yellow = cv2.inRange(H, 12, 45) & cv2.inRange(S, 70, 255) & cv2.inRange(L, 80, 255)
        mask = cv2.bitwise_or(white, yellow)
        if 0 < y_cut < h:
            mask[:y_cut, :] = 0
        if xm > 0 and xm * 2 < w:
            mask[:, :xm] = 0
            mask[:, w - xm :] = 0
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, mask)

    min_line_len = max(10, int(round(w * p.min_line_len_frac)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(p.hough_threshold),
        minLineLength=min_line_len,
        maxLineGap=int(p.max_line_gap),
    )
    if lines is None or len(lines) == 0:
        return _fallback_trapezoid(p)

    # Collect candidate lane-ish lines: y = m*x + b
    left: list[tuple[float, float, float]] = []  # (m, b, weight)
    right: list[tuple[float, float, float]] = []  # (m, b, weight)
    for (x1, y1, x2, y2) in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        m = dy / dx
        if abs(m) < p.min_abs_slope:
            continue
        b = float(y1) - m * float(x1)
        weight = float(np.hypot(dx, dy))
        if weight <= 1e-3:
            continue
        if m < 0:
            left.append((m, b, weight))
        else:
            right.append((m, b, weight))

    ab_left = _weighted_avg(left)
    ab_right = _weighted_avg(right)
    if ab_left is None or ab_right is None:
        return _fallback_trapezoid(p)

    mL, bL = ab_left
    mR, bR = ab_right

    # Define corridor bounds at bottom and at ROI-top horizon
    y_bottom = float(h - 1)
    y_top = float(int(round(h * p.roi_top_frac)))
    y_top = max(0.0, min(float(h - 2), y_top))

    xL_b = _x_at_y(mL, bL, y_bottom)
    xL_t = _x_at_y(mL, bL, y_top)
    xR_b = _x_at_y(mR, bR, y_bottom)
    xR_t = _x_at_y(mR, bR, y_top)
    if not np.isfinite([xL_b, xL_t, xR_b, xR_t]).all():
        return _fallback_trapezoid(p)

    # Ensure a consistent left/right ordering at both horizons.
    xL_t, xR_t = (float(xL_t), float(xR_t))
    xL_b, xR_b = (float(xL_b), float(xR_b))
    if xL_t > xR_t:
        xL_t, xR_t = xR_t, xL_t
    if xL_b > xR_b:
        xL_b, xR_b = xR_b, xL_b

    # Quick sanity: corridor must not collapse
    if (xR_t - xL_t) < 5.0 or (xR_b - xL_b) < 5.0:
        return _fallback_trapezoid(p)

    # Expand and clamp
    x_pad = float(w) * float(p.x_pad_frac)

    def clamp_x(v: float) -> float:
        return float(max(0.0, min(float(w - 1), v)))

    x1_t = clamp_x(float(xL_t) - x_pad)
    x2_t = clamp_x(float(xR_t) + x_pad)
    x1_b = clamp_x(float(xL_b) - x_pad)
    x2_b = clamp_x(float(xR_b) + x_pad)
    if x2_t <= x1_t or x2_b <= x1_b:
        return _fallback_trapezoid(p)

    if (x2_b - x1_b) < int(round(float(w) * float(p.min_width_frac))):
        return _fallback_trapezoid(p)

    y1 = float(int(round(y_top)))
    y2 = float(int(round(y_bottom)))
    if y2 <= y1:
        return _fallback_trapezoid(p)

    dz = DangerZonePct(
        points=[
            ((x1_t / float(w)) * 100.0, (y1 / float(h)) * 100.0),
            ((x2_t / float(w)) * 100.0, (y1 / float(h)) * 100.0),
            ((x2_b / float(w)) * 100.0, (y2 / float(h)) * 100.0),
            ((x1_b / float(w)) * 100.0, (y2 / float(h)) * 100.0),
        ]
    ).clamp()
    return dz


