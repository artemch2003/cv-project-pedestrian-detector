from __future__ import annotations

import cv2
import numpy as np

from pedblock.core.danger_zone import DangerZonePct

from .models import RoadAreaParams


def danger_zone_pct_from_road_mask(
    mask_u8: np.ndarray,
    params: RoadAreaParams | None = None,
    *,
    near_bottom_frac: float | None = None,
) -> DangerZonePct | None:
    """
    Строит danger_zone (трапецию) по маске проезжей части.
    """
    if mask_u8 is None or getattr(mask_u8, "size", 0) == 0:
        return None
    h, w = mask_u8.shape[:2]
    if w <= 1 or h <= 1:
        return None

    p = params or RoadAreaParams()
    if near_bottom_frac is None:
        near_bottom_frac = float(p.dz_near_bottom_frac)
    near_bottom_frac = float(max(0.05, min(0.95, near_bottom_frac)))
    band_h = int(round(h * near_bottom_frac))
    y1 = max(0, h - band_h)
    y2 = h - 1
    if y2 <= y1:
        return None

    # 0) Cleanup for stability: fill small holes + keep only component connected to the bottom.
    try:
        m = (mask_u8 > 0).astype(np.uint8) * 255
        ck = int(max(3, int(p.dz_close_ksize) // 2 * 2 + 1))  # odd >=3
        ok = int(max(3, int(p.dz_open_ksize) // 2 * 2 + 1))  # odd >=3
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8), iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            bottom_y = h - 1
            bottom_labels = labels[bottom_y, :]
            present = np.unique(bottom_labels[bottom_labels > 0])
            best = -1
            best_area = 0
            for lab in present.tolist():
                area = int(stats[int(lab), cv2.CC_STAT_AREA])
                if area > best_area:
                    best_area = area
                    best = int(lab)
            if best > 0:
                m = (labels == best).astype(np.uint8) * 255
        m_u8 = m
    except Exception:
        m_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    # helper: robust left/right estimation at a y (use a small window + quantiles)
    def lr_at(y: int, win: int = 8, *, q: float) -> tuple[float, float] | None:
        ys = range(max(y1, y - win), min(y2, y + win) + 1)
        lefts: list[float] = []
        rights: list[float] = []
        for yy in ys:
            xs = np.flatnonzero(m_u8[yy, :] > 0)
            if xs.size < 40:
                continue
            # Use quantiles to ignore small noisy islands near edges.
            lq = float(np.quantile(xs.astype(np.float32), q))
            rq = float(np.quantile(xs.astype(np.float32), 1.0 - q))
            if rq <= lq:
                continue
            lefts.append(lq)
            rights.append(rq)
        if len(lefts) < 3 or len(rights) < 3:
            return None
        return (float(np.median(lefts)), float(np.median(rights)))

    q = float(max(0.0, min(0.20, float(p.dz_edge_quantile))))
    method = str(getattr(p, "dz_method", "fit") or "fit").strip().lower()
    if method in ("max_width", "maxwidth", "max-width", "max"):
        best_y: int | None = None
        best_lr: tuple[float, float] | None = None
        best_w: float = -1.0
        stride = 2
        for yy in range(int(y1), int(y2) + 1, stride):
            lr = lr_at(int(yy), q=q)
            if lr is None:
                continue
            xl, xr = lr
            ww = float(xr - xl)
            if (ww > best_w + 1e-6) or (abs(ww - best_w) <= 1e-6 and (best_y is None or int(yy) > int(best_y))):
                best_w = ww
                best_y = int(yy)
                best_lr = (float(xl), float(xr))
        if best_y is None or best_lr is None or best_w <= 0:
            return None

        y_top = float(y1)
        y_bot = float(y2)
        x1_b, x2_b = best_lr

        lr0 = lr_at(int(y1), q=q)
        if lr0 is not None:
            x1_t, x2_t = float(lr0[0]), float(lr0[1])
        else:
            target_w = float(best_w) * 0.60
            cx0 = 0.5 * float(x1_b + x2_b)
            x1_t = cx0 - 0.5 * target_w
            x2_t = cx0 + 0.5 * target_w
    elif method in ("hough", "lines", "mask_hough", "mask-lines"):
        y_top = float(y1)
        y_bot = float(y2)

        lr0 = lr_at(int(y1), q=q)
        if lr0 is not None:
            x1_t, x2_t = float(lr0[0]), float(lr0[1])
        else:
            lr_mid = lr_at(int(round((y1 + y2) / 2.0)), q=q)
            if lr_mid is None:
                return None
            mid_l, mid_r = float(lr_mid[0]), float(lr_mid[1])
            mid_w = max(10.0, float(mid_r - mid_l))
            cx0 = 0.5 * (mid_l + mid_r)
            x1_t = cx0 - 0.5 * (mid_w * 0.70)
            x2_t = cx0 + 0.5 * (mid_w * 0.70)

        edges = cv2.Canny((m_u8 > 0).astype(np.uint8) * 255, 60, 180)
        if 0 < int(y1) < h:
            edges[: int(y1), :] = 0
        if 0 <= int(y2) < h - 1:
            edges[int(y2) + 1 :, :] = 0

        min_line_len = max(20, int(round(float(w) * float(max(0.03, min(0.25, p.min_line_len_frac))))))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(max(20, int(p.hough_threshold * 0.6))),
            minLineLength=int(min_line_len),
            maxLineGap=int(max(5, int(p.max_line_gap))),
        )
        if lines is None or len(lines) == 0:
            return None

        left: list[tuple[float, float, float]] = []  # (a, b, wt)
        right: list[tuple[float, float, float]] = []  # (a, b, wt)
        min_abs_a = 0.10
        yb = float(y_bot)
        cx = float(w) * 0.5
        for (x1s, y1s, x2s, y2s) in lines.reshape(-1, 4):
            dx = float(x2s - x1s)
            dy = float(y2s - y1s)
            if abs(dy) < 1e-6:
                continue
            a = dx / dy
            if abs(a) < min_abs_a:
                continue
            b = float(x1s) - a * float(y1s)
            wt = float(np.hypot(dx, dy))
            if wt <= 1e-3:
                continue

            xb = a * yb + b
            if xb < cx:
                left.append((a, b, wt))
            else:
                right.append((a, b, wt))

        def wavg(items: list[tuple[float, float, float]]) -> tuple[float, float] | None:
            if not items:
                return None
            sw = sum(wt for _, _, wt in items)
            if sw <= 1e-6:
                return None
            a_ = sum(av * wt for av, _, wt in items) / sw
            b_ = sum(bv * wt for _, bv, wt in items) / sw
            return float(a_), float(b_)

        abL = wavg(left)
        abR = wavg(right)
        if abL is None or abR is None:
            return None

        aL, bL = abL
        aR, bR = abR

        bL = float(x1_t) - float(aL) * float(y_top)
        bR = float(x2_t) - float(aR) * float(y_top)

        x1_b = float(aL) * float(y_bot) + float(bL)
        x2_b = float(aR) * float(y_bot) + float(bR)
        if x2_t <= x1_t or x2_b <= x1_b:
            return None
    else:
        frs = getattr(p, "dz_sample_fracs", (0.10, 0.35, 0.65, 0.95))
        frs = tuple(float(max(0.0, min(1.0, f))) for f in frs)
        ys_samp = [int(round(y1 + f * (y2 - y1))) for f in frs]
        pts_l: list[tuple[float, float]] = []
        pts_r: list[tuple[float, float]] = []
        for yy in ys_samp:
            lr = lr_at(int(yy), q=q)
            if lr is None:
                continue
            xl, xr = lr
            pts_l.append((float(yy), float(xl)))
            pts_r.append((float(yy), float(xr)))
        if len(pts_l) < 2 or len(pts_r) < 2:
            return None

        def fit_line_yx(samples: list[tuple[float, float]]) -> tuple[float, float]:
            Y = np.array([s[0] for s in samples], dtype=np.float32)
            X = np.array([s[1] for s in samples], dtype=np.float32)
            A = np.stack([Y, np.ones_like(Y)], axis=1)
            coef, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
            a = float(coef[0])
            b = float(coef[1])
            return a, b

        aL, bL = fit_line_yx(pts_l)
        aR, bR = fit_line_yx(pts_r)

        y_top = float(y1)
        y_bot = float(y2)
        x1_t = aL * y_top + bL
        x2_t = aR * y_top + bR
        x1_b = aL * y_bot + bL
        x2_b = aR * y_bot + bR
        if x2_t <= x1_t or x2_b <= x1_b:
            return None

    def cx(v: float) -> float:
        return float(max(0.0, min(float(w - 1), v)))

    x1_t, x2_t, x1_b, x2_b = cx(x1_t), cx(x2_t), cx(x1_b), cx(x2_b)
    if (x2_b - x1_b) < max(20.0, float(p.dz_min_width_frac) * float(w)):
        return None

    def xp(x: float) -> float:
        return (float(x) / float(w)) * 100.0

    def yp(y: float) -> float:
        return (float(y) / float(h)) * 100.0

    return DangerZonePct(
        points=[
            (xp(x1_t), yp(float(y_top))),
            (xp(x2_t), yp(float(y_top))),
            (xp(x2_b), yp(float(y_bot))),
            (xp(x1_b), yp(float(y_bot))),
        ]
    ).clamp()


