from __future__ import annotations

import cv2
import numpy as np

from .models import RoadAreaParams


def _refine_mask_by_color_cc(
    frame_bgr: np.ndarray, corridor_mask_u8: np.ndarray, p: RoadAreaParams, *, debug: dict[str, np.ndarray] | None
) -> np.ndarray | None:
    """
    Уточнение маски «проезжей части»:
    - обучаем простой цветовой “модель дороги” по нижней центральной полосе (внутри коридора),
    - порогуем весь кадр на похожесть,
    - оставляем крупнейшую компоненту, связанную с низом кадра.
    """
    h, w = frame_bgr.shape[:2]
    if corridor_mask_u8.shape[:2] != (h, w):
        return None

    # Build seed region: bottom band & center strip, restricted by corridor
    band_h = max(10, int(round(float(h) * float(max(0.05, min(0.5, p.seed_bottom_band_frac))))))
    y0 = max(0, h - band_h)
    cxw = float(max(0.10, min(0.90, p.seed_center_width_frac)))
    xw = int(round(float(w) * cxw))
    x0 = max(0, int(round((w - xw) / 2)))
    x1 = min(w, x0 + xw)

    seed_mask = np.zeros((h, w), dtype=np.uint8)
    seed_mask[y0:h, x0:x1] = 255
    seed_mask = cv2.bitwise_and(seed_mask, corridor_mask_u8)
    if debug is not None:
        debug["seed_mask_u8"] = seed_mask.copy()

    ys, xs = np.where(seed_mask > 0)
    if ys.size < 250:
        return None

    # Choose colorspace
    cs = (p.color_space or "lab").strip().lower()
    if cs == "hsv":
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    else:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    samples = img[ys, xs].astype(np.float32)  # Nx3
    # Robust center/scale per channel
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med), axis=0) + 1e-3
    scale = 1.4826 * mad  # ~sigma
    scale = np.maximum(scale, 3.0)  # avoid too sharp thresholds on low-variance scenes

    # distance map: sum of squared z-scores (3 channels)
    imgf = img.astype(np.float32)
    z = (imgf - med.reshape(1, 1, 3)) / scale.reshape(1, 1, 3)
    dist = (z[:, :, 0] ** 2 + z[:, :, 1] ** 2 + z[:, :, 2] ** 2).astype(np.float32)

    thr = float(max(1.0, min(60.0, p.color_dist_thresh)))
    cand = (dist < thr).astype(np.uint8) * 255
    if debug is not None:
        # normalize for visualization (clip to 0..thr*3)
        vmax = max(1.0, float(thr * 3.0))
        vis = np.clip((dist / vmax) * 255.0, 0.0, 255.0).astype(np.uint8)
        debug["color_dist_u8"] = vis
        debug["color_cand_u8"] = cand.copy()

    # Keep only bottom-ish region to avoid leaking into sky/buildings.
    y_cut = int(round(h * float(p.analyze_top_frac)))
    if 0 < y_cut < h:
        cand[:y_cut, :] = 0

    # Also require near corridor (dilate corridor a bit to allow road wideness)
    dil = cv2.dilate(corridor_mask_u8, np.ones((9, 9), np.uint8), iterations=1)
    cand = cv2.bitwise_and(cand, dil)

    # Morphology: close gaps then remove speckles
    ck = int(max(3, p.close_ksize // 2 * 2 + 1))
    ok = int(max(3, p.open_ksize // 2 * 2 + 1))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8), iterations=1)

    # Connected components: pick the largest one that touches the bottom band (road must connect to bottom)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return None

    # labels 0 is background
    min_area = int(round(float(w * h) * float(max(0.001, min(0.5, p.min_cc_area_frac)))))
    bottom_touch = np.zeros(num, dtype=np.uint8)
    # mark labels present at bottom row within corridor (strong prior)
    bottom_y = h - 1
    bottom_xs = np.flatnonzero((corridor_mask_u8[bottom_y, :] > 0) & (cand[bottom_y, :] > 0))
    if bottom_xs.size > 0:
        bottom_labels = labels[bottom_y, bottom_xs]
        bottom_touch[bottom_labels] = 1

    best = -1
    best_area = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if bottom_touch[lab] == 0:
            continue
        if area > best_area:
            best_area = area
            best = lab
    if best < 0:
        return None

    out = (labels == best).astype(np.uint8) * 255
    # Final clamp by dilated corridor (safety)
    out = cv2.bitwise_and(out, dil)
    if debug is not None:
        debug["cc_selected_u8"] = out.copy()
    return out


