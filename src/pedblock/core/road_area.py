from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pedblock.core.camera_calib import DEFAULT_CALIB_PATH, load_camera_calib
from pedblock.core.danger_zone import DangerZonePct


@dataclass(frozen=True, slots=True)
class RoadAreaParams:
    """
    Параметры детекции проезжей части (drivable area) из одного кадра.

    Идея (лёгкая эвристика, без нейросети):
    - берём нижнюю часть кадра (где обычно дорога),
    - находим сильные границы/разметку (Canny),
    - выделяем кандидаты линий (HoughLinesP),
    - оставляем две доминирующие линии: «левая» (наклон < 0) и «правая» (наклон > 0),
    - строим полигон-коридор между ними и заливаем его => это и есть маска проезжей части.
    """

    # Анализируем только низ (в процентах высоты) — вверх обычно не относится к дороге.
    analyze_top_frac: float = 0.40  # 0..1 (0.40 => игнорируем верхние 40%)
    analyze_x_margin_frac: float = 0.06  # 0..0.45 обрезка слева/справа (шумы: обочины/бордюр/небо)

    # Горизонт начала «коридора» (где полигон сужается вверх)
    corridor_top_frac: float = 0.60

    # Canny / Hough
    canny1: int = 50
    canny2: int = 150
    hough_threshold: int = 60
    min_line_len_frac: float = 0.06  # доля ширины
    max_line_gap: int = 30

    # Фильтр линий по наклону
    min_abs_slope: float = 0.55

    # Немного расширяем коридор
    x_pad_frac: float = 0.05

    # Если коридор получается слишком узкий — fallback
    min_width_frac: float = 0.22

    # Фоллбек-трапеция (проценты)
    fallback_top_y: float = 60.0
    fallback_bottom_y: float = 98.0
    fallback_top_w: float = 40.0
    fallback_bottom_w: float = 80.0
    fallback_center_x: float = 50.0

    # Усиливаем разметку (белый/жёлтый) — помогает на «обычных» видеорегистраторах
    use_lane_color_mask: bool = True

    # Режим сегментации:
    # - "hough": только коридор между 2 линиями (быстро)
    # - "hybrid": коридор (геометрия) + цветовая сегментация + CC (обычно точнее)
    method: str = "hybrid"

    # OpenADAS-style: если есть калибровка гомографии, можно строить маску в bird-view и
    # проецировать обратно на исходный кадр. Это делает выделение дороги стабильнее на
    # видео с сильной перспективой (как в статье OpenADAS).
    use_perspective: bool = False
    calib_path: str = DEFAULT_CALIB_PATH

    # Настройки цветовой сегментации (hybrid)
    color_space: str = "lab"  # lab|hsv
    # Откуда берём “эталонный цвет дороги”: полоса у низа кадра
    seed_bottom_band_frac: float = 0.14  # доля высоты (снизу)
    seed_center_width_frac: float = 0.35  # доля ширины (по центру)
    # Порог “похожести на дорогу” (чем меньше — тем строже)
    # В LAB: это порог для нормированного квадр. расстояния (по каналам).
    color_dist_thresh: float = 7.5
    # Морфология после порога
    close_ksize: int = 11
    open_ksize: int = 5
    # Минимальная площадь компоненты (в долях кадра) для принятия маски
    min_cc_area_frac: float = 0.03

    # ==== DangerZone-from-mask tuning ====
    # Какая часть маски (снизу) используется для построения трапеции danger_zone.
    dz_near_bottom_frac: float = 0.35
    # Насколько “отрезать” края маски (квантили). 0.06 => берём 6..94% по X,
    # чтобы игнорировать шумные островки у границ.
    dz_edge_quantile: float = 0.06
    # Доп. морфология перед извлечением границ danger_zone (устойчивость на дожде/бликах).
    dz_close_ksize: int = 15
    dz_open_ksize: int = 5
    # Минимальная ширина трапеции на ближней границе (доля кадра).
    dz_min_width_frac: float = 0.18
    # Позиции (фракции) внутри near-band по Y, по которым оцениваем границы и фитчим линии.
    dz_sample_fracs: tuple[float, float, float, float] = (0.10, 0.35, 0.65, 0.95)


@dataclass(frozen=True, slots=True)
class RoadMaskResult:
    mask_u8: np.ndarray  # HxW uint8 {0,255}
    polygon_px: list[tuple[int, int]]  # по часовой: TL, TR, BR, BL (если удалось)


@dataclass(frozen=True, slots=True)
class RoadDebug:
    edges_u8: np.ndarray | None = None
    corridor_mask_u8: np.ndarray | None = None
    seed_mask_u8: np.ndarray | None = None
    color_dist_u8: np.ndarray | None = None  # 0..255 (меньше = ближе к “дороге”)
    color_cand_u8: np.ndarray | None = None
    cc_selected_u8: np.ndarray | None = None


def estimate_road_mask(frame_bgr: np.ndarray, params: RoadAreaParams | None = None) -> RoadMaskResult:
    p = params or RoadAreaParams()
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return _fallback_mask(frame_bgr, p)

    h, w = frame_bgr.shape[:2]
    if w <= 1 or h <= 1:
        return _fallback_mask(frame_bgr, p)

    # Optional perspective mode (OpenADAS-style): compute dense road mask in bird-view
    # and warp it back to the original image coordinates.
    if bool(getattr(p, "use_perspective", False)):
        calib = load_camera_calib(getattr(p, "calib_path", DEFAULT_CALIB_PATH))
        if calib is not None:
            try:
                bev = calib.build_birdeye(frame_w=w, frame_h=h)
                frame_bev = bev.warp_to_bev(frame_bgr)
                # In BEV, the "lane line" slopes may become vertical, so avoid Hough-corridor
                # and rely on color+CC connectivity from the bottom band.
                corridor_all = np.ones(frame_bev.shape[:2], dtype=np.uint8) * 255
                refined_bev = _refine_mask_by_color_cc(frame_bev, corridor_all, p, debug=None)
                if refined_bev is not None:
                    mask_back = bev.warp_mask_to_frame(refined_bev)
                    # ensure binary-ish u8 {0,255}
                    mask_back = (mask_back > 0).astype(np.uint8) * 255
                    return RoadMaskResult(mask_u8=mask_back, polygon_px=[])
            except Exception:
                # fail open: fallback to non-perspective mode
                pass

    method = (p.method or "hybrid").strip().lower()
    if method not in ("hybrid", "hough"):
        method = "hybrid"

    # 1) Build a geometric corridor by lines (always our strongest geometric prior).
    corridor = _estimate_corridor_from_lines(frame_bgr, p, debug=None)
    if corridor is None:
        # If line-based corridor failed (rain, glare, no markings),
        # try "color-only" drivable area as a fallback (still adaptive).
        if method == "hybrid":
            h, w = frame_bgr.shape[:2]
            corridor_all = np.ones((h, w), dtype=np.uint8) * 255
            refined = _refine_mask_by_color_cc(frame_bgr, corridor_all, p, debug=None)
            if refined is not None:
                return RoadMaskResult(mask_u8=refined, polygon_px=[])
        return _fallback_mask(frame_bgr, p)
    corridor_mask, poly = corridor

    if method == "hough":
        return RoadMaskResult(mask_u8=corridor_mask, polygon_px=poly)

    # 2) Hybrid: refine to a dense drivable-area mask using color similarity + connected components.
    refined = _refine_mask_by_color_cc(frame_bgr, corridor_mask, p, debug=None)
    if refined is None:
        # if refinement failed, return corridor-only (still usable)
        return RoadMaskResult(mask_u8=corridor_mask, polygon_px=poly)
    return RoadMaskResult(mask_u8=refined, polygon_px=poly)


def estimate_road_mask_debug(frame_bgr: np.ndarray, params: RoadAreaParams | None = None) -> tuple[RoadMaskResult, RoadDebug]:
    """
    Как estimate_road_mask, но дополнительно возвращает промежуточные карты для отладки в UI.
    """
    p = params or RoadAreaParams()
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        fb = _fallback_mask(frame_bgr, p)
        return fb, RoadDebug()

    h, w = frame_bgr.shape[:2]
    if w <= 1 or h <= 1:
        fb = _fallback_mask(frame_bgr, p)
        return fb, RoadDebug()

    # Perspective mode debug: return BEV intermediates as best-effort grayscale maps.
    if bool(getattr(p, "use_perspective", False)):
        calib = load_camera_calib(getattr(p, "calib_path", DEFAULT_CALIB_PATH))
        if calib is not None:
            try:
                bev = calib.build_birdeye(frame_w=w, frame_h=h)
                frame_bev = bev.warp_to_bev(frame_bgr)
                corridor_all = np.ones(frame_bev.shape[:2], dtype=np.uint8) * 255
                store: dict[str, np.ndarray] = {}
                refined_bev = _refine_mask_by_color_cc(frame_bev, corridor_all, p, debug=store)
                if refined_bev is not None:
                    mask_back = bev.warp_mask_to_frame(refined_bev)
                    mask_back = (mask_back > 0).astype(np.uint8) * 255
                    dbg = RoadDebug(
                        edges_u8=None,
                        corridor_mask_u8=None,
                        seed_mask_u8=store.get("seed_mask_u8"),
                        color_dist_u8=store.get("color_dist_u8"),
                        color_cand_u8=store.get("color_cand_u8"),
                        cc_selected_u8=store.get("cc_selected_u8"),
                    )
                    return RoadMaskResult(mask_u8=mask_back, polygon_px=[]), dbg
            except Exception:
                pass

    method = (p.method or "hybrid").strip().lower()
    if method not in ("hybrid", "hough"):
        method = "hybrid"

    dbg = RoadDebug()
    corridor = _estimate_corridor_from_lines(frame_bgr, p, debug={"edges": True})
    if corridor is None:
        # Debug: try the adaptive "color-only" fallback before giving up to a static trapezoid.
        if method != "hough":
            h, w = frame_bgr.shape[:2]
            corridor_all = np.ones((h, w), dtype=np.uint8) * 255
            store: dict[str, np.ndarray] = {}
            refined = _refine_mask_by_color_cc(frame_bgr, corridor_all, p, debug=store)
            if refined is not None:
                dbg = RoadDebug(
                    edges_u8=None,
                    corridor_mask_u8=None,
                    seed_mask_u8=store.get("seed_mask_u8"),
                    color_dist_u8=store.get("color_dist_u8"),
                    color_cand_u8=store.get("color_cand_u8"),
                    cc_selected_u8=store.get("cc_selected_u8"),
                )
                return RoadMaskResult(mask_u8=refined, polygon_px=[]), dbg
        fb = _fallback_mask(frame_bgr, p)
        return fb, dbg
    corridor_mask, poly, edges_u8 = corridor
    dbg = RoadDebug(edges_u8=edges_u8, corridor_mask_u8=corridor_mask)

    if method == "hough":
        return RoadMaskResult(mask_u8=corridor_mask, polygon_px=poly), dbg

    debug_store: dict[str, np.ndarray] = {}
    refined = _refine_mask_by_color_cc(frame_bgr, corridor_mask, p, debug=debug_store)
    if refined is None:
        return RoadMaskResult(mask_u8=corridor_mask, polygon_px=poly), dbg

    dbg = RoadDebug(
        edges_u8=edges_u8,
        corridor_mask_u8=corridor_mask,
        seed_mask_u8=debug_store.get("seed_mask_u8"),
        color_dist_u8=debug_store.get("color_dist_u8"),
        color_cand_u8=debug_store.get("color_cand_u8"),
        cc_selected_u8=debug_store.get("cc_selected_u8"),
    )
    return RoadMaskResult(mask_u8=refined, polygon_px=poly), dbg


def _estimate_corridor_from_lines(
    frame_bgr: np.ndarray, p: RoadAreaParams, *, debug: dict | None
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray] | tuple[np.ndarray, list[tuple[int, int]]] | None:
    """
    Строит грубую маску-коридор (трапеция) между левой/правой «дорожными» линиями.
    Возвращает (mask_u8, polygon_px).
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


def danger_zone_pct_from_road_mask(
    mask_u8: np.ndarray,
    params: RoadAreaParams | None = None,
    *,
    near_bottom_frac: float | None = None,
) -> DangerZonePct | None:
    """
    Строит danger_zone (трапецию) по маске проезжей части.

    Логика:
    - берём только «ближнюю» часть дороги (нижние near_bottom_frac кадра),
    - на верхней и нижней границе этой полосы оцениваем левую/правую границу дороги,
      беря медиану по нескольким соседним строкам (устойчивее к шуму),
    - возвращаем трапецию (DangerZonePct), которую потом используем для проверки bbox-центра.
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
        ok = int(max(3, int(p.dz_open_ksize) // 2 * 2 + 1))   # odd >=3
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

    # Sample multiple y positions and smooth by fitting left/right as a function of y.
    frs = getattr(p, "dz_sample_fracs", (0.10, 0.35, 0.65, 0.95))
    frs = tuple(float(max(0.0, min(1.0, f))) for f in frs)
    q = float(max(0.0, min(0.20, float(p.dz_edge_quantile))))
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

    # Fit x = a*y + b (least squares) for left and right boundaries.
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

    y_top = float(min(ys_samp))
    y_bot = float(max(ys_samp))
    x1_t = aL * y_top + bL
    x2_t = aR * y_top + bR
    x1_b = aL * y_bot + bL
    x2_b = aR * y_bot + bR
    if x2_t <= x1_t or x2_b <= x1_b:
        return None

    # Clamp + minimal width sanity relative to frame
    def cx(v: float) -> float:
        return float(max(0.0, min(float(w - 1), v)))

    x1_t, x2_t, x1_b, x2_b = cx(x1_t), cx(x2_t), cx(x1_b), cx(x2_b)
    if (x2_b - x1_b) < max(20.0, float(p.dz_min_width_frac) * float(w)):
        return None

    # convert to percent trapezoid
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


