from __future__ import annotations

import numpy as np

from pedblock.core.camera_calib import DEFAULT_CALIB_PATH, load_camera_calib

from .corridor import _estimate_corridor_from_lines
from .fallback import _fallback_mask
from .models import RoadAreaParams, RoadDebug, RoadMaskResult
from .refine import _refine_mask_by_color_cc


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
                # In BEV, rely on color+CC connectivity from the bottom band.
                corridor_all = np.ones(frame_bev.shape[:2], dtype=np.uint8) * 255
                refined_bev = _refine_mask_by_color_cc(frame_bev, corridor_all, p, debug=None)
                if refined_bev is not None:
                    mask_back = bev.warp_mask_to_frame(refined_bev)
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


