from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AutoRoiParams:
    """Heuristics for estimating a 'danger zone' (road corridor) ROI from a frame."""

    # Only analyze the bottom part of the frame (where the road is usually visible)
    analyze_top_frac: float = 0.40  # 0..1 (0.40 => ignore top 40%)
    analyze_x_margin_frac: float = 0.06  # 0..0.45 crop left/right margins to reduce irrelevant edges

    # Horizon for danger-zone top (where the projected zone begins)
    roi_top_frac: float = 0.60  # 0..1 (0.60 => zone starts at 60% of height)

    # Canny / Hough params (fairly conservative defaults)
    canny1: int = 50
    canny2: int = 150
    hough_threshold: int = 60
    min_line_len_frac: float = 0.06  # of width
    max_line_gap: int = 30

    # Line filtering
    min_abs_slope: float = 0.55

    # Expand estimated corridor horizontally
    x_pad_frac: float = 0.05  # of width

    # If the corridor is too narrow, fallback to a centered heuristic trapezoid
    min_width_frac: float = 0.22  # of width

    # Center fallback trapezoid (percents)
    fallback_top_y: float = 60.0
    fallback_bottom_y: float = 98.0
    fallback_top_w: float = 40.0
    fallback_bottom_w: float = 80.0
    fallback_center_x: float = 50.0
    # Prefer lane/edge markings (best-effort). Helps on dashcam footage with visible marking.
    use_lane_color_mask: bool = True


