from __future__ import annotations

import numpy as np

from pedblock.core.danger_zone import DangerZonePct

from .impl import estimate_danger_zone_pct as _estimate_impl
from .models import AutoRoiParams


def estimate_danger_zone_pct(frame_bgr: np.ndarray, params: AutoRoiParams | None = None) -> DangerZonePct:
    return _estimate_impl(frame_bgr, params)


