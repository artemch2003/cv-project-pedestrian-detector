"""
Выделение проезжей части (drivable area) и построение danger-zone по маске.

Это пакет-рефакторинг бывшего модуля `pedblock.core.road_area`.
Публичный API сохранён: внешние импорты вида
`from pedblock.core.road_area import ...` продолжат работать.
"""

from .models import RoadAreaParams, RoadDebug, RoadMaskResult
from .api import estimate_road_mask, estimate_road_mask_debug
from .danger_zone import danger_zone_pct_from_road_mask

__all__ = [
    "RoadAreaParams",
    "RoadDebug",
    "RoadMaskResult",
    "estimate_road_mask",
    "estimate_road_mask_debug",
    "danger_zone_pct_from_road_mask",
]


