"""
Автооценка danger-zone (коридора дороги) по одному кадру.

Это пакет-рефакторинг бывшего модуля `pedblock.core.auto_roi`.
Публичный API сохранён.
"""

from .models import AutoRoiParams
from .api import estimate_danger_zone_pct

__all__ = [
    "AutoRoiParams",
    "estimate_danger_zone_pct",
]


