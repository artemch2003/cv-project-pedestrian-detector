from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal, TypeAlias

__all__ = [
    "RoiPct",
    "DangerZoneMode",
    "DetectionConfig",
    "ExportConfig",
]

DangerZoneMode: TypeAlias = Literal["roi", "road"]


@dataclass(slots=True)
class RoiPct:
    """ROI в процентах от размера кадра."""

    x: float = 35.0
    y: float = 55.0
    w: float = 30.0
    h: float = 40.0

    def clamp(self) -> "RoiPct":
        """Нормализует ROI в диапазон 0..100 и гарантирует ненулевые размеры."""
        self.x = max(0.0, min(100.0, self.x))
        self.y = max(0.0, min(100.0, self.y))
        self.w = max(1.0, min(100.0 - self.x, self.w))
        self.h = max(1.0, min(100.0 - self.y, self.h))
        return self


@dataclass(slots=True)
class DetectionConfig:
    """Параметры детекции (модель/порог/ROI/режим danger-zone)."""

    model_name: str = "yolo11n.pt"  # fallback to yolo8n.pt if not available
    device: str = "auto"  # auto|cpu|mps|cuda
    conf: float = 0.30
    iou: float = 0.45
    max_det: int = 50
    min_area_pct: float = 0.20  # bbox area must be >= X% of frame area to count as "relevant"
    roi: RoiPct = field(default_factory=RoiPct)
    # danger_zone source:
    # - "roi": manual rectangle (converted to quad)
    # - "road": build danger_zone from detected road (drivable area) mask
    danger_zone_mode: DangerZoneMode = "roi"
    # overlay drivable-area mask on preview/export frames (if available)
    show_road_mask: bool = False


@dataclass(slots=True)
class ExportConfig:
    """Параметры экспорта результатов обработки."""

    export_video: bool = False
    export_json: bool = True
    out_dir: str = ""


