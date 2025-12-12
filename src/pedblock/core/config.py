from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class RoiPct:
    """ROI in percents of frame size."""

    x: float = 35.0
    y: float = 55.0
    w: float = 30.0
    h: float = 40.0

    def clamp(self) -> "RoiPct":
        self.x = max(0.0, min(100.0, self.x))
        self.y = max(0.0, min(100.0, self.y))
        self.w = max(1.0, min(100.0 - self.x, self.w))
        self.h = max(1.0, min(100.0 - self.y, self.h))
        return self


@dataclass(slots=True)
class DetectionConfig:
    model_name: str = "yolo11n.pt"  # fallback to yolo8n.pt if not available
    device: str = "auto"  # auto|cpu|mps|cuda
    conf: float = 0.30
    iou: float = 0.45
    max_det: int = 50
    min_area_pct: float = 0.20  # bbox area must be >= X% of frame area to count as "relevant"
    roi: RoiPct = field(default_factory=RoiPct)


@dataclass(slots=True)
class ExportConfig:
    export_video: bool = False
    export_json: bool = True
    out_dir: str = ""


