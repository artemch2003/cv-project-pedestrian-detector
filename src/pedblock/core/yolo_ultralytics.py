from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pedblock.core.config import DetectionConfig
from pedblock.core.detector import Detector
from pedblock.core.types import Box


def _normalize_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in {"", "auto"}:
        return ""
    return d


@dataclass(slots=True)
class YoloUltralyticsDetector(Detector):
    cfg: DetectionConfig

    def __post_init__(self) -> None:
        # Lazy import: ultralytics can be heavy; keep startup fast.
        from ultralytics import YOLO  # type: ignore

        model_name = (self.cfg.model_name or "").strip()
        if not model_name:
            model_name = "yolo11n.pt"
        try:
            self._model = YOLO(model_name)
        except Exception:
            # fallback to widely available model name
            self._model = YOLO("yolov8n.pt")

        self._device = _normalize_device(self.cfg.device)

    def detect_persons(self, bgr_frame: np.ndarray) -> list[Box]:
        # Ultralytics expects BGR numpy ok; we keep it as-is for speed.
        # `classes=[0]` -> person in COCO.
        results = self._model.predict(
            source=bgr_frame,
            verbose=False,
            device=self._device,
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            max_det=int(self.cfg.max_det),
            classes=[0],
        )
        if not results:
            return []

        r0: Any = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy
        conf = boxes.conf
        if xyxy is None or conf is None:
            return []

        xyxy_np = xyxy.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()

        out: list[Box] = []
        for (x1, y1, x2, y2), c in zip(xyxy_np, conf_np, strict=False):
            out.append(Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), conf=float(c)))
        return out


