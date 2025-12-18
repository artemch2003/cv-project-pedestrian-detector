from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pedblock.core.types import Box

__all__ = [
    "Detector",
]


class Detector(ABC):
    @abstractmethod
    def detect_persons(self, bgr_frame: np.ndarray) -> list[Box]:
        raise NotImplementedError


