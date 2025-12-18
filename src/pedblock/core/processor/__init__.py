"""
Обработка видео: детекция людей, проверка перекрытия danger-zone, экспорт событий/видео.

Это пакет-рефакторинг бывшего модуля `pedblock.core.processor`.
Публичный API сохранён: `VideoProcessor`, `FrameResult`, `ProcessorProgress`.
"""

from .models import FrameResult, ProcessorProgress, ProcessorStatus
from .video_processor import VideoProcessor

__all__ = [
    "FrameResult",
    "ProcessorProgress",
    "ProcessorStatus",
    "VideoProcessor",
]


