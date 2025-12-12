from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int


def probe_video(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoMeta(path=path, fps=fps, width=width, height=height, frame_count=frame_count)
    finally:
        cap.release()


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    return cap


def make_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    # mp4v works on macOS out-of-the-box most of the time
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps) if fps > 0 else 25.0, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter: {path}")
    return writer


