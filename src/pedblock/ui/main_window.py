from __future__ import annotations

"""
`main_window.py` — главное окно приложения (CustomTkinter).

Ответственности `MainWindow`:
- собрать UI (sidebar + превью)
- управлять состояниями UI (выбрано видео / идёт обработка / пауза)
- синхронизировать параметры UI с `pedblock.core.VideoProcessor`
- принимать результаты обработки (кадры/прогресс) и отрисовывать превью

Дизайн:
- тяжелая логика детекции/сегментации/экспорта находится в `pedblock.core`
- UI использует `VideoProcessor` как worker: старт/пауза/стоп + poll результатов
- превью перерисовывается на основе последнего `FrameResult`, а также использует небольшой LRU-кэш
  для повторного просмотра без перерасчётов (маска дороги и auto danger_zone)
"""

from collections import OrderedDict
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from pedblock import __version__ as app_version
from pedblock.core.camera_calib import DEFAULT_CALIB_PATH
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.auto_roi import estimate_danger_zone_pct
from pedblock.core.danger_zone import DangerZonePct, danger_zone_pct_to_px
from pedblock.core.road_area import RoadAreaParams, RoadDebug, danger_zone_pct_from_road_mask, estimate_road_mask, estimate_road_mask_debug
from pedblock.core.processor import FrameResult, ProcessorProgress, VideoProcessor
from pedblock.core.types import FrameInfo
from pedblock.core.annotate import draw_annotations
from pedblock.ui.perspective_calib_dialog import open_perspective_calib_dialog
from pedblock.ui.sidebar import build_sidebar
from pedblock.ui.preview_panel import build_preview_panel


class MainWindow(ctk.CTk):
    """
    Главное окно приложения.

    Базовый цикл работы:
    - пользователь выбирает видео и настраивает параметры
    - `Старт` запускает `VideoProcessor` (внутри — фоновый поток/воркер)
    - `MainWindow._tick()` периодически забирает события из `processor.poll()`:
      прогресс и результаты по кадрам, затем обновляет UI.

    Превью:
    - основной кадр рисуется аннотациями (`draw_annotations`)
    - опционально отображается маска дороги (оверлей + отдельная панель)
    - изображения кэшируются в виде BGR-матриц, а при ресайзе окна выполняется пересчёт
      `CTkImage` (debounce через `_schedule_preview_rescale()`).
    """

    _AUTO_ROI_ALPHA = 0.30
    _AUTO_ROI_EVERY_N_MIN = 1
    _AUTO_ROI_EVERY_N_MAX = 60

    def __init__(self) -> None:
        """Создаёт окно, инициализирует состояние, строит UI и запускает цикл `_tick()`."""
        super().__init__()
        self.title(f"Pedestrian Blocker v{app_version} — YOLO + CustomTkinter")
        self.geometry("1200x780")
        self.minsize(1000, 680)

        self._video_path: str | None = None
        # Worker, выполняющий обработку видео в фоне и отдающий результаты через poll().
        self._processor = VideoProcessor()
        # Для CTkLabel-превью используем CTkImage (а не PIL.ImageTk.PhotoImage):
        # иначе будут warning'и и проблемы с HiDPI scaling в CustomTkinter.
        self._preview_ctkimg_main: ctk.CTkImage | None = None
        self._preview_ctkimg_mask: ctk.CTkImage | None = None
        # Храним последние отрисованные изображения (BGR), чтобы при ресайзе окна
        # можно было просто «перефитить» картинку без пересчёта детекции/масок.
        self._last_preview_bgr_main: np.ndarray | None = None
        self._last_preview_bgr_mask: np.ndarray | None = None
        self._preview_rescale_after_id: str | None = None
        self._preview_mask_visible: bool = False
        self._preview_right: ctk.CTkFrame | None = None
        self._paused = False
        # Флаг нужен, чтобы trace'ы TkVar не вызывали рекурсивный пересчёт при программной установке ROI.
        self._roi_internal_update = False
        self._last_frame: FrameResult | None = None
        self._roi_value_labels: dict[str, ctk.CTkLabel] = {}
        self._roi_sliders: list[ctk.CTkSlider] = []
        self._auto_roi_last_frame_index: int | None = None
        self._auto_roi_smoothed: DangerZonePct | None = None
        self._auto_roi_every_n_frames_var = tk.IntVar(value=5)
        self._auto_road_mask_u8: np.ndarray | None = None
        self._auto_road_debug: RoadDebug | None = None
        self._road_color_thresh_var = tk.DoubleVar(value=7.5)
        self._road_debug_mode_var = tk.StringVar(value="Обычный")
        self._use_perspective_var = tk.BooleanVar(value=False)
        self._dz_near_bottom_var = tk.DoubleVar(value=35.0)  # %
        self._dz_edge_q_var = tk.DoubleVar(value=6.0)  # %
        self._dz_max_width_var = tk.BooleanVar(value=False)
        self._dz_hough_sides_var = tk.BooleanVar(value=False)

        # Preview cache: allows "повторный просмотр" без пересчёта маски/авто danger_zone.
        # Храним только последние N кадров (LRU), иначе память улетит на длинных видео.
        self._preview_cache_key: tuple[object, ...] | None = None
        self._road_mask_cache_lru: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._auto_dz_cache_lru: "OrderedDict[int, DangerZonePct]" = OrderedDict()
        self._preview_cache_max_items: int = 80

        self._build_ui()
        self._tick()

    def _get_dz_method(self) -> str:
        """Возвращает выбранный метод построения danger_zone по маске дороги."""
        if bool(self._dz_hough_sides_var.get()):
            return "hough"
        if bool(self._dz_max_width_var.get()):
            return "max_width"
        return "fit"

    def _refresh_preview_after_road_param_change(self, *, force_auto_roi: bool = True) -> None:
        """
        Общая логика для слайдеров/чекбоксов road-детекции:
        - если включён авто danger_zone (по маске дороги) — пересчитать,
        - иначе — просто перерисовать превью по последнему кадру.
        """
        try:
            auto_enabled = bool(getattr(self, "roi_auto_var", tk.BooleanVar(value=False)).get())
        except Exception:
            auto_enabled = False
        if auto_enabled:
            self._auto_roi_recompute(force=bool(force_auto_roi))
        else:
            self._render_preview_from_last()

    def _on_show_road_mask_changed(self) -> None:
        """Обработчик чекбокса «Показывать маску…»: переключает runtime-оверлей и перерисовывает превью."""
        if self._processor.is_running():
            self._processor.set_show_road_mask(bool(self.show_road_mask_var.get()))
        # Overlay toggle doesn't require recompute; just redraw.
        self._render_preview_from_last()

    def _on_road_color_thresh_changed(self) -> None:
        """Обработчик слайдера строгости маски (color distance threshold)."""
        self._road_thresh_lbl.configure(text=f"{self._road_color_thresh_var.get():.1f}")
        if self._processor.is_running():
            self._processor.set_road_params(color_dist_thresh=float(self._road_color_thresh_var.get()))
        self._refresh_preview_after_road_param_change(force_auto_roi=True)

    def _on_dz_near_bottom_changed(self) -> None:
        """Обработчик слайдера высоты danger_zone от низа (в % кадра)."""
        self._dz_near_bottom_lbl.configure(text=f"{self._dz_near_bottom_var.get():.0f}")
        if self._processor.is_running():
            self._processor.set_road_params(dz_near_bottom_frac=float(self._dz_near_bottom_var.get()) / 100.0)
        self._refresh_preview_after_road_param_change(force_auto_roi=True)

    def _on_dz_edge_q_changed(self) -> None:
        """Обработчик слайдера отсечения краёв маски (квантили)."""
        self._dz_edge_q_lbl.configure(text=f"{self._dz_edge_q_var.get():.0f}")
        if self._processor.is_running():
            self._processor.set_road_params(dz_edge_quantile=float(self._dz_edge_q_var.get()) / 100.0)
        self._refresh_preview_after_road_param_change(force_auto_roi=True)

    def _on_dz_method_changed(self) -> None:
        """Обработчик смены метода построения зоны (fit/max_width/hough)."""
        if self._processor.is_running():
            self._processor.set_road_params(dz_method=self._get_dz_method())
        self._refresh_preview_after_road_param_change(force_auto_roi=True)

    def _get_preview_cache_key(self) -> tuple[object, ...] | None:
        """
        Формирует ключ, по которому валидируется кэш превью.

        Идея: если меняются параметры, влияющие на маску/зону (road params, перспектива, mtime калибровки),
        то старые кэшированные значения использовать нельзя.
        """
        # Cache is only meaningful when a video is selected.
        if not self._video_path:
            return None
        rp = self._get_road_params()
        calib_mtime = None
        try:
            if bool(rp.use_perspective) and rp.calib_path and os.path.exists(rp.calib_path):
                calib_mtime = float(os.path.getmtime(rp.calib_path))
        except Exception:
            calib_mtime = None
        # Include road params that affect mask/zone; if they change, cache must be invalidated.
        return (
            str(self._video_path),
            float(rp.color_dist_thresh),
            bool(rp.use_perspective),
            float(rp.dz_near_bottom_frac),
            float(rp.dz_edge_quantile),
            str(getattr(rp, "dz_method", "fit") or "fit"),
            calib_mtime,
        )

    def _ensure_preview_cache(self) -> None:
        """Сбрасывает LRU-кэши маски/auto-ROI, если изменился ключ `_get_preview_cache_key()`."""
        key = self._get_preview_cache_key()
        if key != self._preview_cache_key:
            self._preview_cache_key = key
            self._road_mask_cache_lru.clear()
            self._auto_dz_cache_lru.clear()

    def _lru_get_mask(self, frame_index: int) -> np.ndarray | None:
        """Возвращает маску дороги из LRU по frame_index (best-effort)."""
        try:
            m = self._road_mask_cache_lru.get(int(frame_index))
        except Exception:
            return None
        if m is not None:
            try:
                self._road_mask_cache_lru.move_to_end(int(frame_index))
            except Exception:
                pass
        return m

    def _lru_put_mask(self, frame_index: int, mask_u8: np.ndarray) -> None:
        """Кладёт маску дороги в LRU и обрезает кэш до `_preview_cache_max_items`."""
        try:
            self._road_mask_cache_lru[int(frame_index)] = mask_u8
            self._road_mask_cache_lru.move_to_end(int(frame_index))
            while len(self._road_mask_cache_lru) > int(self._preview_cache_max_items):
                self._road_mask_cache_lru.popitem(last=False)
        except Exception:
            # cache is best-effort; never break preview
            pass

    def _lru_get_auto_dz(self, frame_index: int) -> DangerZonePct | None:
        """Возвращает auto danger_zone (в процентах) из LRU по frame_index."""
        try:
            dz = self._auto_dz_cache_lru.get(int(frame_index))
        except Exception:
            return None
        if dz is not None:
            try:
                self._auto_dz_cache_lru.move_to_end(int(frame_index))
            except Exception:
                pass
        return dz

    def _lru_put_auto_dz(self, frame_index: int, dz: DangerZonePct) -> None:
        """Кладёт auto danger_zone в LRU и обрезает кэш."""
        try:
            self._auto_dz_cache_lru[int(frame_index)] = dz
            self._auto_dz_cache_lru.move_to_end(int(frame_index))
            while len(self._auto_dz_cache_lru) > int(self._preview_cache_max_items):
                self._auto_dz_cache_lru.popitem(last=False)
        except Exception:
            pass

    def _set_preview_bgr(self, bgr, *, target: str = "main") -> None:
        """
        Преобразует BGR (OpenCV) -> RGB (PIL) -> CTkImage и подставляет в label.

        Параметр `target`:
        - `"main"`: основное превью
        - `"mask"`: панель маски (если включена)
        """
        lbl = self.preview_main if target == "main" else self.preview_mask
        try:
            if target == "main":
                self._last_preview_bgr_main = bgr
            else:
                self._last_preview_bgr_mask = bgr
        except Exception:
            pass
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit to preview label size (keep aspect ratio)
        w = max(10, lbl.winfo_width())
        h = max(10, lbl.winfo_height())
        img.thumbnail((w, h))

        # customtkinter expects CTkImage for proper HiDPI scaling
        ctkimg = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        if target == "main":
            self._preview_ctkimg_main = ctkimg
        else:
            self._preview_ctkimg_mask = ctkimg
        lbl.configure(image=ctkimg, text="")

    def _schedule_preview_rescale(self) -> None:
        """Делаем debounce для частых `<Configure>` событий при ресайзе окна."""
        # Debounce frequent <Configure> events.
        if self._preview_rescale_after_id is not None:
            try:
                self.after_cancel(self._preview_rescale_after_id)
            except Exception:
                pass
            self._preview_rescale_after_id = None
        self._preview_rescale_after_id = self.after(60, self._rescale_preview_from_cache)

    def _rescale_preview_from_cache(self) -> None:
        """Перерисовывает CTkImage из сохранённых BGR-картинок (без пересчёта детекции/масок)."""
        self._preview_rescale_after_id = None
        if self._last_preview_bgr_main is not None and getattr(self._last_preview_bgr_main, "size", 0) != 0:
            self._set_preview_bgr(self._last_preview_bgr_main, target="main")
        if (
            self._preview_mask_visible
            and self._last_preview_bgr_mask is not None
            and getattr(self._last_preview_bgr_mask, "size", 0) != 0
        ):
            self._set_preview_bgr(self._last_preview_bgr_mask, target="mask")

    def _set_mask_panel_visible(self, visible: bool) -> None:
        """
        Показывает/скрывает правую колонку с маской и подстраивает grid layout.

        Важно: при скрытии маски основной preview должен занимать всю ширину.
        """
        self._preview_mask_visible = bool(visible)
        # When mask panel is hidden, let the main preview span the full width.
        try:
            if self._preview_right is not None:
                if visible:
                    self._preview_right.grid_columnconfigure(0, weight=3)
                    self._preview_right.grid_columnconfigure(1, weight=1, minsize=280)
                    self.preview_main.grid_configure(column=0, columnspan=1, padx=(12, 6))
                    self.preview_mask.grid()
                else:
                    self._preview_right.grid_columnconfigure(0, weight=1)
                    self._preview_right.grid_columnconfigure(1, weight=0, minsize=0)
                    self.preview_mask.grid_remove()
                    self.preview_main.grid_configure(column=0, columnspan=2, padx=12)
        except Exception:
            # Best-effort layout; never break preview rendering
            if visible:
                self.preview_mask.grid()
            else:
                self.preview_mask.grid_remove()
        self._schedule_preview_rescale()

    def _get_road_params(self) -> RoadAreaParams:
        """Собирает `RoadAreaParams` из текущих значений UI-переменных."""
        thr = float(self._road_color_thresh_var.get())
        return RoadAreaParams(
            color_dist_thresh=max(1.0, min(60.0, thr)),
            use_perspective=bool(self._use_perspective_var.get()),
            calib_path=DEFAULT_CALIB_PATH,
            dz_near_bottom_frac=max(0.05, min(0.95, float(self._dz_near_bottom_var.get()) / 100.0)),
            dz_edge_quantile=max(0.0, min(0.20, float(self._dz_edge_q_var.get()) / 100.0)),
            dz_method=self._get_dz_method(),
        )

    def _on_perspective_toggle(self) -> None:
        """Обработчик чекбокса перспективы: синхронизирует параметр с процессором и обновляет превью."""
        if self._processor.is_running():
            self._processor.set_road_params(use_perspective=bool(self._use_perspective_var.get()))
        # perspective changes can affect both ROI-from-road and the overlay
        if bool(self.roi_auto_var.get()):
            self._auto_roi_recompute(force=True)
        else:
            self._render_preview_from_last()

    def _open_perspective_calib_dialog(self) -> None:
        """Открывает модальный диалог калибровки, используя последний доступный кадр превью."""
        fr = self._last_frame
        if fr is None or fr.frame_bgr is None or getattr(fr.frame_bgr, "size", 0) == 0:
            messagebox.showinfo("Калибровка", "Сначала откройте видео и получите кадр (Старт или предпросмотр).")
            return
        open_perspective_calib_dialog(
            self,
            frame_bgr=fr.frame_bgr,
            use_perspective_var=self._use_perspective_var,
            on_perspective_toggle=self._on_perspective_toggle,
            calib_path=DEFAULT_CALIB_PATH,
        )

    def _get_roi_pct_clamped(self) -> RoiPct:
        """Возвращает ROI из UI (в процентах) с применением clamp (границы 0..100 и валидная ширина/высота)."""
        roi = RoiPct(
            x=float(self.roi_x.get()),
            y=float(self.roi_y.get()),
            w=float(self.roi_w.get()),
            h=float(self.roi_h.get()),
        ).clamp()
        return roi

    def _apply_roi_to_vars(self, roi: RoiPct) -> None:
        """Записывает ROI в tk-переменные без запуска рекурсивных trace'ов."""
        # avoid recursive traces
        self._roi_internal_update = True
        try:
            self.roi_x.set(float(roi.x))
            self.roi_y.set(float(roi.y))
            self.roi_w.set(float(roi.w))
            self.roi_h.set(float(roi.h))
        finally:
            self._roi_internal_update = False

    def _set_roi_controls_enabled(self, enabled: bool) -> None:
        """Включает/выключает ручные ROI-слайдеры (когда включён авто-режим — отключаем)."""
        state = "normal" if enabled else "disabled"
        for s in self._roi_sliders:
            try:
                s.configure(state=state)
            except Exception:
                # ignore if a widget doesn't support state (older CTk versions)
                pass

    def _apply_roi_pct(self, roi: RoiPct) -> None:
        """Применяет ROI (ручной режим): обновляет UI, процессор (если запущен) и превью."""
        # manual ROI sliders define a rectangle; keep them, but convert later for processing/drawing
        roi = roi.clamp()
        self._apply_roi_to_vars(roi)
        self._update_roi_value_labels()
        if self._processor.is_running():
            self._processor.set_roi_pct(roi)
        self._render_preview_from_last()

    def _auto_roi_recompute(self, *, force: bool = False) -> None:
        """
        Пересчитывает auto danger_zone (по маске дороги / fallback по линиям) и обновляет превью.

        Параметр `force`:
        - True: пересчитать немедленно (игнорируя throttle и LRU)
        - False: использовать LRU/ограничение частоты, чтобы избежать дрожания и нагрузки
        """
        if not bool(self.roi_auto_var.get()):
            return
        fr = self._last_frame
        if fr is None:
            return
        self._ensure_preview_cache()

        # Fast path: if we already computed auto-zone for this frame_index earlier,
        # reuse it (even if throttle would normally skip) — this is the "повторный просмотр из кэша".
        cached = self._lru_get_auto_dz(fr.frame_info.frame_index)
        if cached is not None and not force:
            self._auto_roi_smoothed = cached
            self._auto_roi_last_frame_index = fr.frame_info.frame_index
            # also refresh cached road mask for overlay/debug if available
            cm = self._lru_get_mask(fr.frame_info.frame_index)
            if cm is not None:
                self._auto_road_mask_u8 = cm
            self._render_preview_from_last()
            return

        # Throttle updates to avoid visible jitter
        if not force:
            last_idx = self._auto_roi_last_frame_index
            every_n = int(self._auto_roi_every_n_frames_var.get() or 1)
            every_n = max(self._AUTO_ROI_EVERY_N_MIN, min(self._AUTO_ROI_EVERY_N_MAX, every_n))
            if last_idx is not None and (fr.frame_info.frame_index - last_idx) < every_n:
                return

        rp = self._get_road_params()
        # 1) Road mask -> danger zone from drivable area (preferred)
        try:
            rm, dbg = estimate_road_mask_debug(fr.frame_bgr, rp)
            self._auto_road_mask_u8 = rm.mask_u8
            self._auto_road_debug = dbg
        except Exception:
            rm = estimate_road_mask(fr.frame_bgr, rp)
            self._auto_road_mask_u8 = rm.mask_u8
            self._auto_road_debug = None

        if self._auto_road_mask_u8 is not None and getattr(self._auto_road_mask_u8, "size", 0) != 0:
            self._lru_put_mask(fr.frame_info.frame_index, self._auto_road_mask_u8)

        est = danger_zone_pct_from_road_mask(self._auto_road_mask_u8, rp)
        # 2) Fallback: old line-based trapezoid (still helpful if mask failed)
        if est is None:
            est = estimate_danger_zone_pct(fr.frame_bgr)

        # Smooth to avoid flicker on noisy lines
        # EMA-сглаживание точек трапеции: уменьшает визуальное "дёргание" зоны.
        alpha = float(self._AUTO_ROI_ALPHA)
        prev = self._auto_roi_smoothed
        if prev is None:
            sm = est
        else:
            # point-wise smoothing (requires stable number/order of points)
            n = min(len(prev.points), len(est.points))
            pts: list[tuple[float, float]] = []
            for i in range(n):
                px, py = prev.points[i]
                ex, ey = est.points[i]
                pts.append(((1 - alpha) * px + alpha * ex, (1 - alpha) * py + alpha * ey))
            sm = DangerZonePct(points=pts).clamp()

        self._auto_roi_smoothed = sm
        self._auto_roi_last_frame_index = fr.frame_info.frame_index
        self._lru_put_auto_dz(fr.frame_info.frame_index, sm)
        # Важно: не отправляем полигон как "ручной override" в процессор, иначе он перестанет
        # строить danger_zone по маске дороги (mode="road") и маска может не попадать в экспорт/оверлей.
        # Переключение режима делается в _on_auto_roi_toggle(), этого достаточно.
        self._render_preview_from_last()

    def _on_auto_roi_toggle(self) -> None:
        """Обработчик чекбокса авто danger_zone: переключает режим в процессоре и пересчитывает зону."""
        enabled = bool(self.roi_auto_var.get())
        self._set_roi_controls_enabled(not enabled)
        if self._processor.is_running():
            self._processor.set_danger_zone_mode("road" if enabled else "roi")
        if enabled:
            # reset smoothing to quickly lock onto the scene
            self._auto_roi_smoothed = None
            self._auto_roi_last_frame_index = None
            self._auto_road_mask_u8 = None
            self._auto_road_debug = None
            self._auto_roi_recompute(force=True)

    def _update_roi_value_labels(self) -> None:
        """Обновляет подписи X/Y/W/H рядом со слайдерами ROI."""
        # show as percents with 0 decimals (sliders are integer-stepped)
        if "X" in self._roi_value_labels:
            self._roi_value_labels["X"].configure(text=f"{self.roi_x.get():.0f}%")
        if "Y" in self._roi_value_labels:
            self._roi_value_labels["Y"].configure(text=f"{self.roi_y.get():.0f}%")
        if "W" in self._roi_value_labels:
            self._roi_value_labels["W"].configure(text=f"{self.roi_w.get():.0f}%")
        if "H" in self._roi_value_labels:
            self._roi_value_labels["H"].configure(text=f"{self.roi_h.get():.0f}%")

    def _render_preview_from_last(self) -> None:
        """
        Перерисовывает превью на основе `self._last_frame`.

        Здесь намеренно нет тяжёлой обработки (YOLO и экспорт живут в `VideoProcessor`).
        Метод:
        - выбирает danger_zone (ручной ROI или auto по дороге)
        - при необходимости строит/берёт road mask (и debug-вид)
        - вызывает `draw_annotations()` и отправляет BGR в `_set_preview_bgr()`
        """
        fr = self._last_frame
        if fr is None:
            return
        self._ensure_preview_cache()
        w = int(fr.frame_info.width)
        h = int(fr.frame_info.height)
        if bool(getattr(self, "roi_auto_var", tk.BooleanVar(value=False)).get()) and self._auto_roi_smoothed is not None:
            dz_px = danger_zone_pct_to_px(self._auto_roi_smoothed, w, h)
        else:
            roi_pct = self._get_roi_pct_clamped()
            # rectangle -> quad
            dz = DangerZonePct.from_quad(
                roi_pct.x,
                roi_pct.y,
                roi_pct.x + roi_pct.w,
                roi_pct.y,
                roi_pct.x + roi_pct.w,
                roi_pct.y + roi_pct.h,
                roi_pct.x,
                roi_pct.y + roi_pct.h,
            ).clamp()
            dz_px = danger_zone_pct_to_px(dz, w, h)

        # recompute obstructing for the preview (so pause + ROI move feels instant)
        # В режиме паузы/при движении ROI хотим мгновенную обратную связь,
        # поэтому пересчитываем obstructing прямо тут (на последнем наборе person bbox).
        frame_area = float(w * h) if w > 0 and h > 0 else 1.0
        min_area = (float(self.min_area_var.get()) / 100.0) * frame_area
        obstructing = False
        for b in fr.persons:
            if b.area < min_area:
                continue
            if dz_px.contains_point(b.cx, b.cy):
                obstructing = True
                break

        # Optional debug view for road segmentation
        # Road Debug: позволяет визуализировать промежуточные карты сегментации дороги.
        view = str(self._road_debug_mode_var.get() or "Обычный")
        base_bgr = fr.frame_bgr
        if view != "Обычный":
            dbg = self._auto_road_debug
            if dbg is None:
                try:
                    _, dbg = estimate_road_mask_debug(fr.frame_bgr, self._get_road_params())
                except Exception:
                    dbg = None
            src = None
            if dbg is not None:
                if view == "Canny":
                    src = dbg.edges_u8
                elif view == "Коридор":
                    src = dbg.corridor_mask_u8
                elif view == "Seed":
                    src = dbg.seed_mask_u8
                elif view == "Dist":
                    src = dbg.color_dist_u8
                elif view == "Candidate":
                    src = dbg.color_cand_u8
                elif view == "Final":
                    src = dbg.cc_selected_u8
            # Если debug-канал не дал нужной карты (часто так бывает, когда refine/CC не сошёлся),
            # то для "Final" показываем фактическую маску, которая используется в зоне.
            if (src is None or getattr(src, "size", 0) == 0) and view == "Final":
                if self._auto_road_mask_u8 is not None and getattr(self._auto_road_mask_u8, "size", 0) != 0:
                    src = self._auto_road_mask_u8
            if src is not None and getattr(src, "size", 0) != 0:
                base_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        road_mask = None
        if bool(getattr(self, "show_road_mask_var", tk.BooleanVar(value=False)).get()):
            # Prefer cached mask for this frame index (repeat playback), then last auto recompute.
            cached_mask = self._lru_get_mask(fr.frame_info.frame_index)
            road_mask = cached_mask if cached_mask is not None else self._auto_road_mask_u8
            if road_mask is None or getattr(road_mask, "size", 0) == 0:
                try:
                    road_mask = estimate_road_mask(fr.frame_bgr, self._get_road_params()).mask_u8
                    if road_mask is not None and getattr(road_mask, "size", 0) != 0:
                        self._lru_put_mask(fr.frame_info.frame_index, road_mask)
                except Exception:
                    road_mask = None

        annotated = draw_annotations(base_bgr, dz_px, fr.persons, obstructing, road_mask)
        self._set_preview_bgr(annotated, target="main")

        show_mask = bool(getattr(self, "show_road_mask_var", tk.BooleanVar(value=False)).get())
        self._set_mask_panel_visible(show_mask)
        if show_mask and road_mask is not None and getattr(road_mask, "size", 0) != 0:
            mask_bgr = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            self._set_preview_bgr(mask_bgr, target="mask")
        elif not show_mask:
            # nothing
            pass
        else:
            # show placeholder if mask couldn't be computed
            self.preview_mask.configure(image=None, text="Маска недоступна")

    def _on_roi_change(self) -> None:
        """Обработчик изменения ROI-слайдеров: clamp, синхронизация UI и обновление превью/процессора."""
        if self._roi_internal_update:
            return
        # ignore manual changes when auto-mode is enabled (sliders are disabled, but keep it safe)
        if bool(getattr(self, "roi_auto_var", tk.BooleanVar(value=False)).get()):
            return
        roi = self._get_roi_pct_clamped()
        # if clamped changed values, sync back to UI
        if (
            abs(roi.x - float(self.roi_x.get())) > 1e-6
            or abs(roi.y - float(self.roi_y.get())) > 1e-6
            or abs(roi.w - float(self.roi_w.get())) > 1e-6
            or abs(roi.h - float(self.roi_h.get())) > 1e-6
        ):
            self._apply_roi_to_vars(roi)

        self._update_roi_value_labels()

        # Apply realtime (processor + preview)
        if self._processor.is_running():
            self._processor.set_roi_pct(roi)
        self._render_preview_from_last()

    def _try_show_first_frame(self, path: str) -> None:
        """Пытается открыть видео и показать первый кадр в превью (чтобы UI был "живым" до старта)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.preview_main.configure(image=None, text="Не удалось открыть видео для предпросмотра")
            self._set_mask_panel_visible(False)
            return
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.preview_main.configure(image=None, text="Не удалось прочитать первый кадр")
                self._set_mask_panel_visible(False)
                return
            # Сделаем preview "живым": сохраним кадр как _last_frame,
            # чтобы работали Road Debug / маска / авто-zone ещё до нажатия "Старт".
            h, w = frame.shape[:2]
            self._last_frame = FrameResult(
                frame_info=FrameInfo(frame_index=0, fps=0.0, width=int(w), height=int(h)),
                frame_bgr=frame,
                persons=[],
                obstructing=False,
            )
            if bool(getattr(self, "roi_auto_var", tk.BooleanVar(value=False)).get()):
                self._auto_roi_recompute(force=True)
            self._render_preview_from_last()
        finally:
            cap.release()

    def _build_ui(self) -> None:
        """Собирает layout главного окна (grid) и подключает UI-компоненты из `sidebar`/`preview_panel`."""
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        build_sidebar(self)
        build_preview_panel(self)

    def _add_roi_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        base_row: int,
    ) -> None:
        """Утилита для sidebar: добавляет слайдер ROI + числовую подпись значения."""
        # header row: name + numeric value
        ctk.CTkLabel(parent, text=f"{label}:", anchor="w").grid(row=base_row, column=0, sticky="w", padx=10, pady=(0, 0))
        val_lbl = ctk.CTkLabel(parent, text="", anchor="e", text_color="#bbb")
        val_lbl.grid(row=base_row, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
        self._roi_value_labels[label] = val_lbl

        # slider row
        s = ctk.CTkSlider(parent, from_=from_, to=to, variable=var, number_of_steps=int(to - from_))
        s.grid(row=base_row + 1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self._roi_sliders.append(s)

    def _on_open_video(self) -> None:
        """Открывает диалог выбора файла видео и инициализирует предпросмотр/состояние UI."""
        path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Видео", "*.mp4 *.mov *.mkv *.avi"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        self._video_path = path
        # new video -> drop preview caches
        self._preview_cache_key = None
        self._road_mask_cache_lru.clear()
        self._auto_dz_cache_lru.clear()
        self.lbl_video.configure(text=f"Видео: {path}")
        self.lbl_status.configure(text="Статус: видео выбрано, можно стартовать")
        self.progress.set(0.0)
        try:
            self._try_show_first_frame(path)
        except Exception as e:
            # Не валим UI из-за предпросмотра — просто показываем причину
            self.preview_main.configure(image=None, text=f"Предпросмотр недоступен: {e}")
            self._set_mask_panel_visible(False)

    def _on_choose_outdir(self) -> None:
        """Открывает диалог выбора папки вывода (для экспорта видео/JSON)."""
        d = filedialog.askdirectory(title="Папка вывода")
        if not d:
            return
        self.export_dir_var.set(d)
        self.lbl_outdir.configure(text=d)

    def _on_start(self) -> None:
        """Запускает обработку видео в `VideoProcessor` с текущими настройками UI."""
        if self._processor.is_running():
            return
        if not self._video_path:
            messagebox.showwarning("Нет видео", "Сначала выберите видео.")
            return
        if not os.path.exists(self._video_path):
            messagebox.showerror("Ошибка", "Файл видео не найден.")
            return

        roi = RoiPct(
            x=float(self.roi_x.get()),
            y=float(self.roi_y.get()),
            w=float(self.roi_w.get()),
            h=float(self.roi_h.get()),
        ).clamp()

        det_cfg = DetectionConfig(
            model_name=self.model_var.get().strip() or "yolo11n.pt",
            device=self.device_var.get().strip() or "auto",
            conf=float(self.conf_var.get()),
            min_area_pct=float(self.min_area_var.get()),
            roi=roi,
            danger_zone_mode=("road" if bool(self.roi_auto_var.get()) else "roi"),
            show_road_mask=bool(getattr(self, "show_road_mask_var", tk.BooleanVar(value=False)).get()),
        )
        exp_cfg = ExportConfig(
            export_video=bool(self.export_video_var.get()),
            export_json=bool(self.export_json_var.get()),
            out_dir=self.export_dir_var.get().strip(),
        )

        try:
            # Если видео запускается повторно, frame_index снова пойдёт с 0.
            # В этом случае старый _auto_roi_last_frame_index (из прошлого прогона) ломает throttle
            # в _auto_roi_recompute(): (0 - last_idx) < every_n => авто-ROI/маска никогда не пересчитаются,
            # а оверлеи остаются статичными. Сбросим кэши заранее.
            if bool(getattr(self, "roi_auto_var", tk.BooleanVar(value=False)).get()):
                self._auto_roi_smoothed = None
                self._auto_roi_last_frame_index = None
                self._auto_road_mask_u8 = None
                self._auto_road_debug = None

            self._processor.start(self._video_path, det_cfg, exp_cfg)
            self._paused = False
            self.btn_pause.configure(text="Пауза")
            self._processor.set_realtime(bool(self.realtime_var.get()))
            # sync runtime params (so processing == preview)
            self._processor.set_road_params(
                color_dist_thresh=float(self._road_color_thresh_var.get()),
                use_perspective=bool(self._use_perspective_var.get()),
                dz_near_bottom_frac=float(self._dz_near_bottom_var.get()) / 100.0,
                dz_edge_quantile=float(self._dz_edge_q_var.get()) / 100.0,
                dz_method=self._get_dz_method(),
            )
            self.lbl_status.configure(text="Статус: воспроизведение…")
        except Exception as e:
            messagebox.showerror("Ошибка запуска", str(e))

    def _on_pause_toggle(self) -> None:
        """Ставит обработку на паузу/продолжает (через `VideoProcessor.pause/resume`)."""
        if not self._processor.is_running():
            return
        self._paused = not self._paused
        if self._paused:
            self._processor.pause()
            self.btn_pause.configure(text="Продолжить")
            self.lbl_status.configure(text="Статус: пауза")
        else:
            self._processor.resume()
            self.btn_pause.configure(text="Пауза")
            self._processor.set_realtime(bool(self.realtime_var.get()))
            self.lbl_status.configure(text="Статус: воспроизведение…")

    def _on_stop(self) -> None:
        """Останавливает обработку (best-effort) и обновляет UI-статус."""
        self._processor.stop()
        self.lbl_status.configure(text="Статус: остановка…")
        self._paused = False
        self.btn_pause.configure(text="Пауза")

    def _tick(self) -> None:
        """
        Таймер UI: регулярно забирает элементы из `VideoProcessor.poll()` и применяет их.

        Почему так: Tkinter не потокобезопасен, поэтому любые обновления UI делаем только в главном потоке.
        """
        # Pull results from worker thread
        for item in self._processor.poll(max_items=10):
            if isinstance(item, Exception):
                messagebox.showerror("Ошибка", str(item))
                self.lbl_status.configure(text=f"Статус: ошибка — {item}")
                continue
            if isinstance(item, ProcessorProgress):
                self._apply_progress(item)
                continue
            if isinstance(item, FrameResult):
                self._apply_frame(item)
                continue

        # schedule next tick
        self.after(30, self._tick)

    def _apply_progress(self, p: ProcessorProgress) -> None:
        """Обновляет прогресс-бар и статус-строку по `ProcessorProgress`."""
        if p.frame_count > 0:
            self.progress.set(min(1.0, max(0.0, p.frame_index / p.frame_count)))
        else:
            self.progress.set(0.0)
        st = "МЕШАЕТ" if p.obstructing else "OK"
        if not self._paused:
            self._processor.set_realtime(bool(self.realtime_var.get()))
        state = "пауза" if self._paused else "воспроизведение"
        self.lbl_status.configure(text=f"Статус: {state} | кадр {p.frame_index} / {p.frame_count} | {st}")

    def _apply_frame(self, fr: FrameResult) -> None:
        """
        Принимает очередной кадр от процессора и обновляет превью.

        Если включён авто-режим danger_zone, то пересчитываем/обновляем его по throttling-правилам.
        """
        self._last_frame = fr
        if bool(self.roi_auto_var.get()):
            # Если воспроизведение стартует заново (frame_index сбросился),
            # сбрасываем throttle/caches, иначе авто-маска и авто danger_zone "залипают"
            # на результатах предыдущего прогона.
            last_idx = self._auto_roi_last_frame_index
            if last_idx is not None and fr.frame_info.frame_index < last_idx:
                self._auto_roi_smoothed = None
                self._auto_roi_last_frame_index = None
                self._auto_road_mask_u8 = None
                self._auto_road_debug = None
                self._auto_roi_recompute(force=True)
            else:
                self._auto_roi_recompute(force=False)
        self._render_preview_from_last()


