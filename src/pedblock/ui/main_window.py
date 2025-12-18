from __future__ import annotations

from collections import OrderedDict
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from pedblock import __version__ as app_version
from pedblock.core.camera_calib import DEFAULT_CALIB_PATH, CameraCalib, load_camera_calib, order_quad_tl_tr_br_bl, save_camera_calib
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.auto_roi import estimate_danger_zone_pct
from pedblock.core.danger_zone import DangerZonePct, danger_zone_pct_to_px
from pedblock.core.road_area import RoadAreaParams, RoadDebug, danger_zone_pct_from_road_mask, estimate_road_mask, estimate_road_mask_debug
from pedblock.core.processor import FrameResult, ProcessorProgress, VideoProcessor
from pedblock.core.types import FrameInfo
from pedblock.core.annotate import draw_annotations


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Pedestrian Blocker v{app_version} — YOLO + CustomTkinter")
        self.geometry("1200x780")
        self.minsize(1000, 680)

        self._video_path: str | None = None
        self._processor = VideoProcessor()
        self._preview_imgtk_main: ImageTk.PhotoImage | None = None
        self._preview_imgtk_mask: ImageTk.PhotoImage | None = None
        self._paused = False
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

        # Preview cache: allows "повторный просмотр" без пересчёта маски/авто danger_zone.
        # Храним только последние N кадров (LRU), иначе память улетит на длинных видео.
        self._preview_cache_key: tuple[object, ...] | None = None
        self._road_mask_cache_lru: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._auto_dz_cache_lru: "OrderedDict[int, DangerZonePct]" = OrderedDict()
        self._preview_cache_max_items: int = 80

        self._build_ui()
        self._tick()

    def _get_preview_cache_key(self) -> tuple[object, ...] | None:
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
        key = self._get_preview_cache_key()
        if key != self._preview_cache_key:
            self._preview_cache_key = key
            self._road_mask_cache_lru.clear()
            self._auto_dz_cache_lru.clear()

    def _lru_get_mask(self, frame_index: int) -> np.ndarray | None:
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
        try:
            self._road_mask_cache_lru[int(frame_index)] = mask_u8
            self._road_mask_cache_lru.move_to_end(int(frame_index))
            while len(self._road_mask_cache_lru) > int(self._preview_cache_max_items):
                self._road_mask_cache_lru.popitem(last=False)
        except Exception:
            # cache is best-effort; never break preview
            pass

    def _lru_get_auto_dz(self, frame_index: int) -> DangerZonePct | None:
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
        try:
            self._auto_dz_cache_lru[int(frame_index)] = dz
            self._auto_dz_cache_lru.move_to_end(int(frame_index))
            while len(self._auto_dz_cache_lru) > int(self._preview_cache_max_items):
                self._auto_dz_cache_lru.popitem(last=False)
        except Exception:
            pass

    def _set_preview_bgr(self, bgr, *, target: str = "main") -> None:
        lbl = self.preview_main if target == "main" else self.preview_mask
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit to preview label size (keep aspect ratio)
        w = max(10, lbl.winfo_width())
        h = max(10, lbl.winfo_height())
        img.thumbnail((w, h))

        imgtk = ImageTk.PhotoImage(img)
        if target == "main":
            self._preview_imgtk_main = imgtk
        else:
            self._preview_imgtk_mask = imgtk
        lbl.configure(image=imgtk, text="")

    def _set_mask_panel_visible(self, visible: bool) -> None:
        if visible:
            self.preview_mask.grid()
        else:
            self.preview_mask.grid_remove()

    def _get_road_params(self) -> RoadAreaParams:
        thr = float(self._road_color_thresh_var.get())
        return RoadAreaParams(
            color_dist_thresh=max(1.0, min(60.0, thr)),
            use_perspective=bool(self._use_perspective_var.get()),
            calib_path=DEFAULT_CALIB_PATH,
            dz_near_bottom_frac=max(0.05, min(0.95, float(self._dz_near_bottom_var.get()) / 100.0)),
            dz_edge_quantile=max(0.0, min(0.20, float(self._dz_edge_q_var.get()) / 100.0)),
            dz_method=("max_width" if bool(self._dz_max_width_var.get()) else "fit"),
        )

    def _on_perspective_toggle(self) -> None:
        if self._processor.is_running():
            self._processor.set_road_params(use_perspective=bool(self._use_perspective_var.get()))
        # perspective changes can affect both ROI-from-road and the overlay
        if bool(self.roi_auto_var.get()):
            self._auto_roi_recompute(force=True)
        else:
            self._render_preview_from_last()

    def _open_perspective_calib_dialog(self) -> None:
        fr = self._last_frame
        if fr is None or fr.frame_bgr is None or getattr(fr.frame_bgr, "size", 0) == 0:
            messagebox.showinfo("Калибровка", "Сначала откройте видео и получите кадр (Старт или предпросмотр).")
            return

        # Copy frame to avoid accidental mutation
        frame_bgr = fr.frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        top = ctk.CTkToplevel(self)
        top.title("Калибровка перспективы (OpenADAS)")
        top.geometry("980x720")
        top.grab_set()

        info = (
            "Кликните 4 точки на кадре (порядок не важен — я сам нормализую в TL/TR/BR/BL).\n"
            "TL/TR — дальние (выше на кадре), BL/BR — ближние (ниже на кадре).\n"
            "Далее введите измерения (метры): L1/L2 — расстояния до ближней/дальней линии, "
            "W1/W2 — ширина дороги на этих дистанциях."
        )
        ctk.CTkLabel(top, text=info, anchor="w", justify="left", wraplength=940).pack(padx=14, pady=(14, 8), fill="x")

        frm = ctk.CTkFrame(top)
        frm.pack(padx=14, pady=(0, 10), fill="x")
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(2, weight=1)
        frm.grid_columnconfigure(3, weight=1)
        frm.grid_columnconfigure(4, weight=1)

        def _mk_entry(col: int, label: str, default: str) -> tk.Entry:
            ctk.CTkLabel(frm, text=label, anchor="w").grid(row=0, column=col, sticky="ew", padx=8, pady=(10, 0))
            e = tk.Entry(frm)
            e.insert(0, default)
            e.grid(row=1, column=col, sticky="ew", padx=8, pady=(0, 10))
            return e

        existing = load_camera_calib(DEFAULT_CALIB_PATH)
        eL1 = _mk_entry(0, "L1 (м):", f"{existing.L1_m:.2f}" if existing else "5.00")
        eL2 = _mk_entry(1, "L2 (м):", f"{existing.L2_m:.2f}" if existing else "20.00")
        eW1 = _mk_entry(2, "W1 (м):", f"{existing.W1_m:.2f}" if existing else "3.50")
        eW2 = _mk_entry(3, "W2 (м):", f"{existing.W2_m:.2f}" if existing else "3.50")
        ePPM = _mk_entry(4, "px/м:", f"{existing.px_per_meter:.1f}" if existing else "60.0")

        ctk.CTkLabel(top, text=f"Файл: {DEFAULT_CALIB_PATH}", anchor="w", text_color="#bbb").pack(
            padx=14, pady=(0, 8), fill="x"
        )

        # Canvas preview
        canvas = tk.Canvas(top, bg="black", width=940, height=520, highlightthickness=0)
        canvas.pack(padx=14, pady=(0, 10), fill="both", expand=True)

        # Fit frame into canvas
        max_w, max_h = 940, 520
        scale = min(max_w / max(1, w), max_h / max(1, h))
        disp_w = int(round(w * scale))
        disp_h = int(round(h * scale))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_base = Image.fromarray(frame_rgb).resize((disp_w, disp_h))

        points: list[tuple[float, float]] = []
        imgtk_holder: dict[str, ImageTk.PhotoImage] = {}

        def _render() -> None:
            img = pil_base.copy()
            # draw points and poly
            arr = np.array(img)
            # point markers
            labels = ["P1", "P2", "P3", "P4"]
            ordered: list[tuple[float, float]] | None = None
            if len(points) == 4:
                try:
                    ordered = order_quad_tl_tr_br_bl(points)
                    labels = ["TL", "TR", "BR", "BL"]
                except Exception:
                    ordered = None

            draw_pts = ordered if ordered is not None else points
            for i, (px, py) in enumerate(draw_pts):
                dx = int(round(px * scale))
                dy = int(round(py * scale))
                cv2.circle(arr, (dx, dy), 6, (255, 120, 0), -1)
                cv2.putText(arr, labels[i], (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
            if ordered is not None:
                poly = np.array([(int(round(x * scale)), int(round(y * scale))) for (x, y) in ordered], dtype=np.int32)
                cv2.polylines(arr, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
            img2 = Image.fromarray(arr)
            imgtk = ImageTk.PhotoImage(img2)
            imgtk_holder["img"] = imgtk
            canvas.delete("all")
            canvas.create_image(0, 0, image=imgtk, anchor="nw")

        def _on_click(ev) -> None:
            nonlocal points
            if len(points) >= 4:
                return
            x_disp = float(ev.x)
            y_disp = float(ev.y)
            if x_disp < 0 or y_disp < 0 or x_disp >= disp_w or y_disp >= disp_h:
                return
            x_orig = x_disp / scale
            y_orig = y_disp / scale
            points.append((float(x_orig), float(y_orig)))
            _render()

        canvas.bind("<Button-1>", _on_click)

        # preload points from existing calib (if any)
        if existing is not None:
            try:
                points = [(float(x), float(y)) for (x, y) in existing.image_points_px]
            except Exception:
                points = []
        _render()

        frm_btn = ctk.CTkFrame(top)
        frm_btn.pack(padx=14, pady=(0, 14), fill="x")
        frm_btn.grid_columnconfigure(0, weight=1)
        frm_btn.grid_columnconfigure(1, weight=1)
        frm_btn.grid_columnconfigure(2, weight=1)

        def _reset() -> None:
            nonlocal points
            points = []
            _render()

        def _save() -> None:
            try:
                if len(points) != 4:
                    raise ValueError("Нужно выбрать 4 точки")
                pts = order_quad_tl_tr_br_bl(points)
                calib = CameraCalib(
                    image_points_px=[(float(x), float(y)) for (x, y) in pts],
                    L1_m=float(eL1.get()),
                    L2_m=float(eL2.get()),
                    W1_m=float(eW1.get()),
                    W2_m=float(eW2.get()),
                    px_per_meter=float(ePPM.get()),
                )
                save_camera_calib(calib, DEFAULT_CALIB_PATH)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить калибровку: {e}")
                return

            # Enable perspective mode immediately
            self._use_perspective_var.set(True)
            self._on_perspective_toggle()
            messagebox.showinfo("Готово", "Калибровка сохранена. Перспективный режим включён.")
            top.destroy()

        ctk.CTkButton(frm_btn, text="Сбросить точки", command=_reset).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=10)
        ctk.CTkButton(frm_btn, text="Сохранить", command=_save).grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=10)
        ctk.CTkButton(frm_btn, text="Закрыть", command=top.destroy).grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=10)

    def _get_roi_pct_clamped(self) -> RoiPct:
        roi = RoiPct(
            x=float(self.roi_x.get()),
            y=float(self.roi_y.get()),
            w=float(self.roi_w.get()),
            h=float(self.roi_h.get()),
        ).clamp()
        return roi

    def _apply_roi_to_vars(self, roi: RoiPct) -> None:
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
        state = "normal" if enabled else "disabled"
        for s in self._roi_sliders:
            try:
                s.configure(state=state)
            except Exception:
                # ignore if a widget doesn't support state (older CTk versions)
                pass

    def _apply_roi_pct(self, roi: RoiPct) -> None:
        # manual ROI sliders define a rectangle; keep them, but convert later for processing/drawing
        roi = roi.clamp()
        self._apply_roi_to_vars(roi)
        self._update_roi_value_labels()
        if self._processor.is_running():
            self._processor.set_roi_pct(roi)
        self._render_preview_from_last()

    def _auto_roi_recompute(self, *, force: bool = False) -> None:
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
            every_n = max(1, min(60, every_n))
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

        est = danger_zone_pct_from_road_mask(self._auto_road_mask_u8, self._get_road_params())
        # 2) Fallback: old line-based trapezoid (still helpful if mask failed)
        if est is None:
            est = estimate_danger_zone_pct(fr.frame_bgr)

        # Smooth to avoid flicker on noisy lines
        alpha = 0.30
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
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left panel: controls (scrollable)
        left_outer = ctk.CTkFrame(self)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left_outer.grid_columnconfigure(0, weight=1)
        left_outer.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(left_outer, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        hdr.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(hdr, text="Настройки", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, sticky="w")

        ver = ctk.CTkLabel(hdr, text=f"v{app_version}", text_color="#888")
        ver.grid(row=0, column=1, sticky="e")

        left = ctk.CTkScrollableFrame(left_outer, width=380)
        left.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        left.grid_columnconfigure(0, weight=1)

        self.btn_open = ctk.CTkButton(left, text="Открыть видео…", command=self._on_open_video)
        self.btn_open.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.lbl_video = ctk.CTkLabel(left, text="Видео: (не выбрано)", anchor="w", justify="left", wraplength=340)
        self.lbl_video.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        # Model / device
        frm_md = ctk.CTkFrame(left)
        frm_md.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        frm_md.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frm_md, text="Модель YOLO (Ultralytics):", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=10, pady=(10, 4)
        )
        self.model_var = tk.StringVar(value="yolo11n.pt")
        self.model_entry = ctk.CTkEntry(frm_md, textvariable=self.model_var)
        self.model_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        ctk.CTkLabel(frm_md, text="Напр.: yolo11n.pt / yolov8n.pt / путь к .pt", anchor="w", text_color="#888").grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 10)
        )

        ctk.CTkLabel(frm_md, text="Устройство:", anchor="w").grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
        self.device_var = tk.StringVar(value="auto")
        self.device_menu = ctk.CTkOptionMenu(frm_md, variable=self.device_var, values=["auto", "cpu", "mps", "cuda"])
        self.device_menu.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Thresholds
        frm_thr = ctk.CTkFrame(left)
        frm_thr.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
        frm_thr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frm_thr, text="Порог confidence:", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.conf_var = tk.DoubleVar(value=0.30)
        self.conf_slider = ctk.CTkSlider(frm_thr, from_=0.05, to=0.95, variable=self.conf_var, number_of_steps=90)
        self.conf_slider.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.conf_lbl = ctk.CTkLabel(frm_thr, text="0.30", anchor="w")
        self.conf_lbl.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

        ctk.CTkLabel(frm_thr, text="Минимальная площадь bbox (% кадра):", anchor="w").grid(
            row=3, column=0, sticky="ew", padx=10, pady=(0, 0)
        )
        self.min_area_var = tk.DoubleVar(value=0.20)
        self.min_area_slider = ctk.CTkSlider(frm_thr, from_=0.01, to=5.0, variable=self.min_area_var, number_of_steps=250)
        self.min_area_slider.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.min_area_lbl = ctk.CTkLabel(frm_thr, text="0.20%", anchor="w")
        self.min_area_lbl.grid(row=5, column=0, sticky="w", padx=10, pady=(0, 10))

        # ROI
        frm_roi = ctk.CTkFrame(left)
        frm_roi.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))
        frm_roi.grid_columnconfigure(0, weight=1)
        frm_roi.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(frm_roi, text="ROI (проценты кадра):", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        self.roi_x = tk.DoubleVar(value=35.0)
        self.roi_y = tk.DoubleVar(value=55.0)
        self.roi_w = tk.DoubleVar(value=30.0)
        self.roi_h = tk.DoubleVar(value=40.0)

        self._add_roi_slider(frm_roi, "X", self.roi_x, 0, 100, base_row=1)
        self._add_roi_slider(frm_roi, "Y", self.roi_y, 0, 100, base_row=3)
        self._add_roi_slider(frm_roi, "W", self.roi_w, 1, 100, base_row=5)
        self._add_roi_slider(frm_roi, "H", self.roi_h, 1, 100, base_row=7)

        self.roi_auto_var = tk.BooleanVar(value=False)
        self.chk_auto_roi = ctk.CTkCheckBox(
            frm_roi,
            text="Авто danger_zone (по проезжей части)",
            variable=self.roi_auto_var,
            command=self._on_auto_roi_toggle,
        )
        self.chk_auto_roi.grid(row=9, column=0, sticky="w", padx=10, pady=(0, 6))

        self.btn_roi_recalc = ctk.CTkButton(frm_roi, text="Пересчитать по кадру", command=lambda: self._auto_roi_recompute(force=True))
        self.btn_roi_recalc.grid(row=9, column=1, sticky="e", padx=(0, 10), pady=(0, 6))

        # Show road mask overlay
        self.show_road_mask_var = tk.BooleanVar(value=False)
        self.chk_show_road = ctk.CTkCheckBox(frm_roi, text="Показывать маску проезжей части", variable=self.show_road_mask_var)
        self.chk_show_road.grid(row=12, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self.show_road_mask_var.trace_add(
            "write",
            lambda *_: (
                self._processor.set_show_road_mask(bool(self.show_road_mask_var.get())) if self._processor.is_running() else None,
                self._render_preview_from_last(),
            ),
        )

        # Road segmentation strictness (color threshold)
        ctk.CTkLabel(frm_roi, text="Строгость маски (цвет, меньше = строже):", anchor="w", text_color="#bbb").grid(
            row=13, column=0, sticky="w", padx=10, pady=(0, 0)
        )
        self._road_thresh_lbl = ctk.CTkLabel(frm_roi, text="7.5", anchor="e", text_color="#bbb")
        self._road_thresh_lbl.grid(row=13, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
        self._road_thresh_slider = ctk.CTkSlider(
            frm_roi,
            from_=2.0,
            to=25.0,
            variable=self._road_color_thresh_var,
            number_of_steps=230,
        )
        self._road_thresh_slider.grid(row=14, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self._road_color_thresh_var.trace_add(
            "write",
            lambda *_: (
                self._road_thresh_lbl.configure(text=f"{self._road_color_thresh_var.get():.1f}"),
                self._processor.set_road_params(color_dist_thresh=float(self._road_color_thresh_var.get())) if self._processor.is_running() else None,
                self._auto_roi_recompute(force=True) if bool(self.roi_auto_var.get()) else self._render_preview_from_last(),
            ),
        )

        # Danger-zone tuning from road mask
        ctk.CTkLabel(frm_roi, text="Высота danger_zone от низа (%):", anchor="w", text_color="#bbb").grid(
            row=19, column=0, sticky="w", padx=10, pady=(0, 0)
        )
        self._dz_near_bottom_lbl = ctk.CTkLabel(frm_roi, text="35", anchor="e", text_color="#bbb")
        self._dz_near_bottom_lbl.grid(row=19, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
        self._dz_near_bottom_slider = ctk.CTkSlider(frm_roi, from_=15.0, to=60.0, variable=self._dz_near_bottom_var, number_of_steps=45)
        self._dz_near_bottom_slider.grid(row=20, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self._dz_near_bottom_var.trace_add(
            "write",
            lambda *_: (
                self._dz_near_bottom_lbl.configure(text=f"{self._dz_near_bottom_var.get():.0f}"),
                self._processor.set_road_params(dz_near_bottom_frac=float(self._dz_near_bottom_var.get()) / 100.0)
                if self._processor.is_running()
                else None,
                self._auto_roi_recompute(force=True) if bool(self.roi_auto_var.get()) else self._render_preview_from_last(),
            ),
        )

        ctk.CTkLabel(frm_roi, text="Отсечь края маски (%):", anchor="w", text_color="#bbb").grid(
            row=21, column=0, sticky="w", padx=10, pady=(0, 0)
        )
        self._dz_edge_q_lbl = ctk.CTkLabel(frm_roi, text="6", anchor="e", text_color="#bbb")
        self._dz_edge_q_lbl.grid(row=21, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
        self._dz_edge_q_slider = ctk.CTkSlider(frm_roi, from_=0.0, to=15.0, variable=self._dz_edge_q_var, number_of_steps=150)
        self._dz_edge_q_slider.grid(row=22, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self._dz_edge_q_var.trace_add(
            "write",
            lambda *_: (
                self._dz_edge_q_lbl.configure(text=f"{self._dz_edge_q_var.get():.0f}"),
                self._processor.set_road_params(dz_edge_quantile=float(self._dz_edge_q_var.get()) / 100.0)
                if self._processor.is_running()
                else None,
                self._auto_roi_recompute(force=True) if bool(self.roi_auto_var.get()) else self._render_preview_from_last(),
            ),
        )

        # Danger-zone method from road mask
        self._dz_max_width_chk = ctk.CTkCheckBox(
            frm_roi,
            text="Трапеция по макс. ширине маски дороги",
            variable=self._dz_max_width_var,
        )
        self._dz_max_width_chk.grid(row=23, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self._dz_max_width_var.trace_add(
            "write",
            lambda *_: (
                self._processor.set_road_params(dz_method=("max_width" if bool(self._dz_max_width_var.get()) else "fit"))
                if self._processor.is_running()
                else None,
                self._auto_roi_recompute(force=True) if bool(self.roi_auto_var.get()) else self._render_preview_from_last(),
            ),
        )

        # Debug view selector
        ctk.CTkLabel(frm_roi, text="Road Debug (превью):", anchor="w", text_color="#bbb").grid(
            row=15, column=0, sticky="w", padx=10, pady=(0, 0)
        )
        self._road_debug_menu = ctk.CTkOptionMenu(
            frm_roi,
            variable=self._road_debug_mode_var,
            values=["Обычный", "Canny", "Коридор", "Seed", "Dist", "Candidate", "Final"],
            command=lambda *_: self._render_preview_from_last(),
        )
        self._road_debug_menu.grid(row=16, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # OpenADAS-style perspective mode (bird-view calibration)
        self.chk_perspective = ctk.CTkCheckBox(
            frm_roi,
            text="Перспектива (OpenADAS): bird-view калибровка",
            variable=self._use_perspective_var,
            command=self._on_perspective_toggle,
        )
        self.chk_perspective.grid(row=17, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        self.btn_calib = ctk.CTkButton(frm_roi, text="Калибровка перспективы…", command=self._open_perspective_calib_dialog)
        self.btn_calib.grid(row=18, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # Auto-ROI frequency (frames)
        ctk.CTkLabel(frm_roi, text="Авто-обновление (кадров):", anchor="w", text_color="#bbb").grid(
            row=10, column=0, sticky="w", padx=10, pady=(0, 0)
        )
        self._auto_roi_every_n_lbl = ctk.CTkLabel(frm_roi, text="5", anchor="e", text_color="#bbb")
        self._auto_roi_every_n_lbl.grid(row=10, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
        self._auto_roi_every_n_slider = ctk.CTkSlider(
            frm_roi,
            from_=1,
            to=30,
            variable=self._auto_roi_every_n_frames_var,
            number_of_steps=29,
        )
        self._auto_roi_every_n_slider.grid(row=11, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # Export
        frm_exp = ctk.CTkFrame(left)
        frm_exp.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 12))
        frm_exp.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frm_exp, text="Экспорт:", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        self.export_video_var = tk.BooleanVar(value=False)
        self.export_json_var = tk.BooleanVar(value=True)
        self.export_dir_var = tk.StringVar(value="")
        self.chk_video = ctk.CTkCheckBox(frm_exp, text="Аннотированное видео (.mp4)", variable=self.export_video_var)
        self.chk_json = ctk.CTkCheckBox(frm_exp, text="JSON события (интервалы 'мешает')", variable=self.export_json_var)
        self.chk_video.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        self.chk_json.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

        self.btn_outdir = ctk.CTkButton(frm_exp, text="Папка вывода…", command=self._on_choose_outdir)
        self.btn_outdir.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.lbl_outdir = ctk.CTkLabel(frm_exp, text="(рядом с видео / pedblock_out)", anchor="w", wraplength=340)
        self.lbl_outdir.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Run controls
        frm_play = ctk.CTkFrame(left)
        frm_play.grid(row=7, column=0, sticky="ew", padx=12, pady=(0, 8))
        frm_play.grid_columnconfigure(0, weight=1)
        frm_play.grid_columnconfigure(1, weight=1)

        self.btn_start = ctk.CTkButton(frm_play, text="Старт", command=self._on_start)
        self.btn_pause = ctk.CTkButton(frm_play, text="Пауза", command=self._on_pause_toggle)
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=10)
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=10)

        self.btn_stop = ctk.CTkButton(left, text="Стоп", command=self._on_stop, fg_color="#7a2222", hover_color="#8f2a2a")
        self.btn_stop.grid(row=8, column=0, sticky="ew", padx=12, pady=(0, 12))

        self.realtime_var = tk.BooleanVar(value=True)
        self.chk_realtime = ctk.CTkCheckBox(left, text="Воспроизведение по FPS (реальное время)", variable=self.realtime_var)
        self.chk_realtime.grid(row=9, column=0, sticky="w", padx=12, pady=(0, 10))

        self.progress = ctk.CTkProgressBar(left)
        self.progress.grid(row=10, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.progress.set(0.0)
        self.lbl_status = ctk.CTkLabel(left, text="Статус: ожидание", anchor="w", justify="left", wraplength=360)
        self.lbl_status.grid(row=11, column=0, sticky="ew", padx=12, pady=(0, 12))

        # Right panel: preview
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        self.preview_main = ctk.CTkLabel(right, text="Откройте видео и нажмите Старт", anchor="center")
        self.preview_main.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        self.preview_mask = ctk.CTkLabel(right, text="Маска (включите «Показывать маску…»)", anchor="center")
        self.preview_mask.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.preview_mask.grid_remove()

        # Bind updates for labels
        self.conf_var.trace_add("write", lambda *_: self.conf_lbl.configure(text=f"{self.conf_var.get():.2f}"))
        self.min_area_var.trace_add("write", lambda *_: self.min_area_lbl.configure(text=f"{self.min_area_var.get():.2f}%"))
        self._auto_roi_every_n_frames_var.trace_add("write", lambda *_: self._auto_roi_every_n_lbl.configure(text=str(int(self._auto_roi_every_n_frames_var.get() or 1))))
        # ROI realtime (values + apply)
        self.roi_x.trace_add("write", lambda *_: self._on_roi_change())
        self.roi_y.trace_add("write", lambda *_: self._on_roi_change())
        self.roi_w.trace_add("write", lambda *_: self._on_roi_change())
        self.roi_h.trace_add("write", lambda *_: self._on_roi_change())
        self._update_roi_value_labels()

    def _add_roi_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        base_row: int,
    ) -> None:
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
        d = filedialog.askdirectory(title="Папка вывода")
        if not d:
            return
        self.export_dir_var.set(d)
        self.lbl_outdir.configure(text=d)

    def _on_start(self) -> None:
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
                dz_method=("max_width" if bool(self._dz_max_width_var.get()) else "fit"),
            )
            self.lbl_status.configure(text="Статус: воспроизведение…")
        except Exception as e:
            messagebox.showerror("Ошибка запуска", str(e))

    def _on_pause_toggle(self) -> None:
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
        self._processor.stop()
        self.lbl_status.configure(text="Статус: остановка…")
        self._paused = False
        self.btn_pause.configure(text="Пауза")

    def _tick(self) -> None:
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


