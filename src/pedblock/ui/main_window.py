from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from pedblock import __version__ as app_version
from pedblock.core.config import DetectionConfig, ExportConfig, RoiPct
from pedblock.core.roi import roi_pct_to_px
from pedblock.core.processor import FrameResult, ProcessorProgress, VideoProcessor
from pedblock.core.annotate import draw_annotations


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Pedestrian Blocker v{app_version} — YOLO + CustomTkinter")
        self.geometry("1200x780")
        self.minsize(1000, 680)

        self._video_path: str | None = None
        self._processor = VideoProcessor()
        self._preview_imgtk: ImageTk.PhotoImage | None = None
        self._paused = False
        self._roi_internal_update = False
        self._last_frame: FrameResult | None = None
        self._roi_value_labels: dict[str, ctk.CTkLabel] = {}

        self._build_ui()
        self._tick()

    def _set_preview_bgr(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit to preview label size (keep aspect ratio)
        w = max(10, self.preview.winfo_width())
        h = max(10, self.preview.winfo_height())
        img.thumbnail((w, h))

        self._preview_imgtk = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._preview_imgtk, text="")

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
        roi_pct = self._get_roi_pct_clamped()
        w = int(fr.frame_info.width)
        h = int(fr.frame_info.height)
        roi_px = roi_pct_to_px(roi_pct.x, roi_pct.y, roi_pct.w, roi_pct.h, w, h)

        # recompute obstructing for the preview (so pause + ROI move feels instant)
        frame_area = float(w * h) if w > 0 and h > 0 else 1.0
        min_area = (float(self.min_area_var.get()) / 100.0) * frame_area
        obstructing = False
        for b in fr.persons:
            if b.area < min_area:
                continue
            if roi_px.contains_point(b.cx, b.cy):
                obstructing = True
                break

        annotated = draw_annotations(fr.frame_bgr, roi_px, fr.persons, obstructing)
        self._set_preview_bgr(annotated)

    def _on_roi_change(self) -> None:
        if self._roi_internal_update:
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
            self.preview.configure(image=None, text="Не удалось открыть видео для предпросмотра")
            return
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.preview.configure(image=None, text="Не удалось прочитать первый кадр")
                return
            # show ROI overlay even before старт
            h, w = frame.shape[:2]
            roi = self._get_roi_pct_clamped()
            roi_px = roi_pct_to_px(roi.x, roi.y, roi.w, roi.h, w, h)
            tmp = frame.copy()
            cv2.rectangle(tmp, (roi_px.x1, roi_px.y1), (roi_px.x2, roi_px.y2), (255, 200, 0), 2)
            self._set_preview_bgr(tmp)
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

        self.preview = ctk.CTkLabel(right, text="Откройте видео и нажмите Старт", anchor="center")
        self.preview.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # Bind updates for labels
        self.conf_var.trace_add("write", lambda *_: self.conf_lbl.configure(text=f"{self.conf_var.get():.2f}"))
        self.min_area_var.trace_add("write", lambda *_: self.min_area_lbl.configure(text=f"{self.min_area_var.get():.2f}%"))
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
        self.lbl_video.configure(text=f"Видео: {path}")
        self.lbl_status.configure(text="Статус: видео выбрано, можно стартовать")
        self.progress.set(0.0)
        try:
            self._try_show_first_frame(path)
        except Exception as e:
            # Не валим UI из-за предпросмотра — просто показываем причину
            self.preview.configure(image=None, text=f"Предпросмотр недоступен: {e}")

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
        )
        exp_cfg = ExportConfig(
            export_video=bool(self.export_video_var.get()),
            export_json=bool(self.export_json_var.get()),
            out_dir=self.export_dir_var.get().strip(),
        )

        try:
            self._processor.start(self._video_path, det_cfg, exp_cfg)
            self._paused = False
            self.btn_pause.configure(text="Пауза")
            self._processor.set_realtime(bool(self.realtime_var.get()))
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
        self._render_preview_from_last()


