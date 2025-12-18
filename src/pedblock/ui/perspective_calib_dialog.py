from __future__ import annotations

from typing import Callable
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from pedblock.core.camera_calib import DEFAULT_CALIB_PATH, CameraCalib, load_camera_calib, order_quad_tl_tr_br_bl, save_camera_calib


def open_perspective_calib_dialog(
    parent: ctk.CTk,
    *,
    frame_bgr: np.ndarray,
    use_perspective_var: tk.BooleanVar,
    on_perspective_toggle: Callable[[], None],
    calib_path: str = DEFAULT_CALIB_PATH,
) -> None:
    """
    OpenADAS-style диалог калибровки bird-view (4 точки + L1/L2/W1/W2/px_per_meter).

    Вынесено из main_window.py, чтобы держать MainWindow компактнее.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        messagebox.showinfo("Калибровка", "Нет кадра для калибровки.")
        return

    # Copy frame to avoid accidental mutation
    frame_bgr = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    top = ctk.CTkToplevel(parent)
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
    for col in range(5):
        frm.grid_columnconfigure(col, weight=1)

    def _mk_entry(col: int, label: str, default: str) -> tk.Entry:
        ctk.CTkLabel(frm, text=label, anchor="w").grid(row=0, column=col, sticky="ew", padx=8, pady=(10, 0))
        e = tk.Entry(frm)
        e.insert(0, default)
        e.grid(row=1, column=col, sticky="ew", padx=8, pady=(0, 10))
        return e

    existing = load_camera_calib(calib_path)
    eL1 = _mk_entry(0, "L1 (м):", f"{existing.L1_m:.2f}" if existing else "5.00")
    eL2 = _mk_entry(1, "L2 (м):", f"{existing.L2_m:.2f}" if existing else "20.00")
    eW1 = _mk_entry(2, "W1 (м):", f"{existing.W1_m:.2f}" if existing else "3.50")
    eW2 = _mk_entry(3, "W2 (м):", f"{existing.W2_m:.2f}" if existing else "3.50")
    ePPM = _mk_entry(4, "px/м:", f"{existing.px_per_meter:.1f}" if existing else "60.0")

    ctk.CTkLabel(top, text=f"Файл: {calib_path}", anchor="w", text_color="#bbb").pack(padx=14, pady=(0, 8), fill="x")

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
            save_camera_calib(calib, calib_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить калибровку: {e}")
            return

        # Enable perspective mode immediately
        use_perspective_var.set(True)
        on_perspective_toggle()
        messagebox.showinfo("Готово", "Калибровка сохранена. Перспективный режим включён.")
        top.destroy()

    ctk.CTkButton(frm_btn, text="Сбросить точки", command=_reset).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=10)
    ctk.CTkButton(frm_btn, text="Сохранить", command=_save).grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=10)
    ctk.CTkButton(frm_btn, text="Закрыть", command=top.destroy).grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=10)


