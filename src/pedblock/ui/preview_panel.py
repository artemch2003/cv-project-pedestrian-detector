from __future__ import annotations

import customtkinter as ctk


def build_preview_panel(app: object) -> None:
    """
    Собирает правую панель превью (основной кадр + опциональная маска).

    Вынесено из `MainWindow._build_ui()`, чтобы не держать layout всего окна в одном файле.
    Ожидается, что `app` имеет методы/атрибуты:
    - `_schedule_preview_rescale()`
    """
    right = ctk.CTkFrame(app)  # type: ignore[arg-type]
    right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
    right.grid_rowconfigure(0, weight=1)
    right.grid_columnconfigure(0, weight=1)
    right.grid_columnconfigure(1, weight=0, minsize=0)

    app._preview_right = right  # type: ignore[attr-defined]

    app.preview_main = ctk.CTkLabel(  # type: ignore[attr-defined]
        right,
        text="Откройте видео и нажмите Старт",
        anchor="center",
    )
    app.preview_main.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=12, pady=12)  # type: ignore[attr-defined]

    app.preview_mask = ctk.CTkLabel(  # type: ignore[attr-defined]
        right,
        text="Маска (включите «Показывать маску…»)",
        anchor="center",
    )
    app.preview_mask.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)  # type: ignore[attr-defined]
    app.preview_mask.grid_remove()  # type: ignore[attr-defined]
    app._preview_mask_visible = False  # type: ignore[attr-defined]

    # Re-fit previews when user resizes the window.
    app.preview_main.bind("<Configure>", lambda _e: app._schedule_preview_rescale())  # type: ignore[attr-defined]
    app.preview_mask.bind("<Configure>", lambda _e: app._schedule_preview_rescale())  # type: ignore[attr-defined]
    right.bind("<Configure>", lambda _e: app._schedule_preview_rescale())


