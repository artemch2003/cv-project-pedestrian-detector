from __future__ import annotations

"""
`sidebar.py` — левая панель управления (настройки + кнопки).

Цели модуля:
- держать код сборки UI компактным и изолированным от `main_window.py`
- создавать виджеты и привязывать их к переменным/колбэкам окна

Почему `app: object`:
- чтобы компонент можно было переиспользовать и не тянуть циклические импорты
- фактически ожидается `MainWindow`, поэтому доступ к атрибутам помечен `type: ignore`
"""

import tkinter as tk

import customtkinter as ctk

from pedblock import __version__ as app_version


def build_sidebar(app: object) -> None:
    """
    Собирает левую панель управления (sidebar) для MainWindow.

    Зачем: `main_window.py` быстро разрастается, а sidebar по сути является отдельным
    компонентом: создание виджетов + бинды. Здесь мы оставляем существующее поведение,
    просто выносим код в отдельный файл.

    Ожидается, что `app` (обычно MainWindow) имеет методы/атрибуты:
    - callbacks: _on_open_video, _on_choose_outdir, _on_start, _on_pause_toggle, _on_stop
    - ROI helpers: _add_roi_slider, _on_auto_roi_toggle, _auto_roi_recompute, _on_show_road_mask_changed,
      _on_road_color_thresh_changed, _on_dz_near_bottom_changed, _on_dz_edge_q_changed, _on_dz_method_changed,
      _on_perspective_toggle, _open_perspective_calib_dialog, _render_preview_from_last
    - tk variables already on app: _road_color_thresh_var, _road_debug_mode_var, _use_perspective_var,
      _dz_near_bottom_var, _dz_edge_q_var, _dz_max_width_var, _dz_hough_sides_var, _auto_roi_every_n_frames_var
    """

    # Left panel: controls (scrollable)
    left_outer = ctk.CTkFrame(app)  # type: ignore[arg-type]
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

    # ---- Video ----
    # Кнопка выбора видео + label с путём к файлу.
    app.btn_open = ctk.CTkButton(left, text="Открыть видео…", command=app._on_open_video)  # type: ignore[attr-defined]
    app.btn_open.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    app.lbl_video = ctk.CTkLabel(left, text="Видео: (не выбрано)", anchor="w", justify="left", wraplength=340)  # type: ignore[attr-defined]
    app.lbl_video.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    # ---- Model / device ----
    # Настройки инференса: путь/имя модели и устройство (auto/cpu/mps/cuda).
    frm_md = ctk.CTkFrame(left)
    frm_md.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
    frm_md.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(frm_md, text="Модель YOLO (Ultralytics):", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
    app.model_var = tk.StringVar(value="yolo11n.pt")  # type: ignore[attr-defined]
    app.model_entry = ctk.CTkEntry(frm_md, textvariable=app.model_var)  # type: ignore[attr-defined]
    app.model_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
    ctk.CTkLabel(frm_md, text="Напр.: yolo11n.pt / yolov8n.pt / путь к .pt", anchor="w", text_color="#888").grid(
        row=2, column=0, sticky="ew", padx=10, pady=(0, 10)
    )

    ctk.CTkLabel(frm_md, text="Устройство:", anchor="w").grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
    app.device_var = tk.StringVar(value="auto")  # type: ignore[attr-defined]
    app.device_menu = ctk.CTkOptionMenu(frm_md, variable=app.device_var, values=["auto", "cpu", "mps", "cuda"])  # type: ignore[attr-defined]
    app.device_menu.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))

    # ---- Thresholds ----
    # Порог confidence и фильтрация «мелких» bbox по площади в % кадра.
    frm_thr = ctk.CTkFrame(left)
    frm_thr.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
    frm_thr.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(frm_thr, text="Порог confidence:", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
    app.conf_var = tk.DoubleVar(value=0.30)  # type: ignore[attr-defined]
    app.conf_slider = ctk.CTkSlider(frm_thr, from_=0.05, to=0.95, variable=app.conf_var, number_of_steps=90)  # type: ignore[attr-defined]
    app.conf_slider.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
    app.conf_lbl = ctk.CTkLabel(frm_thr, text="0.30", anchor="w")  # type: ignore[attr-defined]
    app.conf_lbl.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

    ctk.CTkLabel(frm_thr, text="Минимальная площадь bbox (% кадра):", anchor="w").grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 0))
    app.min_area_var = tk.DoubleVar(value=0.20)  # type: ignore[attr-defined]
    app.min_area_slider = ctk.CTkSlider(frm_thr, from_=0.01, to=5.0, variable=app.min_area_var, number_of_steps=250)  # type: ignore[attr-defined]
    app.min_area_slider.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
    app.min_area_lbl = ctk.CTkLabel(frm_thr, text="0.20%", anchor="w")  # type: ignore[attr-defined]
    app.min_area_lbl.grid(row=5, column=0, sticky="w", padx=10, pady=(0, 10))

    # ---- ROI ----
    # ROI задаёт область интереса / «опасную зону» (danger_zone) в процентах кадра.
    frm_roi = ctk.CTkFrame(left)
    frm_roi.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))
    frm_roi.grid_columnconfigure(0, weight=1)
    frm_roi.grid_columnconfigure(1, weight=0)

    ctk.CTkLabel(frm_roi, text="ROI (проценты кадра):", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
    app.roi_x = tk.DoubleVar(value=35.0)  # type: ignore[attr-defined]
    app.roi_y = tk.DoubleVar(value=55.0)  # type: ignore[attr-defined]
    app.roi_w = tk.DoubleVar(value=30.0)  # type: ignore[attr-defined]
    app.roi_h = tk.DoubleVar(value=40.0)  # type: ignore[attr-defined]

    app._add_roi_slider(frm_roi, "X", app.roi_x, 0, 100, base_row=1)  # type: ignore[attr-defined]
    app._add_roi_slider(frm_roi, "Y", app.roi_y, 0, 100, base_row=3)  # type: ignore[attr-defined]
    app._add_roi_slider(frm_roi, "W", app.roi_w, 1, 100, base_row=5)  # type: ignore[attr-defined]
    app._add_roi_slider(frm_roi, "H", app.roi_h, 1, 100, base_row=7)  # type: ignore[attr-defined]

    app.roi_auto_var = tk.BooleanVar(value=False)  # type: ignore[attr-defined]
    app.chk_auto_roi = ctk.CTkCheckBox(  # type: ignore[attr-defined]
        frm_roi,
        text="Авто danger_zone (по проезжей части)",
        variable=app.roi_auto_var,
        command=app._on_auto_roi_toggle,
    )
    app.chk_auto_roi.grid(row=9, column=0, sticky="w", padx=10, pady=(0, 6))

    app.btn_roi_recalc = ctk.CTkButton(frm_roi, text="Пересчитать по кадру", command=lambda: app._auto_roi_recompute(force=True))  # type: ignore[attr-defined]
    app.btn_roi_recalc.grid(row=9, column=1, sticky="e", padx=(0, 10), pady=(0, 6))

    # Show road mask overlay
    # Переключатель отображения маски дороги на превью (оверлей + отдельная панель).
    app.show_road_mask_var = tk.BooleanVar(value=False)  # type: ignore[attr-defined]
    app.chk_show_road = ctk.CTkCheckBox(frm_roi, text="Показывать маску проезжей части", variable=app.show_road_mask_var)  # type: ignore[attr-defined]
    app.chk_show_road.grid(row=12, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    app.show_road_mask_var.trace_add("write", lambda *_: app._on_show_road_mask_changed())  # type: ignore[attr-defined]

    # Road segmentation strictness (color threshold)
    # Основной параметр сегментации дороги: допустимая «дистанция по цвету».
    # Меньше => строже => меньше ложноположительных «дорог».
    ctk.CTkLabel(frm_roi, text="Строгость маски (цвет, меньше = строже):", anchor="w", text_color="#bbb").grid(
        row=13, column=0, sticky="w", padx=10, pady=(0, 0)
    )
    app._road_thresh_lbl = ctk.CTkLabel(frm_roi, text="7.5", anchor="e", text_color="#bbb")  # type: ignore[attr-defined]
    app._road_thresh_lbl.grid(row=13, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
    app._road_thresh_slider = ctk.CTkSlider(  # type: ignore[attr-defined]
        frm_roi,
        from_=2.0,
        to=25.0,
        variable=app._road_color_thresh_var,  # type: ignore[attr-defined]
        number_of_steps=230,
    )
    app._road_thresh_slider.grid(row=14, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
    app._road_color_thresh_var.trace_add("write", lambda *_: app._on_road_color_thresh_changed())  # type: ignore[attr-defined]

    # Debug view selector
    ctk.CTkLabel(frm_roi, text="Road Debug (превью):", anchor="w", text_color="#bbb").grid(row=15, column=0, sticky="w", padx=10, pady=(0, 0))
    app._road_debug_menu = ctk.CTkOptionMenu(  # type: ignore[attr-defined]
        frm_roi,
        variable=app._road_debug_mode_var,  # type: ignore[attr-defined]
        values=["Обычный", "Canny", "Коридор", "Seed", "Dist", "Candidate", "Final"],
        command=lambda *_: app._render_preview_from_last(),  # type: ignore[attr-defined]
    )
    app._road_debug_menu.grid(row=16, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

    # OpenADAS-style perspective mode (bird-view calibration)
    # Перспективное преобразование: из 4 точек на изображении строится bird-view,
    # что может стабилизировать сегментацию дороги (особенно на наклонных камерах).
    app.chk_perspective = ctk.CTkCheckBox(  # type: ignore[attr-defined]
        frm_roi,
        text="Перспектива (OpenADAS): bird-view калибровка",
        variable=app._use_perspective_var,  # type: ignore[attr-defined]
        command=app._on_perspective_toggle,  # type: ignore[attr-defined]
    )
    app.chk_perspective.grid(row=17, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))
    app.btn_calib = ctk.CTkButton(frm_roi, text="Калибровка перспективы…", command=app._open_perspective_calib_dialog)  # type: ignore[attr-defined]
    app.btn_calib.grid(row=18, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

    # Danger-zone tuning from road mask
    # Тюнинг вычисления трапеции danger_zone по маске дороги.
    ctk.CTkLabel(frm_roi, text="Высота danger_zone от низа (%):", anchor="w", text_color="#bbb").grid(row=19, column=0, sticky="w", padx=10, pady=(0, 0))
    app._dz_near_bottom_lbl = ctk.CTkLabel(frm_roi, text="35", anchor="e", text_color="#bbb")  # type: ignore[attr-defined]
    app._dz_near_bottom_lbl.grid(row=19, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
    app._dz_near_bottom_slider = ctk.CTkSlider(frm_roi, from_=15.0, to=60.0, variable=app._dz_near_bottom_var, number_of_steps=45)  # type: ignore[attr-defined]
    app._dz_near_bottom_slider.grid(row=20, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
    app._dz_near_bottom_var.trace_add("write", lambda *_: app._on_dz_near_bottom_changed())  # type: ignore[attr-defined]

    ctk.CTkLabel(frm_roi, text="Отсечь края маски (%):", anchor="w", text_color="#bbb").grid(row=21, column=0, sticky="w", padx=10, pady=(0, 0))
    app._dz_edge_q_lbl = ctk.CTkLabel(frm_roi, text="6", anchor="e", text_color="#bbb")  # type: ignore[attr-defined]
    app._dz_edge_q_lbl.grid(row=21, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
    app._dz_edge_q_slider = ctk.CTkSlider(frm_roi, from_=0.0, to=15.0, variable=app._dz_edge_q_var, number_of_steps=150)  # type: ignore[attr-defined]
    app._dz_edge_q_slider.grid(row=22, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
    app._dz_edge_q_var.trace_add("write", lambda *_: app._on_dz_edge_q_changed())  # type: ignore[attr-defined]

    app._dz_max_width_chk = ctk.CTkCheckBox(  # type: ignore[attr-defined]
        frm_roi,
        text="Трапеция по макс. ширине маски дороги",
        variable=app._dz_max_width_var,  # type: ignore[attr-defined]
    )
    app._dz_max_width_chk.grid(row=23, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    app._dz_max_width_var.trace_add("write", lambda *_: app._on_dz_method_changed())  # type: ignore[attr-defined]

    app._dz_hough_chk = ctk.CTkCheckBox(  # type: ignore[attr-defined]
        frm_roi,
        text="Боковые линии по границам маски (Hough)",
        variable=app._dz_hough_sides_var,  # type: ignore[attr-defined]
    )
    app._dz_hough_chk.grid(row=24, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
    app._dz_hough_sides_var.trace_add("write", lambda *_: app._on_dz_method_changed())  # type: ignore[attr-defined]

    # Auto-ROI frequency (frames)
    # Частота пересчёта авто ROI: влияет на стабильность/нагрузку.
    ctk.CTkLabel(frm_roi, text="Авто-обновление (кадров):", anchor="w", text_color="#bbb").grid(row=10, column=0, sticky="w", padx=10, pady=(0, 0))
    app._auto_roi_every_n_lbl = ctk.CTkLabel(frm_roi, text="5", anchor="e", text_color="#bbb")  # type: ignore[attr-defined]
    app._auto_roi_every_n_lbl.grid(row=10, column=1, sticky="e", padx=(0, 10), pady=(0, 0))
    app._auto_roi_every_n_slider = ctk.CTkSlider(  # type: ignore[attr-defined]
        frm_roi,
        from_=1,
        to=30,
        variable=app._auto_roi_every_n_frames_var,  # type: ignore[attr-defined]
        number_of_steps=29,
    )
    app._auto_roi_every_n_slider.grid(row=11, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

    # ---- Export ----
    frm_exp = ctk.CTkFrame(left)
    frm_exp.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 12))
    frm_exp.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(frm_exp, text="Экспорт:", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
    app.export_video_var = tk.BooleanVar(value=False)  # type: ignore[attr-defined]
    app.export_json_var = tk.BooleanVar(value=True)  # type: ignore[attr-defined]
    app.export_dir_var = tk.StringVar(value="")  # type: ignore[attr-defined]
    app.chk_video = ctk.CTkCheckBox(frm_exp, text="Аннотированное видео (.mp4)", variable=app.export_video_var)  # type: ignore[attr-defined]
    app.chk_json = ctk.CTkCheckBox(frm_exp, text="JSON события (интервалы 'мешает')", variable=app.export_json_var)  # type: ignore[attr-defined]
    app.chk_video.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
    app.chk_json.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

    app.btn_outdir = ctk.CTkButton(frm_exp, text="Папка вывода…", command=app._on_choose_outdir)  # type: ignore[attr-defined]
    app.btn_outdir.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
    app.lbl_outdir = ctk.CTkLabel(frm_exp, text="(рядом с видео / pedblock_out)", anchor="w", wraplength=340)  # type: ignore[attr-defined]
    app.lbl_outdir.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))

    # ---- Run controls ----
    # Управление запуском/пауза/стоп.
    frm_play = ctk.CTkFrame(left)
    frm_play.grid(row=7, column=0, sticky="ew", padx=12, pady=(0, 8))
    frm_play.grid_columnconfigure(0, weight=1)
    frm_play.grid_columnconfigure(1, weight=1)

    app.btn_start = ctk.CTkButton(frm_play, text="Старт", command=app._on_start)  # type: ignore[attr-defined]
    app.btn_pause = ctk.CTkButton(frm_play, text="Пауза", command=app._on_pause_toggle)  # type: ignore[attr-defined]
    app.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=10)
    app.btn_pause.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=10)

    app.btn_stop = ctk.CTkButton(left, text="Стоп", command=app._on_stop, fg_color="#7a2222", hover_color="#8f2a2a")  # type: ignore[attr-defined]
    app.btn_stop.grid(row=8, column=0, sticky="ew", padx=12, pady=(0, 12))

    app.realtime_var = tk.BooleanVar(value=True)  # type: ignore[attr-defined]
    app.chk_realtime = ctk.CTkCheckBox(left, text="Воспроизведение по FPS (реальное время)", variable=app.realtime_var)  # type: ignore[attr-defined]
    app.chk_realtime.grid(row=9, column=0, sticky="w", padx=12, pady=(0, 10))

    app.progress = ctk.CTkProgressBar(left)  # type: ignore[attr-defined]
    app.progress.grid(row=10, column=0, sticky="ew", padx=12, pady=(0, 8))
    app.progress.set(0.0)  # type: ignore[attr-defined]
    app.lbl_status = ctk.CTkLabel(left, text="Статус: ожидание", anchor="w", justify="left", wraplength=360)  # type: ignore[attr-defined]
    app.lbl_status.grid(row=11, column=0, sticky="ew", padx=12, pady=(0, 12))

    # ---- Bind updates for labels / realtime controls (moved from main_window.py) ----
    # Trace'ы обновляют подписи рядом со слайдерами/переключателями без ручного refresh.
    app.conf_var.trace_add("write", lambda *_: app.conf_lbl.configure(text=f"{app.conf_var.get():.2f}"))  # type: ignore[attr-defined]
    app.min_area_var.trace_add("write", lambda *_: app.min_area_lbl.configure(text=f"{app.min_area_var.get():.2f}%"))  # type: ignore[attr-defined]
    app._auto_roi_every_n_frames_var.trace_add(  # type: ignore[attr-defined]
        "write",
        lambda *_: app._auto_roi_every_n_lbl.configure(text=str(int(app._auto_roi_every_n_frames_var.get() or 1))),  # type: ignore[attr-defined]
    )

    # ROI realtime (values + apply)
    # Любое движение слайдера вызывает пересчёт ROI и перерисовку превью (и, если запущено, обновляет процессор).
    app.roi_x.trace_add("write", lambda *_: app._on_roi_change())  # type: ignore[attr-defined]
    app.roi_y.trace_add("write", lambda *_: app._on_roi_change())  # type: ignore[attr-defined]
    app.roi_w.trace_add("write", lambda *_: app._on_roi_change())  # type: ignore[attr-defined]
    app.roi_h.trace_add("write", lambda *_: app._on_roi_change())  # type: ignore[attr-defined]
    app._update_roi_value_labels()  # type: ignore[attr-defined]


