from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedblock.core.camera_calib import DEFAULT_CALIB_PATH


@dataclass(frozen=True, slots=True)
class RoadAreaParams:
    """
    Параметры детекции проезжей части (drivable area) из одного кадра.

    Идея (лёгкая эвристика, без нейросети):
    - берём нижнюю часть кадра (где обычно дорога),
    - находим сильные границы/разметку (Canny),
    - выделяем кандидаты линий (HoughLinesP),
    - оставляем две доминирующие линии: «левая» (наклон < 0) и «правая» (наклон > 0),
    - строим полигон-коридор между ними и заливаем его => это и есть маска проезжей части.
    """

    # Анализируем только низ (в процентах высоты) — вверх обычно не относится к дороге.
    analyze_top_frac: float = 0.40  # 0..1 (0.40 => игнорируем верхние 40%)
    analyze_x_margin_frac: float = 0.06  # 0..0.45 обрезка слева/справа (шумы: обочины/бордюр/небо)

    # Горизонт начала «коридора» (где полигон сужается вверх)
    corridor_top_frac: float = 0.60

    # Canny / Hough
    canny1: int = 50
    canny2: int = 150
    hough_threshold: int = 60
    min_line_len_frac: float = 0.06  # доля ширины
    max_line_gap: int = 30

    # Фильтр линий по наклону
    min_abs_slope: float = 0.55

    # Немного расширяем коридор
    x_pad_frac: float = 0.05

    # Если коридор получается слишком узкий — fallback
    min_width_frac: float = 0.22

    # Фоллбек-трапеция (проценты)
    fallback_top_y: float = 60.0
    fallback_bottom_y: float = 98.0
    fallback_top_w: float = 40.0
    fallback_bottom_w: float = 80.0
    fallback_center_x: float = 50.0

    # Усиливаем разметку (белый/жёлтый) — помогает на «обычных» видеорегистраторах
    use_lane_color_mask: bool = True

    # Режим сегментации:
    # - "hough": только коридор между 2 линиями (быстро)
    # - "hybrid": коридор (геометрия) + цветовая сегментация + CC (обычно точнее)
    method: str = "hybrid"

    # OpenADAS-style: если есть калибровка гомографии, можно строить маску в bird-view и
    # проецировать обратно на исходный кадр.
    use_perspective: bool = False
    calib_path: str = DEFAULT_CALIB_PATH

    # Настройки цветовой сегментации (hybrid)
    color_space: str = "lab"  # lab|hsv
    # Откуда берём “эталонный цвет дороги”: полоса у низа кадра
    seed_bottom_band_frac: float = 0.14  # доля высоты (снизу)
    seed_center_width_frac: float = 0.35  # доля ширины (по центру)
    # Порог “похожести на дорогу” (чем меньше — тем строже)
    # В LAB: это порог для нормированного квадр. расстояния (по каналам).
    color_dist_thresh: float = 7.5
    # Морфология после порога
    close_ksize: int = 11
    open_ksize: int = 5
    # Минимальная площадь компоненты (в долях кадра) для принятия маски
    min_cc_area_frac: float = 0.03

    # ==== DangerZone-from-mask tuning ====
    # Какая часть маски (снизу) используется для построения трапеции danger_zone.
    dz_near_bottom_frac: float = 0.35
    # Насколько “отрезать” края маски (квантили). 0.06 => берём 6..94% по X,
    # чтобы игнорировать шумные островки у границ.
    dz_edge_quantile: float = 0.06
    # Доп. морфология перед извлечением границ danger_zone (устойчивость на дожде/бликах).
    dz_close_ksize: int = 15
    dz_open_ksize: int = 5
    # Минимальная ширина трапеции на ближней границе (доля кадра).
    dz_min_width_frac: float = 0.18
    # Позиции (фракции) внутри near-band по Y, по которым оцениваем границы и фитчим линии.
    dz_sample_fracs: tuple[float, float, float, float] = (0.10, 0.35, 0.65, 0.95)
    # Способ построения danger_zone из маски:
    # - "fit": фитим левую/правую границу как x = a*y + b по нескольким срезам
    # - "max_width": нижнее основание по строке с максимальной шириной маски (в near-band)
    # - "hough": боковые стороны из Canny+Hough по краям маски, верхнее основание фиксируем по y_top
    dz_method: str = "fit"


@dataclass(frozen=True, slots=True)
class RoadMaskResult:
    mask_u8: np.ndarray  # HxW uint8 {0,255}
    polygon_px: list[tuple[int, int]]  # по часовой: TL, TR, BR, BL (если удалось)


@dataclass(frozen=True, slots=True)
class RoadDebug:
    edges_u8: np.ndarray | None = None
    corridor_mask_u8: np.ndarray | None = None
    seed_mask_u8: np.ndarray | None = None
    color_dist_u8: np.ndarray | None = None  # 0..255 (меньше = ближе к “дороге”)
    color_cand_u8: np.ndarray | None = None
    cc_selected_u8: np.ndarray | None = None


