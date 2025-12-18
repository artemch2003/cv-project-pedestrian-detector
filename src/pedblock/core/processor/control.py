from __future__ import annotations

from dataclasses import replace
import queue

from pedblock.core.config import RoiPct
from pedblock.core.danger_zone import DangerZonePct
from pedblock.core.road_area import RoadAreaParams

from .models import _ControlCmd
from .protocols import ControlTarget


def _to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return bool(v)


def _clamped_float(v: object, *, lo: float, hi: float) -> float:
    try:
        fv = float(v)  # type: ignore[arg-type]
    except Exception:
        return float(lo)
    return max(float(lo), min(float(hi), fv))


def _parse_dz_method(current: str, v: object | None) -> str:
    # Backward compatibility: historically encoded as float (0/1/2).
    if v is None:
        return str(current or "fit")
    if isinstance(v, (int, float)):
        fv = float(v)
        if fv >= 1.5:
            return "hough"
        if fv >= 0.5:
            return "max_width"
        return "fit"
    s = str(v).strip().lower()
    if s in ("max_width", "maxwidth", "max-width", "max"):
        return "max_width"
    if s in ("hough", "lines", "mask_hough", "mask-lines"):
        return "hough"
    return "fit"


def drain_ctrl(vp: ControlTarget) -> None:
    """
    Считывает команды из очереди управления и применяет их к состоянию VideoProcessor.
    (Вынесено из `VideoProcessor._drain_ctrl` ради читаемости.)
    """
    while True:
        try:
            cmd: _ControlCmd = vp._ctrl_queue.get_nowait()
        except queue.Empty:
            break

        if cmd.kind == "pause":
            vp._paused = True  # type: ignore[attr-defined]
        elif cmd.kind == "resume":
            vp._paused = False  # type: ignore[attr-defined]
        elif cmd.kind == "realtime":
            vp._realtime = bool(cmd.value)  # type: ignore[attr-defined]
        elif cmd.kind == "speed":
            v = float(cmd.value) if cmd.value is not None else 1.0
            vp._speed = max(vp._SPEED_MIN, min(vp._SPEED_MAX, v))
        elif cmd.kind == "roi":
            try:
                x, y, w, h = cmd.value  # type: ignore[misc]
                vp._roi_pct = RoiPct(x=float(x), y=float(y), w=float(w), h=float(h)).clamp()
                # If user sets a rectangle ROI manually, treat it as a quad (rectangle).
                vp._danger_zone_pct = DangerZonePct.from_quad(
                    vp._roi_pct.x,
                    vp._roi_pct.y,
                    vp._roi_pct.x + vp._roi_pct.w,
                    vp._roi_pct.y,
                    vp._roi_pct.x + vp._roi_pct.w,
                    vp._roi_pct.y + vp._roi_pct.h,
                    vp._roi_pct.x,
                    vp._roi_pct.y + vp._roi_pct.h,
                ).clamp()
                vp._danger_zone_manual_override = False
            except Exception:
                pass
        elif cmd.kind == "danger_zone":
            try:
                flat = list(cmd.value)  # type: ignore[arg-type]
                pts: list[tuple[float, float]] = []
                for i in range(0, len(flat) - 1, 2):
                    pts.append((float(flat[i]), float(flat[i + 1])))
                vp._danger_zone_pct = DangerZonePct(points=pts).clamp()
                vp._danger_zone_manual_override = bool(vp._danger_zone_mode != vp._DZ_MODE_ROAD)
            except Exception:
                pass
        elif cmd.kind == "danger_zone_mode":
            try:
                m = str(cmd.value or "roi").strip().lower()
                if m not in (vp._DZ_MODE_ROI, vp._DZ_MODE_ROAD):
                    m = vp._DZ_MODE_ROI
                vp._danger_zone_mode = m
                vp._danger_zone_manual_override = False
                if m == vp._DZ_MODE_ROI:
                    vp._danger_zone_pct = DangerZonePct.from_quad(
                        vp._roi_pct.x,
                        vp._roi_pct.y,
                        vp._roi_pct.x + vp._roi_pct.w,
                        vp._roi_pct.y,
                        vp._roi_pct.x + vp._roi_pct.w,
                        vp._roi_pct.y + vp._roi_pct.h,
                        vp._roi_pct.x,
                        vp._roi_pct.y + vp._roi_pct.h,
                    ).clamp()
            except Exception:
                pass
        elif cmd.kind == "show_road_mask":
            try:
                vp._show_road_mask = bool(cmd.value)
            except Exception:
                pass
        elif cmd.kind == "road_params":
            try:
                d = dict(cmd.value or {})  # type: ignore[arg-type]
                updates: dict[str, object] = {}

                if "color_dist_thresh" in d:
                    updates["color_dist_thresh"] = _clamped_float(d["color_dist_thresh"], lo=1.0, hi=60.0)
                if "use_perspective" in d:
                    updates["use_perspective"] = _to_bool(d["use_perspective"])
                if "dz_near_bottom_frac" in d:
                    updates["dz_near_bottom_frac"] = _clamped_float(d["dz_near_bottom_frac"], lo=0.05, hi=0.95)
                if "dz_edge_quantile" in d:
                    updates["dz_edge_quantile"] = _clamped_float(d["dz_edge_quantile"], lo=0.0, hi=0.20)
                if "dz_method" in d:
                    cur = str(getattr(vp._road_params, "dz_method", "fit"))
                    updates["dz_method"] = _parse_dz_method(cur, d.get("dz_method"))

                if updates:
                    rp = vp._road_params if isinstance(vp._road_params, RoadAreaParams) else RoadAreaParams()  # type: ignore[attr-defined]
                    vp._road_params = replace(rp, **updates)
            except Exception:
                pass


