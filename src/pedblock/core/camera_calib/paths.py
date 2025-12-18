from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """
    Ищем корень проекта (там, где лежит `pyproject.toml` или `requirements.txt`).
    Это надёжнее, чем фиксированный parents[N], т.к. файл мог быть перенесён.
    """
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    # fallback (на случай необычной структуры)
    return cur.parents[4] if len(cur.parents) >= 5 else cur.parents[-1]


def default_calib_path() -> str:
    root = _find_project_root(Path(__file__))
    return str(root / "data" / "camera_calib.json")


DEFAULT_CALIB_PATH = default_calib_path()

__all__ = [
    "DEFAULT_CALIB_PATH",
    "default_calib_path",
]


