from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Allow running without editable install (src/ layout)."""
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from pedblock.app import main  # noqa: E402


if __name__ == "__main__":
    main()


