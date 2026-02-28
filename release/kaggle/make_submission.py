from __future__ import annotations

from pathlib import Path
import runpy
import sys


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
ENTRY = SRC_DIR / "make_submission.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    runpy.run_path(str(ENTRY), run_name="__main__")
