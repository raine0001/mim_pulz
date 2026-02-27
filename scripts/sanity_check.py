from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mim_pulz.submit import run


def main() -> None:
    out = run("submission_sanity.csv", dry_run=True, dry_rows=8)
    print("Sanity submission written to:", out)


if __name__ == "__main__":
    main()
