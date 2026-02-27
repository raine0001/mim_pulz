from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def is_kaggle() -> bool:
    # Kaggle sets KAGGLE_URL_BASE in notebooks
    return bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()

@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    outputs: Path = PROJECT_ROOT / "outputs"

    # Kaggle mounts competition datasets under /kaggle/input/<competition-slug>/
    kaggle_input_root: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")

PATHS = Paths()
