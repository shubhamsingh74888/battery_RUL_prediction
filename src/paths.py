"""
paths.py
--------
Single source of truth for every file path used in this project.
Anchors all paths to the project root so scripts run correctly
from any working directory.
"""
from pathlib import Path

# battery-rul/src/paths.py -> .parent -> src/ -> .parent -> battery-rul/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
RAW_MAT_DIR   = DATA_DIR / "raw_mat"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV  = PROCESSED_DIR / "battery_features.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"

for _dir in (DATA_DIR, RAW_MAT_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"RAW_MAT_DIR  = {RAW_MAT_DIR}")
    print(f"FEATURES_CSV = {FEATURES_CSV}")
    print(f"MODELS_DIR   = {MODELS_DIR}")
    print(f"REPORTS_DIR  = {REPORTS_DIR}")
