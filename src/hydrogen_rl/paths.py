from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"
SOURCE_MODEL_PATH = MODEL_DIR / "sourcefn_model.pth"
PPO_MODEL_PATH = MODEL_DIR / "best_model.zip"

for directory in (MODEL_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
