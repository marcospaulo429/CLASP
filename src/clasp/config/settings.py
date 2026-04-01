from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

