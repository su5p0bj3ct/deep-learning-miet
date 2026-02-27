import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    SEED: int = 9999
    IMG_SIZE: int = 640
    BATCH_SIZE: int = 32
    VAL_BATCH_SIZE: int = 64
    ACCUM_STEPS: int = 2
    NUM_WORKERS: int = 4
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    USE_FOLDS: bool = True
    N_FOLDS: int = 5
    N_EPOCHS_STAGE1: int = 4
    N_EPOCHS_STAGE2: int = 30
    PATIENCE: int = 8

    CHECKPOINT_DIR: str = "/content/checkpoints"
    BEST_MODELS_DIR: str = "/content/best_models"
    SUBMISSION_PATH: str = "submission.csv"
    RESUME_CHECKPOINT: str = ""


cfg = Config()
