"""Configuration management for ML pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model hyperparameters and configuration."""
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    alpha: float = 0.1
    l1_ratio: float = 0.5
    random_state: int = 42


@dataclass
class PathConfig:
    """Path configuration."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]

        self.repo_root = repo_root
        self.data_dir = repo_root / "data"
        self.models_dir = repo_root / "ml_models"
        self.train_data = self.data_dir / "training" / "train_data.parquet"
        self.test_data = self.data_dir / "training" / "test_data.parquet"
