"""Model training pipeline."""

import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet
from ml.config.config import ModelConfig
from ml.utils.validation import validate_data, detect_data_drift
from utils.logging_utils import setup_logger


class RegressionTrainer:
    """Regression model training with validation using ElasticNet."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.logger = setup_logger(__name__)
        self.model_name = "elasticnet"

    def load_and_validate_data(self, data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate training data."""
        self.logger.info(f"Loading data from {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_parquet(data_path)

        # Validate data
        required_columns = ["target_price"]
        validate_data(df, required_columns)

        y = df["target_price"]
        X = df.drop(columns=["target_price"])

        self.logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and validation sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=None,  # For regression
        )

        # Check for data drift
        drift_columns = detect_data_drift(X_train, X_test)
        if drift_columns:
            self.logger.warning(
                f"Potential data drift detected in columns: {drift_columns}"
            )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ElasticNet:
        """Train the regression model."""
        model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=self.config.random_state,
            max_iter=1000,  # Ensure convergence
        )

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        self.logger.info(
            f"Cross-validation R² scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

        # Fit model
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self, model: ElasticNet, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """Evaluate model performance."""
        y_pred = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test))
            * 100,  # Mean Absolute Percentage Error
        }

        return metrics

    def save_model(self, model: ElasticNet, models_dir: Path) -> Path:
        """Save model with timestamp."""
        models_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"{timestamp}_{self.model_name}.pkl"

        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def train_and_evaluate_model(self, train_data_path: Path, models_dir: Path) -> Path:
        """Complete training pipeline."""
        try:
            # Load and prepare data
            X, y = self.load_and_validate_data(train_data_path)
            X_train, X_test, y_train, y_test = self.split_data(X, y)

            # Train model
            model = self.train_model(X_train, y_train)

            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            self.logger.info(f"Evaluation metrics: {metrics}")

            # Save model locally
            model_path = self.save_model(model, models_dir)

            return model_path

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "data" / "training" / "training_dataset.parquet"
    models_dir = repo_root / "ml_models"

    model_config = ModelConfig()

    trainer = RegressionTrainer(model_config)
    trainer.train_and_evaluate_model(data_path, models_dir)
