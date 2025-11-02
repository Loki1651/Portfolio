from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainingConfig:
    model_path: Path
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None


class AQIPredictor:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.config.test_size, random_state=self.config.random_state
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        metrics = {
            "rmse": mean_squared_error(y_test, predictions, squared=False),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": self.model.score(X_test, y_test),
        }

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        return metrics

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            if not self.config.model_path.exists():
                raise FileNotFoundError("Model file not found. Train or provide a model before predicting.")
            self.model = joblib.load(self.config.model_path)
        return pd.Series(self.model.predict(features), index=features.index)
