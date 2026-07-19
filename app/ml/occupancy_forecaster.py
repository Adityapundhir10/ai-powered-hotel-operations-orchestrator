from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "lead_time",
    "booking_pickup_7d",
    "rolling_demand_14d",
    "cancellation_rate_28d",
    "available_inventory",
    "event_intensity",
    "lag_occupancy_7d",
    "lag_occupancy_28d",
    "day_of_week",
    "month",
    "is_weekend",
]
TARGET = "occupancy_rate"


@dataclass
class TrainingMetrics:
    mae: float
    rmse: float
    r2: float
    backend: str
    rows: int


class OccupancyForecaster:
    """XGBoost occupancy forecaster with a RandomForest fallback.

    Chronological validation is used by the training script to reduce temporal
    leakage. The class intentionally stores the feature order with the model.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.model: Any | None = None
        self.backend = "untrained"
        if self.model_path and self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.backend = bundle.get("backend", type(self.model).__name__)

    @staticmethod
    def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "forecast_date" in frame.columns:
            dt = pd.to_datetime(frame["forecast_date"])
            frame["day_of_week"] = dt.dt.dayofweek
            frame["month"] = dt.dt.month
            frame["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        missing = [name for name in FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing forecasting features: {missing}")
        return frame

    def _build_model(self, random_state: int = 42):
        if XGBRegressor is not None:
            self.backend = "XGBoost"
            return XGBRegressor(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.045,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_alpha=0.05,
                reg_lambda=1.2,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=2,
            )
        self.backend = "RandomForest fallback"
        return RandomForestRegressor(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=2,
        )

    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame | None = None) -> TrainingMetrics:
        train = self.prepare_frame(train_df)
        self.model = self._build_model()
        self.model.fit(train[FEATURES], train[TARGET])
        evaluation = self.prepare_frame(validation_df) if validation_df is not None else train
        prediction = np.clip(self.model.predict(evaluation[FEATURES]), 0.03, 0.99)
        metrics = TrainingMetrics(
            mae=float(mean_absolute_error(evaluation[TARGET], prediction)),
            rmse=float(math.sqrt(mean_squared_error(evaluation[TARGET], prediction))),
            r2=float(r2_score(evaluation[TARGET], prediction)),
            backend=self.backend,
            rows=len(train),
        )
        return metrics

    def save(self, model_path: str | Path | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("Train the model before saving it.")
        path = Path(model_path or self.model_path or "models/occupancy_xgb.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "backend": self.backend, "features": FEATURES}, path)
        self.model_path = path
        return path

    def predict(self, feature_row: dict[str, Any]) -> float:
        frame = self.prepare_frame(pd.DataFrame([feature_row]))
        if self.model is None:
            # Deterministic demand formula for zero-setup local use.
            row = frame.iloc[0]
            raw = (
                0.18
                + 0.42 * float(row["rolling_demand_14d"])
                + 0.22 * float(row["lag_occupancy_7d"])
                + 0.12 * float(row["lag_occupancy_28d"])
                + 0.004 * min(float(row["booking_pickup_7d"]), 50)
                + 0.08 * float(row["event_intensity"])
                + 0.04 * float(row["is_weekend"])
                - 0.20 * float(row["cancellation_rate_28d"])
                - 0.0015 * float(row["available_inventory"])
                - 0.00035 * float(row["lead_time"])
            )
            self.backend = "deterministic fallback"
            return float(np.clip(raw, 0.05, 0.98))
        return float(np.clip(self.model.predict(frame[FEATURES])[0], 0.03, 0.99))

    def feature_contributions(self, feature_row: dict[str, Any]) -> dict[str, float]:
        frame = self.prepare_frame(pd.DataFrame([feature_row]))
        if self.model is None:
            values = {
                "rolling_demand_14d": 0.42 * float(frame.iloc[0]["rolling_demand_14d"]),
                "lag_occupancy_7d": 0.22 * float(frame.iloc[0]["lag_occupancy_7d"]),
                "booking_pickup_7d": 0.004 * min(float(frame.iloc[0]["booking_pickup_7d"]), 50),
                "event_intensity": 0.08 * float(frame.iloc[0]["event_intensity"]),
                "cancellation_rate_28d": -0.20 * float(frame.iloc[0]["cancellation_rate_28d"]),
            }
            return {k: round(v, 5) for k, v in values.items()}
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(frame[FEATURES])
            row_values = np.asarray(shap_values)[0]
            return {name: round(float(value), 5) for name, value in zip(FEATURES, row_values)}
        except Exception:
            importances = getattr(self.model, "feature_importances_", np.zeros(len(FEATURES)))
            return {name: round(float(value), 5) for name, value in zip(FEATURES, importances)}
