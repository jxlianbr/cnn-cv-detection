"""XGBoost spectral baseline, hyperparameters fixed per
papers/spectral-baseline-specifications.md (Optuna-tuned in the original
study; not re-tuned here -- see Global Constraints in plan.md)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from src.config import XGBoostConfig

FEATURE_COLUMNS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "MNDWI", "PVSI"]


def train_baseline(train_df: pd.DataFrame, cfg: XGBoostConfig) -> xgb.XGBClassifier:
    X = train_df[FEATURE_COLUMNS].values
    y = (train_df["label"] == "pv").astype(int).values
    model = xgb.XGBClassifier(
        max_depth=cfg.max_depth,
        learning_rate=cfg.eta,
        n_estimators=cfg.n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].values
    return model.predict_proba(X)[:, 1]
