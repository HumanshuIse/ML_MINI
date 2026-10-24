from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_evaluation import evaluate


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    if n < 15:
        raise ValueError("At least 15 usable observations are required for train/validation/test evaluation.")
    train_end = max(1, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    return frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]


def model_factories() -> dict[str, object]:
    return {
        "Linear Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=0.1, random_state=42))]),
    }


def train_models(frame: pd.DataFrame, features: list[str], selected: list[str] | None = None) -> dict[str, dict]:
    train, validation, test = chronological_split(frame)
    names = selected or list(model_factories())
    outputs: dict[str, dict] = {}
    x_train, y_train = train[features], train["target_return"]
    x_test, y_test = test[features], test["target_return"]
    for name in names:
        if name == "Zero-Return Baseline":
            prediction = np.zeros(len(test))
            fitted = None
        else:
            fitted = model_factories()[name]
            fitted.fit(x_train, y_train)
            prediction = fitted.predict(x_test)
        outputs[name] = {
            "model": fitted, "metrics": evaluate(y_test, prediction),
            "actual": y_test.to_numpy(), "predicted": np.asarray(prediction),
            "test_dates": test["date"].to_numpy(), "train_range": (train["date"].min(), train["date"].max()),
        }
    return outputs

