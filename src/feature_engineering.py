from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "daily_return", "return_lag_1", "return_lag_2", "return_lag_3",
    "return_lag_5", "return_lag_10", "momentum_5", "momentum_10",
    "momentum_20", "rolling_mean_5", "rolling_volatility_20",
    "high_low_range", "open_close_change", "volume_change",
]


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = data.sort_values("date").reset_index(drop=True).copy()
    frame["daily_return"] = frame["close"].pct_change()
    for lag in [1, 2, 3, 5, 10]:
        frame[f"return_lag_{lag}"] = frame["daily_return"].shift(lag)
    for window in [5, 10, 20]:
        frame[f"momentum_{window}"] = frame["close"].pct_change(window)
    frame["rolling_mean_5"] = frame["daily_return"].rolling(5).mean()
    frame["rolling_volatility_20"] = frame["daily_return"].rolling(20).std()
    frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"] if "high" in frame else 0.0
    frame["open_close_change"] = (frame["close"] - frame["open"]) / frame["open"] if "open" in frame else 0.0
    frame["volume_change"] = frame["volume"].pct_change() if "volume" in frame else 0.0
    frame["target_return"] = frame["close"].shift(-1) / frame["close"] - 1
    usable = [name for name in FEATURE_NAMES if name in frame.columns]
    result = frame.dropna(subset=usable + ["target_return"]).reset_index(drop=True)
    return result, usable

