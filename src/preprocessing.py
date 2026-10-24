from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_series(data: pd.DataFrame, index_name: str | None = None) -> pd.DataFrame:
    frame = data.copy()
    if index_name and index_name != "All":
        frame = frame[frame["index"] == index_name].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["daily_return"] = frame["close"].pct_change()
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["cumulative_return"] = (1 + frame["daily_return"].fillna(0)).cumprod() - 1
    frame["rolling_volatility"] = frame["daily_return"].rolling(20, min_periods=5).std() * np.sqrt(252)
    frame["running_peak"] = frame["close"].cummax()
    frame["drawdown"] = frame["close"] / frame["running_peak"] - 1
    return frame


def filter_dates(data: pd.DataFrame, start, end) -> pd.DataFrame:
    return data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))].copy()

