import numpy as np
import pandas as pd

from src.feature_engineering import build_features
from src.market_analysis import metrics
from src.model_evaluation import directional_accuracy, evaluate
from src.model_training import chronological_split
from src.preprocessing import prepare_series


def sample_frame(n=40):
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "close": np.arange(100, 100 + n, dtype=float),
        "index": "Sensex",
        "open": np.arange(99, 99 + n, dtype=float),
        "high": np.arange(101, 101 + n, dtype=float),
        "low": np.arange(98, 98 + n, dtype=float),
        "volume": np.arange(1000, 1000 + n),
    })


def test_returns_cumulative_volatility_and_drawdown():
    result = prepare_series(sample_frame())
    assert result["daily_return"].iloc[1] > 0
    assert result["cumulative_return"].iloc[-1] > 0
    assert result["drawdown"].max() == 0


def test_target_uses_next_observation_and_features_do_not_use_future():
    frame, features = build_features(sample_frame())
    row = frame.iloc[0]
    source = sample_frame()
    source_position = source.index[source["date"] == row["date"]][0]
    expected = source["close"].iloc[source_position + 1] / source["close"].iloc[source_position] - 1
    assert np.isclose(row["target_return"], expected)
    changed = sample_frame()
    changed.loc[len(changed) - 1, "close"] = 100000
    changed_frame, _ = build_features(changed)
    assert np.isclose(frame.iloc[0]["daily_return"], changed_frame.iloc[0]["daily_return"])
    assert "target_return" not in features


def test_chronological_split_and_metrics():
    frame, _ = build_features(sample_frame(60))
    train, validation, test = chronological_split(frame)
    assert train["date"].max() < validation["date"].min() < test["date"].min()
    values = evaluate([-.1, .1, 0], [-.05, .05, 0])
    assert values["rmse"] >= 0
    assert directional_accuracy([-.1, .1], [-.01, .02]) == 1
