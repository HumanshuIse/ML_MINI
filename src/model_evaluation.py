from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def directional_accuracy(actual, predicted) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean(np.sign(actual) == np.sign(predicted)))


def evaluate(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    result = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 and np.std(predicted) > 0 else np.nan,
    }
    result["r2"] = float(r2_score(actual, predicted)) if len(actual) > 1 and np.std(actual) > 0 else np.nan
    return result

