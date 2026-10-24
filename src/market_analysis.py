from __future__ import annotations

import numpy as np
import pandas as pd


def annual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(year=frame["date"].dt.year).groupby("year", as_index=False).agg(
        annual_return=("daily_return", lambda values: (1 + values.dropna()).prod() - 1),
        observations=("close", "size"),
    )


def top_movements(frame: pd.DataFrame, n=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = frame.dropna(subset=["daily_return"])
    return returns.nsmallest(n, "daily_return"), returns.nlargest(n, "daily_return")


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = frame["daily_return"].dropna()
    return {
        "latest_close": float(frame["close"].iloc[-1]),
        "total_return": float((1 + returns).prod() - 1) if len(returns) else 0.0,
        "annualized_volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0,
        "max_drawdown": float(frame["drawdown"].min()),
        "observations": int(len(frame)),
    }

