from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import BinaryIO

import pandas as pd

ALIASES = {
    "date": ["date", "datetime", "timestamp", "time"],
    "open": ["open", "opening", "openprice"],
    "high": ["high", "highprice"],
    "low": ["low", "lowprice"],
    "close": ["close", "closing", "closeprice", "price", "adjclose", "adjustedclose"],
    "volume": ["volume", "vol", "shares"],
    "index": ["index", "symbol", "ticker", "name", "series", "market"],
}


@dataclass
class DataQuality:
    source: str
    rows_loaded: int
    rows_cleaned: int
    coverage_start: str | None
    coverage_end: str | None
    missing: dict[str, int]
    duplicate_rows: int
    warnings: list[str]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize_column_name(c): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for key, original in normalized.items():
        if any(alias in key for alias in aliases):
            return original
    return None


def _read_source(source: str | Path | BinaryIO) -> tuple[pd.DataFrame, str]:
    if hasattr(source, "getvalue"):
        return pd.read_csv(source), getattr(source, "name", "uploaded CSV")
    path = Path(source)
    return pd.read_csv(path), str(path)


def load_market_data(source: str | Path | BinaryIO, start="2000-01-01", end="2020-12-31") -> tuple[pd.DataFrame, DataQuality]:
    raw, source_name = _read_source(source)
    if raw.empty:
        raise ValueError("The CSV contains no rows.")
    original_rows = len(raw)
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        found = _find_column(list(raw.columns), aliases)
        if found:
            mapping[target] = found
    if "date" not in mapping or "close" not in mapping:
        raise ValueError("CSV must contain a date column and a close/price column.")
    data = pd.DataFrame()
    for target, column in mapping.items():
        data[target] = raw[column]
    data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=False)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "close"]).copy()
    data = data[data["close"] > 0].copy()
    duplicate_rows = int(data.duplicated().sum())
    # Keep the last source record for an identical date/index key; do not average OHLC.
    keys = ["date"] + (["index"] if "index" in data else [])
    data = data.drop_duplicates(subset=keys, keep="last")
    data = data.sort_values(keys).reset_index(drop=True)
    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"Removed {duplicate_rows} exact duplicate row(s).")
    if "open" in data and "high" in data and (data["high"] < data["open"]).any():
        warnings.append("Some high values are below open; source OHLC values were not rewritten.")
    if "low" in data and "high" in data and (data["low"] > data["high"]).any():
        warnings.append("Some low values are above high; source OHLC values were not rewritten.")
    data = data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))].copy()
    data["index"] = data.get("index", pd.Series("Market", index=data.index)).fillna("Market").astype(str)
    data["index"] = data["index"].replace("", "Market")
    missing = {column: int(value) for column, value in data.isna().sum().items() if value}
    quality = DataQuality(
        source=source_name,
        rows_loaded=original_rows,
        rows_cleaned=len(data),
        coverage_start=data["date"].min().date().isoformat() if not data.empty else None,
        coverage_end=data["date"].max().date().isoformat() if not data.empty else None,
        missing=missing,
        duplicate_rows=duplicate_rows,
        warnings=warnings,
    )
    if data.empty:
        raise ValueError("No valid rows remain in the requested 2000–2020 period.")
    return data, quality


def discover_csvs(folder: str | Path = "data") -> list[Path]:
    return sorted(Path(folder).glob("*.csv"))

