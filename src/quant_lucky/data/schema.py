"""Canonical OHLCV schema used throughout the project.

All providers must return DataFrames conforming to this schema. The
validation is strict by default — bad data should fail fast.

Canonical columns (in order):
    timestamp   datetime64[ns, UTC]   monotonically increasing
    open        float64
    high        float64
    low         float64
    close       float64
    volume      float64

Optional columns providers may add:
    adjusted_close, dividends, stock_splits, amount, turnover
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class Market(StrEnum):
    """Supported markets. Used for routing + storage partitioning."""

    A_SHARE = "cn"
    US = "us"
    CRYPTO = "crypto"
    FUTURES_CN = "futures_cn"
    HK = "hk"


class Frequency(StrEnum):
    """Bar frequencies. Values match ccxt's timeframe convention."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "adjusted_close",
    "dividends",
    "stock_splits",
    "amount",
    "turnover",
)


def validate_ohlcv(df: pd.DataFrame, *, allow_nan: bool = False) -> None:
    """Validate ``df`` against the canonical OHLCV schema.

    Raises:
        ValueError: schema or value violation.
        TypeError: column dtype violation.
    """
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty")

    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {sorted(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("`timestamp` column must be datetime64")

    for col in ("open", "high", "low", "close", "volume"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"`{col}` must be numeric, got {df[col].dtype}")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("`timestamp` must be monotonically increasing")

    if df["timestamp"].duplicated().any():
        raise ValueError("`timestamp` contains duplicates")

    if not allow_nan:
        numeric_cols = [c for c in OHLCV_COLUMNS if c != "timestamp"]
        if df[numeric_cols].isna().any().any():
            raise ValueError("OHLCV contains NaN (pass allow_nan=True to permit)")

    if (df["high"] < df["low"]).any():
        raise ValueError("Found rows where high < low")

    if ((df["high"] < df["open"]) | (df["high"] < df["close"])).any():
        raise ValueError("Found rows where high < open or high < close")

    if ((df["low"] > df["open"]) | (df["low"] > df["close"])).any():
        raise ValueError("Found rows where low > open or low > close")

    if (df["volume"] < 0).any():
        raise ValueError("Found negative volume")
