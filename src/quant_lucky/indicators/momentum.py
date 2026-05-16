"""Momentum and Volatility indicators implementation.

This module contains implementations for common momentum and volatility indicators:
- Relative Strength Index (RSI)
- Bollinger Bands (BB)
- Average True Range (ATR)
"""

import numpy as np
import pandas as pd

from .base import check_input_df, check_input_series
from .trend import sma


def rsi(series: pd.Series | np.ndarray, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Uses the Wilder's Smoothing Method (equivalent to an EMA with alpha=1/window).

    Args:
        series: The input time series data.
        window: The lookback window size.

    Returns:
        pd.Series: The RSI series bounded between 0 and 100.

    Raises:
        ValueError: If window is less than 1.
    """
    if window < 1:
        raise ValueError("Window must be at least 1")

    series = check_input_series(series)

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing is exactly EMA with alpha=1/window
    # pandas ewm(alpha=...) requires pandas >= 1.2
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    # Handle the case where avg_loss is 0
    rsi_val = rsi_val.fillna(100).where(avg_loss != 0, 100.0)
    # The first (window-1) values should be NaN
    rsi_val.iloc[:window] = np.nan

    return rsi_val


def bollinger_bands(
    series: pd.Series | np.ndarray, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Args:
        series: The input time series data.
        window: The moving average window size.
        num_std: The number of standard deviations for upper/lower bands.

    Returns:
        pd.DataFrame: A DataFrame containing 'bb_upper', 'bb_mid', 'bb_lower', and 'bb_width'.

    Raises:
        ValueError: If window is less than 1 or num_std is negative.
    """
    if window < 1:
        raise ValueError("Window must be at least 1")
    if num_std < 0:
        raise ValueError("Number of standard deviations must be non-negative")

    series = check_input_series(series)

    mid_band = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)

    upper_band = mid_band + (std * num_std)
    lower_band = mid_band - (std * num_std)
    width = (upper_band - lower_band) / mid_band

    return pd.DataFrame(
        {"bb_upper": upper_band, "bb_mid": mid_band, "bb_lower": lower_band, "bb_width": width},
        index=series.index,
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate the True Range (TR).

    TR is the maximum of:
    1. High - Low
    2. |High - Previous Close|
    3. |Low - Previous Close|

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns.

    Returns:
        pd.Series: The True Range series.
    """
    check_input_df(df, ["high", "low", "close"])

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # Calculate max across columns
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR).

    Uses the Wilder's Smoothing Method for the average.

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns.
        window: The lookback window size.

    Returns:
        pd.Series: The ATR series.

    Raises:
        ValueError: If window is less than 1.
    """
    if window < 1:
        raise ValueError("Window must be at least 1")

    tr = true_range(df)

    # ATR typically uses Wilder's smoothing (alpha=1/window)
    atr_val = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr_val
