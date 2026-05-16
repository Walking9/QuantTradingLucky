"""Trend indicators implementation.

This module contains implementations for common trend-following indicators:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Moving Average Convergence Divergence (MACD)
"""

import numpy as np
import pandas as pd

from .base import check_input_series


def sma(series: pd.Series | np.ndarray, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        series: The input time series data (typically close prices).
        window: The lookback window size.

    Returns:
        pd.Series: The SMA series.

    Raises:
        ValueError: If window is less than 1.
    """
    if window < 1:
        raise ValueError("Window must be at least 1")

    series = check_input_series(series)
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series | np.ndarray, window: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        series: The input time series data.
        window: The lookback window size.

    Returns:
        pd.Series: The EMA series.

    Raises:
        ValueError: If window is less than 1.
    """
    if window < 1:
        raise ValueError("Window must be at least 1")

    series = check_input_series(series)
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def macd(
    series: pd.Series | np.ndarray,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD).

    Args:
        series: The input time series data.
        fast_window: The fast EMA window.
        slow_window: The slow EMA window.
        signal_window: The signal line EMA window.

    Returns:
        pd.DataFrame: A DataFrame containing 'macd', 'macd_signal', and 'macd_hist'.

    Raises:
        ValueError: If any window is less than 1 or if fast_window >= slow_window.
    """
    if fast_window < 1 or slow_window < 1 or signal_window < 1:
        raise ValueError("All windows must be at least 1")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    series = check_input_series(series)

    ema_fast = ema(series, fast_window)
    ema_slow = ema(series, slow_window)

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist}, index=series.index
    )
