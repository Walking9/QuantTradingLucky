"""Base interface for technical indicators."""

import numpy as np
import pandas as pd


def check_input_series(series: pd.Series | np.ndarray) -> pd.Series:
    """Ensure the input is a pandas Series.

    Args:
        series: The input time series data.

    Returns:
        pd.Series: The validated pandas Series.

    Raises:
        TypeError: If the input is not a pandas Series or numpy array.
        ValueError: If the input is empty.
    """
    if len(series) == 0:
        raise ValueError("Input series cannot be empty")

    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series or numpy array, got {type(series)}")

    return series


def check_input_df(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure the input DataFrame contains the required columns.

    Args:
        df: The input DataFrame (e.g., OHLCV).
        required_columns: List of column names that must be present.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If required columns are missing or dataframe is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
