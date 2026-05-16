import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from quant_lucky.indicators import (
    atr,
    bollinger_bands,
    check_input_df,
    check_input_series,
    ema,
    macd,
    rsi,
    sma,
    true_range,
)


@pytest.fixture
def sample_series():
    """Create a sample pandas Series for testing."""
    return pd.Series(
        [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0],
        index=pd.date_range("2023-01-01", periods=10),
    )


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=5)
    return pd.DataFrame(
        {
            "high": [12.0, 13.0, 14.0, 13.0, 12.0],
            "low": [10.0, 11.0, 12.0, 10.0, 9.0],
            "close": [11.0, 12.0, 13.0, 11.0, 10.0],
        },
        index=dates,
    )


class TestBase:
    def test_check_input_series(self, sample_series):
        # Valid pandas Series
        result = check_input_series(sample_series)
        assert_series_equal(result, sample_series)

        # Valid numpy array
        arr = np.array([1, 2, 3])
        result = check_input_series(arr)
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, arr)

        # Empty input
        with pytest.raises(ValueError):
            check_input_series(pd.Series([]))

        # Invalid type
        with pytest.raises(TypeError):
            check_input_series([1, 2, 3])  # list is not accepted

    def test_check_input_df(self, sample_df):
        # Valid dataframe
        check_input_df(sample_df, ["high", "low", "close"])

        # Missing columns
        with pytest.raises(ValueError):
            check_input_df(sample_df, ["high", "volume"])

        # Empty dataframe
        with pytest.raises(ValueError):
            check_input_df(pd.DataFrame(), ["high"])

        # Invalid type
        with pytest.raises(TypeError):
            check_input_df([1, 2, 3], ["high"])


class TestTrend:
    def test_sma(self, sample_series):
        result = sma(sample_series, window=3)
        expected = sample_series.rolling(3).mean()
        assert_series_equal(result, expected)

        with pytest.raises(ValueError):
            sma(sample_series, window=0)

    def test_ema(self, sample_series):
        result = ema(sample_series, window=3)
        # Manually calculate EMA to verify pandas behavior
        expected = sample_series.ewm(span=3, adjust=False, min_periods=3).mean()
        assert_series_equal(result, expected)

    def test_macd(self, sample_series):
        # Needs more data points for MACD defaults, so using smaller windows
        result = macd(sample_series, fast_window=2, slow_window=4, signal_window=2)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["macd", "macd_signal", "macd_hist"]

        ema_fast = ema(sample_series, 2)
        ema_slow = ema(sample_series, 4)
        expected_macd = (ema_fast - ema_slow).rename("macd")

        assert_series_equal(result["macd"], expected_macd)

        with pytest.raises(ValueError):
            macd(sample_series, fast_window=4, slow_window=2)  # fast >= slow


class TestMomentum:
    def test_rsi(self, sample_series):
        # Using a small window for the short sample series
        result = rsi(sample_series, window=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        # First 3 elements should be NaN (since window=3 and we use 1-based index logic where min_periods=window)
        assert np.isnan(result.iloc[0:3]).all()
        # RSI bounds
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_bollinger_bands(self, sample_series):
        result = bollinger_bands(sample_series, window=3, num_std=2.0)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["bb_upper", "bb_mid", "bb_lower", "bb_width"]

        expected_mid = sma(sample_series, 3).rename("bb_mid")
        assert_series_equal(result["bb_mid"], expected_mid)
        # Upper should be strictly greater than lower when std > 0
        valid_idx = result.dropna().index
        assert (result.loc[valid_idx, "bb_upper"] >= result.loc[valid_idx, "bb_lower"]).all()

    def test_true_range(self, sample_df):
        result = true_range(sample_df)
        assert isinstance(result, pd.Series)

        # Calculate expected values manually for the first two rows
        # Row 1 (index 0): High(12) - Low(10) = 2.0 (No previous close)
        # Row 2 (index 1): High(13) - Low(11) = 2.0, |13-11| = 2.0, |11-11| = 0.0 -> max is 2.0
        assert result.iloc[0] == 2.0
        assert result.iloc[1] == 2.0

    def test_atr(self, sample_df):
        result = atr(sample_df, window=2)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)

        with pytest.raises(ValueError):
            atr(sample_df, window=0)
