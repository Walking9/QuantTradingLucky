"""Tests for the canonical OHLCV schema validator."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lucky.data.schema import Frequency, Market, validate_ohlcv


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000.0, 1_500.0],
        }
    )


class TestEnums:
    def test_market_values(self) -> None:
        assert Market.A_SHARE.value == "cn"
        assert Market.US.value == "us"
        assert Market.CRYPTO.value == "crypto"

    def test_frequency_values(self) -> None:
        assert Frequency.DAILY.value == "1d"
        assert Frequency.HOUR_1.value == "1h"
        assert Frequency.MINUTE_1.value == "1m"


class TestValidateOhlcv:
    def test_valid_df_passes(self) -> None:
        validate_ohlcv(_make_valid_df())  # no exception

    def test_missing_column_raises(self) -> None:
        df = _make_valid_df().drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_ohlcv(df)

    def test_non_datetime_timestamp_raises(self) -> None:
        df = _make_valid_df()
        df["timestamp"] = ["2024-01-01", "2024-01-02"]
        with pytest.raises(TypeError, match="datetime64"):
            validate_ohlcv(df)

    def test_non_numeric_price_raises(self) -> None:
        df = _make_valid_df()
        df["close"] = ["a", "b"]
        with pytest.raises(TypeError, match="must be numeric"):
            validate_ohlcv(df)

    def test_unsorted_timestamp_raises(self) -> None:
        df = _make_valid_df().iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="monotonically"):
            validate_ohlcv(df)

    def test_duplicated_timestamp_raises(self) -> None:
        df = _make_valid_df()
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        with pytest.raises(ValueError, match="duplicates"):
            validate_ohlcv(df)

    def test_high_less_than_low_raises(self) -> None:
        df = _make_valid_df()
        df.loc[0, "high"] = 50.0
        with pytest.raises(ValueError, match="high < low"):
            validate_ohlcv(df)

    def test_high_less_than_close_raises(self) -> None:
        df = _make_valid_df()
        df.loc[0, "close"] = 999.0
        with pytest.raises(ValueError, match="high < open or high < close"):
            validate_ohlcv(df)

    def test_low_greater_than_open_raises(self) -> None:
        df = _make_valid_df()
        df.loc[0, "low"] = 500.0  # above open/close but need to also set high
        df.loc[0, "high"] = 600.0
        with pytest.raises(ValueError, match="low > open or low > close"):
            validate_ohlcv(df)

    def test_negative_volume_raises(self) -> None:
        df = _make_valid_df()
        df.loc[0, "volume"] = -1.0
        with pytest.raises(ValueError, match="negative volume"):
            validate_ohlcv(df)

    def test_nan_raises_by_default(self) -> None:
        df = _make_valid_df()
        df.loc[0, "close"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_ohlcv(df)

    def test_nan_allowed_with_flag(self) -> None:
        df = _make_valid_df()
        df.loc[0, "close"] = float("nan")
        validate_ohlcv(df, allow_nan=True)  # no exception

    def test_empty_df_raises(self) -> None:
        df = _make_valid_df().iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            validate_ohlcv(df)
