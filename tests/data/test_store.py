"""Parquet store round-trip and metadata tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import SCHEMA_VERSION, ParquetStore, _safe_symbol


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_500.0, 1_200.0],
        }
    )


class TestSafeSymbol:
    def test_slash_replaced(self) -> None:
        assert _safe_symbol("BTC/USDT") == "BTC-USDT"

    def test_colon_replaced(self) -> None:
        assert _safe_symbol("AAPL:NASDAQ") == "AAPL-NASDAQ"

    def test_plain_symbol_unchanged(self) -> None:
        assert _safe_symbol("AAPL") == "AAPL"


class TestParquetStore:
    def test_roundtrip(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        store = ParquetStore(root=tmp_path)
        store.write(
            sample_df,
            provider="test",
            symbol="ABC",
            frequency=Frequency.DAILY,
        )
        loaded = store.read("test", "ABC", Frequency.DAILY)
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_metadata_recorded(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        store = ParquetStore(root=tmp_path)
        store.write(
            sample_df,
            provider="test",
            symbol="ABC",
            frequency=Frequency.DAILY,
        )
        md = store.read_metadata("test", "ABC", Frequency.DAILY)
        assert md["provider"] == "test"
        assert md["symbol"] == "ABC"
        assert md["frequency"] == "1d"
        assert md["schema_version"] == SCHEMA_VERSION
        assert md["row_count"] == "3"
        assert "downloaded_at" in md
        assert md["start"].startswith("2024-01-01")
        assert md["end"].startswith("2024-01-03")

    def test_extra_metadata(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        store = ParquetStore(root=tmp_path)
        store.write(
            sample_df,
            provider="test",
            symbol="ABC",
            frequency=Frequency.DAILY,
            extra_metadata={"note": "synthetic fixture"},
        )
        md = store.read_metadata("test", "ABC", Frequency.DAILY)
        assert md["note"] == "synthetic fixture"

    def test_exists_before_and_after(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        assert not store.exists("test", "ABC", Frequency.DAILY)
        store.write(sample_df, provider="test", symbol="ABC", frequency=Frequency.DAILY)
        assert store.exists("test", "ABC", Frequency.DAILY)

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        store = ParquetStore(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.read("test", "ABC", Frequency.DAILY)

    def test_symbol_with_slash_safe_on_disk(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        store.write(
            sample_df,
            provider="ccxt.binance",
            symbol="BTC/USDT",
            frequency=Frequency.HOUR_1,
        )
        assert store.exists("ccxt.binance", "BTC/USDT", Frequency.HOUR_1)
        # Check the on-disk path uses the sanitised name
        path = store.path_for("ccxt.binance", "BTC/USDT", Frequency.HOUR_1)
        assert "BTC-USDT" in str(path)
        assert "/" not in path.parent.name  # directory name is safe

    def test_invalid_df_rejected(self, tmp_path: Path) -> None:
        store = ParquetStore(root=tmp_path)
        bad = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            store.write(bad, provider="test", symbol="ABC", frequency=Frequency.DAILY)
