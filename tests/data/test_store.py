"""Parquet store round-trip and metadata tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import (
    SCHEMA_VERSION,
    InventoryEntry,
    ParquetStore,
    _safe_symbol,
)


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


class TestInventory:
    """Tests for ``ParquetStore.iter_inventory()`` — the local cache index."""

    def _populate(
        self,
        store: ParquetStore,
        sample_df: pd.DataFrame,
        items: list[tuple[str, str, Frequency]],
    ) -> None:
        for provider, symbol, freq in items:
            store.write(sample_df, provider=provider, symbol=symbol, frequency=freq)

    def test_empty_root(self, tmp_path: Path) -> None:
        store = ParquetStore(root=tmp_path)
        assert list(store.iter_inventory()) == []

    def test_single_file_has_full_metadata(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        self._populate(store, sample_df, [("yfinance", "AAPL", Frequency.DAILY)])

        entries = list(store.iter_inventory())
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, InventoryEntry)
        assert e.provider == "yfinance"
        assert e.symbol == "AAPL"
        assert e.frequency == "1d"
        assert e.path.exists()
        assert e.size_bytes > 0
        assert e.row_count == len(sample_df)
        assert e.start is not None and e.start.date().isoformat() == "2024-01-01"
        assert e.end is not None and e.end.date().isoformat() == "2024-01-03"
        assert e.downloaded_at is not None
        assert e.schema_version == SCHEMA_VERSION
        assert e.is_healthy is True
        assert e.error is None

    def test_deterministic_sort_order(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        self._populate(
            store,
            sample_df,
            [
                ("yfinance", "SPY", Frequency.DAILY),
                ("ccxt.binance", "BTC-USDT", Frequency.DAILY),
                ("yfinance", "AAPL", Frequency.DAILY),
                ("yfinance", "AAPL", Frequency.HOUR_1),
            ],
        )
        entries = list(store.iter_inventory())
        keys = [(e.provider, e.symbol, e.frequency) for e in entries]
        assert keys == sorted(keys)

    def test_filter_by_provider(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        self._populate(
            store,
            sample_df,
            [
                ("yfinance", "AAPL", Frequency.DAILY),
                ("ccxt.binance", "BTC-USDT", Frequency.DAILY),
                ("ccxt.kraken", "BTC-USDT", Frequency.HOUR_1),
            ],
        )
        entries = list(store.iter_inventory(provider="ccxt.binance"))
        assert [e.provider for e in entries] == ["ccxt.binance"]

    def test_filter_by_symbol_substring_case_insensitive(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        self._populate(
            store,
            sample_df,
            [
                ("yfinance", "AAPL", Frequency.DAILY),
                ("ccxt.binance", "BTC/USDT", Frequency.DAILY),
                ("ccxt.kraken", "BTC/USDT", Frequency.HOUR_1),
            ],
        )
        entries = list(store.iter_inventory(symbol_substring="btc"))
        assert len(entries) == 2
        assert all("BTC" in e.symbol for e in entries)

    def test_corrupt_parquet_surfaces_as_unhealthy(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        # One good file
        self._populate(store, sample_df, [("yfinance", "AAPL", Frequency.DAILY)])
        # One corrupt file in a valid layout position
        corrupt_path = tmp_path / "yfinance" / "BAD" / "1d.parquet"
        corrupt_path.parent.mkdir(parents=True)
        corrupt_path.write_bytes(b"this is not a parquet file")

        entries = list(store.iter_inventory())
        assert len(entries) == 2
        healthy = [e for e in entries if e.is_healthy]
        unhealthy = [e for e in entries if not e.is_healthy]
        assert len(healthy) == 1 and healthy[0].symbol == "AAPL"
        assert len(unhealthy) == 1
        assert unhealthy[0].symbol == "BAD"
        assert unhealthy[0].error is not None
        assert "unreadable" in unhealthy[0].error

    def test_stray_files_at_root_are_skipped(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        store = ParquetStore(root=tmp_path)
        self._populate(store, sample_df, [("yfinance", "AAPL", Frequency.DAILY)])
        # Drop a README at the root and a parquet at the wrong depth.
        (tmp_path / "README.md").write_text("hi")
        (tmp_path / "stray.parquet").write_bytes(b"x")
        (tmp_path / "yfinance" / "loose.parquet").write_bytes(b"x")

        entries = list(store.iter_inventory())
        assert len(entries) == 1
        assert entries[0].symbol == "AAPL"

    def test_missing_root_yields_nothing(self, tmp_path: Path) -> None:
        # Construct then delete root: walker must not raise.
        store = ParquetStore(root=tmp_path / "nonexistent")
        store.root.rmdir()
        assert list(store.iter_inventory()) == []
