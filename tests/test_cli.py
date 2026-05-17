"""Tests for the ``quant`` CLI commands.

Uses click's ``CliRunner`` so no real subprocess is spawned and stdout
is captured deterministically. The ``data ls`` command reads from
``settings.raw_dir``; tests redirect that via monkeypatch so each test
operates on its own ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from quant_lucky.cli import main
from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import ParquetStore


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_500.0, 1_200.0],
        }
    )


@pytest.fixture
def cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the CLI's data cache lookup to a temp dir.

    The command reads ``settings.raw_dir`` lazily inside the command
    body. ``settings`` is a pydantic model — its `raw_dir` is a
    ``@property``, so we patch the underlying ``data_root`` instead.
    """
    monkeypatch.setattr("quant_lucky.cli.settings.data_root", tmp_path)
    cache = tmp_path / "raw"
    cache.mkdir()
    return cache


def _populate(
    cache_root: Path, sample_df: pd.DataFrame, items: list[tuple[str, str, Frequency]]
) -> None:
    store = ParquetStore(root=cache_root)
    for provider, symbol, freq in items:
        store.write(sample_df, provider=provider, symbol=symbol, frequency=freq)


class TestDataLs:
    def test_empty_cache_prints_hint(self, cache_root: Path) -> None:
        result = CliRunner().invoke(main, ["data", "ls"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "empty" in result.output
        assert "quant download" in result.output

    def test_tree_contains_provider_and_frequency(
        self, cache_root: Path, sample_df: pd.DataFrame
    ) -> None:
        _populate(cache_root, sample_df, [("yfinance", "AAPL", Frequency.DAILY)])
        result = CliRunner().invoke(main, ["data", "ls"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "yfinance" in result.output
        assert "AAPL" in result.output
        assert "1d" in result.output
        assert "3 rows" in result.output

    def test_provider_filter(self, cache_root: Path, sample_df: pd.DataFrame) -> None:
        _populate(
            cache_root,
            sample_df,
            [
                ("yfinance", "AAPL", Frequency.DAILY),
                ("ccxt.binance", "BTC-USDT", Frequency.DAILY),
            ],
        )
        result = CliRunner().invoke(
            main, ["data", "ls", "--provider", "yfinance"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "BTC-USDT" not in result.output

    def test_symbol_substring_filter_case_insensitive(
        self, cache_root: Path, sample_df: pd.DataFrame
    ) -> None:
        _populate(
            cache_root,
            sample_df,
            [
                ("yfinance", "AAPL", Frequency.DAILY),
                ("ccxt.binance", "BTC-USDT", Frequency.DAILY),
            ],
        )
        result = CliRunner().invoke(main, ["data", "ls", "--symbol", "btc"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "BTC-USDT" in result.output
        assert "AAPL" not in result.output

    def test_json_output_is_valid(self, cache_root: Path, sample_df: pd.DataFrame) -> None:
        _populate(
            cache_root,
            sample_df,
            [
                ("yfinance", "AAPL", Frequency.DAILY),
                ("ccxt.binance", "BTC-USDT", Frequency.DAILY),
            ],
        )
        result = CliRunner().invoke(main, ["data", "ls", "--json"], catch_exceptions=False)
        assert result.exit_code == 0

        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        first = parsed[0]
        for key in (
            "provider",
            "symbol",
            "frequency",
            "path",
            "size_bytes",
            "row_count",
            "start",
            "end",
            "downloaded_at",
            "schema_version",
            "is_healthy",
        ):
            assert key in first
        assert all(e["is_healthy"] for e in parsed)
        assert all(e["row_count"] == 3 for e in parsed)

    def test_unhealthy_entry_shown_with_error(
        self, cache_root: Path, sample_df: pd.DataFrame
    ) -> None:
        _populate(cache_root, sample_df, [("yfinance", "AAPL", Frequency.DAILY)])
        bad = cache_root / "yfinance" / "BAD" / "1d.parquet"
        bad.parent.mkdir()
        bad.write_bytes(b"not parquet")

        result = CliRunner().invoke(main, ["data", "ls"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "BAD" in result.output
        # Either the ✗ marker or the "unreadable" word should appear.
        assert "unreadable" in result.output or "✗" in result.output


class TestDataLsHelp:
    def test_help_lists_command(self) -> None:
        result = CliRunner().invoke(main, ["data", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "ls" in result.output
