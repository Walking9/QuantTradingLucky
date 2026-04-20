"""Downloader tests using a fake provider (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from quant_lucky.data.base import DataProvider, DataProviderError, DownloadRequest
from quant_lucky.data.downloader import Downloader
from quant_lucky.data.schema import Frequency, Market
from quant_lucky.data.store import ParquetStore


class _FakeProvider(DataProvider):
    """In-memory fake that returns a pre-canned DataFrame and counts calls."""

    name: ClassVar[str] = "fake"
    supported_markets: ClassVar[set[Market]] = {Market.US}
    supported_frequencies: ClassVar[set[Frequency]] = {Frequency.DAILY}
    requires_credentials: ClassVar[bool] = False

    def __init__(self, df: pd.DataFrame, *, fail_on: set[str] | None = None) -> None:
        self._df = df
        self._fail_on = fail_on or set()
        self.fetch_count = 0
        self.last_request: DownloadRequest | None = None

    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        self.fetch_count += 1
        self.last_request = request
        if request.symbol in self._fail_on:
            raise DataProviderError(f"fake failure for {request.symbol}")
        return self._df.copy()


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


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    return (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
    )


class TestDownloaderSingle:
    def test_fetches_and_stores(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df)
        store = ParquetStore(root=tmp_path)
        dl = Downloader(provider=provider, store=store)

        start, end = date_range
        df = dl.download("ABC", start, end, Frequency.DAILY)

        pd.testing.assert_frame_equal(df, sample_df)
        assert provider.fetch_count == 1
        assert store.exists("fake", "ABC", Frequency.DAILY)

    def test_request_passed_correctly(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df)
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range
        dl.download("ABC", start, end, Frequency.DAILY)

        assert provider.last_request is not None
        assert provider.last_request.symbol == "ABC"
        assert provider.last_request.start == start
        assert provider.last_request.end == end
        assert provider.last_request.frequency == Frequency.DAILY

    def test_cache_hit_skips_fetch(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df)
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range

        dl.download("ABC", start, end, Frequency.DAILY)
        dl.download("ABC", start, end, Frequency.DAILY)  # second call = cache hit

        assert provider.fetch_count == 1

    def test_force_bypasses_cache(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df)
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range

        dl.download("ABC", start, end, Frequency.DAILY)
        dl.download("ABC", start, end, Frequency.DAILY, force=True)

        assert provider.fetch_count == 2


class TestDownloaderBatch:
    def test_multi_symbol_all_success(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df)
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range

        results = dl.download_many(["AAA", "BBB", "CCC"], start, end, Frequency.DAILY)
        assert set(results.keys()) == {"AAA", "BBB", "CCC"}
        assert all(isinstance(v, pd.DataFrame) for v in results.values())

    def test_multi_symbol_partial_failure(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df, fail_on={"BBB"})
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range

        results = dl.download_many(["AAA", "BBB", "CCC"], start, end, Frequency.DAILY)

        assert isinstance(results["AAA"], pd.DataFrame)
        assert isinstance(results["BBB"], DataProviderError)
        assert isinstance(results["CCC"], pd.DataFrame)

    def test_stop_on_error(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider = _FakeProvider(sample_df, fail_on={"AAA"})
        dl = Downloader(provider=provider, store=ParquetStore(root=tmp_path))
        start, end = date_range

        results = dl.download_many(
            ["AAA", "BBB", "CCC"], start, end, Frequency.DAILY, stop_on_error=True
        )
        # Only AAA attempted (and failed), BBB/CCC never reached
        assert "AAA" in results
        assert "BBB" not in results
        assert "CCC" not in results
