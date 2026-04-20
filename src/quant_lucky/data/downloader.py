"""High-level download orchestrator.

Combines a :class:`DataProvider` with a :class:`ParquetStore` and adds:

- Local cache awareness (skip re-download unless ``force=True``).
- Batch API for downloading multiple symbols with per-symbol error capture.

Typical usage::

    from datetime import datetime, timezone
    from quant_lucky.data import Downloader, YFinanceProvider, Frequency

    dl = Downloader(YFinanceProvider())
    df = dl.download(
        "AAPL",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        frequency=Frequency.DAILY,
    )
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_lucky.data.base import DataProvider, DataProviderError, DownloadRequest
from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import ParquetStore
from quant_lucky.utils.logging import logger


class Downloader:
    """Glue between a provider and a local Parquet cache."""

    def __init__(
        self,
        provider: DataProvider,
        store: ParquetStore | None = None,
    ) -> None:
        self.provider = provider
        self.store = store if store is not None else ParquetStore()

    def download(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAILY,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Download or load from cache and return a canonical OHLCV DataFrame."""
        if not force and self.store.exists(self.provider.name, symbol, frequency):
            logger.info(
                "Cache hit: {p}/{s}/{f}",
                p=self.provider.name,
                s=symbol,
                f=frequency.value,
            )
            return self.store.read(self.provider.name, symbol, frequency)

        request = DownloadRequest(
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
        )
        df = self.provider.fetch(request)
        self.store.write(
            df,
            provider=self.provider.name,
            symbol=symbol,
            frequency=frequency,
        )
        return df

    def download_many(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAILY,
        *,
        force: bool = False,
        stop_on_error: bool = False,
    ) -> dict[str, pd.DataFrame | Exception]:
        """Batch download. Errors are captured per-symbol, not re-raised.

        Set ``stop_on_error=True`` to abort the batch on first failure
        (useful for debugging new providers).
        """
        results: dict[str, pd.DataFrame | Exception] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.download(
                    symbol, start, end, frequency, force=force
                )
            except DataProviderError as e:
                logger.error("Failed {symbol}: {err}", symbol=symbol, err=e)
                results[symbol] = e
                if stop_on_error:
                    break
        return results
