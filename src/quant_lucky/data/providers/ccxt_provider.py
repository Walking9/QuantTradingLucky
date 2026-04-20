"""Crypto exchange provider using the ccxt library.

Public OHLCV endpoints don't require credentials. Default exchange is
Binance; pass ``exchange_id`` to use a different one (okx, bybit, ...).

Pagination handles the typical 1000-candle-per-request limit by stepping
``since`` forward until we cover the requested range.
"""

from __future__ import annotations

from typing import ClassVar

import ccxt
import pandas as pd

from quant_lucky.data.base import DataProvider, DataProviderError, DownloadRequest
from quant_lucky.data.schema import Frequency, Market
from quant_lucky.utils.logging import logger

_FREQUENCY_MAP: dict[Frequency, str] = {
    Frequency.MINUTE_1: "1m",
    Frequency.MINUTE_5: "5m",
    Frequency.MINUTE_15: "15m",
    Frequency.MINUTE_30: "30m",
    Frequency.HOUR_1: "1h",
    Frequency.HOUR_4: "4h",
    Frequency.DAILY: "1d",
    Frequency.WEEKLY: "1w",
    Frequency.MONTHLY: "1M",
}


class CCXTProvider(DataProvider):
    """OHLCV from any ccxt-supported exchange."""

    # Class-level defaults; instances may override ``name``.
    name: ClassVar[str] = "ccxt"
    supported_markets: ClassVar[set[Market]] = {Market.CRYPTO}
    supported_frequencies: ClassVar[set[Frequency]] = set(_FREQUENCY_MAP.keys())
    requires_credentials: ClassVar[bool] = False

    def __init__(self, exchange_id: str = "binance", *, page_limit: int = 1000) -> None:
        if not hasattr(ccxt, exchange_id):
            raise DataProviderError(f"Unknown ccxt exchange: {exchange_id}")
        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls({"enableRateLimit": True})
        self.page_limit = page_limit
        # Per-instance name so the store partitions by exchange:
        #   data/raw/ccxt.binance/BTC-USDT/1h.parquet
        self.name = f"ccxt.{exchange_id}"

    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        timeframe = _FREQUENCY_MAP.get(request.frequency)
        if timeframe is None:
            raise DataProviderError(f"Unsupported frequency: {request.frequency}")

        start_ms = int(request.start.timestamp() * 1000)
        end_ms = int(request.end.timestamp() * 1000)

        logger.info(
            "{name}: {symbol} {start} -> {end} @ {freq}",
            name=self.name,
            symbol=request.symbol,
            start=request.start.date(),
            end=request.end.date(),
            freq=request.frequency.value,
        )

        all_candles: list[list[float]] = []
        since = start_ms

        while since < end_ms:
            try:
                page = self.exchange.fetch_ohlcv(
                    request.symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=self.page_limit,
                )
            except Exception as e:
                raise DataProviderError(f"ccxt fetch failed: {e}") from e

            if not page:
                break

            all_candles.extend(page)

            last_ts = page[-1][0]
            if last_ts <= since:
                # Exchange returned a page that doesn't advance the cursor
                break
            since = last_ts + 1

            if len(page) < self.page_limit:
                break  # reached the tail

        if not all_candles:
            raise DataProviderError(f"ccxt returned no data for {request.symbol}")

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        # Coerce request.end to UTC-aware so the comparison works regardless of
        # whether the caller passed a tz-naive or tz-aware datetime.
        end_utc = pd.Timestamp(request.end)
        if end_utc.tz is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        df = df[df["timestamp"] <= end_utc]
        df = (
            df[["timestamp", "open", "high", "low", "close", "volume"]]
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df
