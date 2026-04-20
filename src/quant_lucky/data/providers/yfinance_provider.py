"""Yahoo Finance provider (free, credential-less).

Covers US equities and most global tickers including major ETFs and
``BTC-USD``/``ETH-USD`` ticker aliases. Good for learning; NOT suitable
for production — yfinance has known adjustment inconsistencies and is
unofficial scraping.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import yfinance as yf

from quant_lucky.data.base import DataProvider, DataProviderError, DownloadRequest
from quant_lucky.data.schema import Frequency, Market
from quant_lucky.utils.logging import logger

_FREQUENCY_MAP: dict[Frequency, str] = {
    Frequency.MINUTE_1: "1m",
    Frequency.MINUTE_5: "5m",
    Frequency.MINUTE_15: "15m",
    Frequency.MINUTE_30: "30m",
    Frequency.HOUR_1: "60m",
    Frequency.DAILY: "1d",
    Frequency.WEEKLY: "1wk",
    Frequency.MONTHLY: "1mo",
}


class YFinanceProvider(DataProvider):
    """Yahoo Finance OHLCV via the `yfinance` package."""

    name: ClassVar[str] = "yfinance"
    supported_markets: ClassVar[set[Market]] = {Market.US, Market.CRYPTO, Market.HK}
    supported_frequencies: ClassVar[set[Frequency]] = set(_FREQUENCY_MAP.keys())
    requires_credentials: ClassVar[bool] = False

    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        yf_interval = _FREQUENCY_MAP.get(request.frequency)
        if yf_interval is None:
            raise DataProviderError(f"Unsupported frequency: {request.frequency}")

        logger.info(
            "yfinance: {symbol} {start} -> {end} @ {freq}",
            symbol=request.symbol,
            start=request.start.date(),
            end=request.end.date(),
            freq=request.frequency.value,
        )

        try:
            ticker = yf.Ticker(request.symbol)
            df = ticker.history(
                start=request.start,
                end=request.end,
                interval=yf_interval,
                auto_adjust=False,
                actions=True,
            )
        except Exception as e:  # yfinance is noisy — normalise errors
            raise DataProviderError(f"yfinance fetch failed: {e}") from e

        if df.empty:
            raise DataProviderError(f"yfinance returned empty for {request.symbol}")

        return _normalise(df)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and fix timezone so the output matches the canonical schema."""
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(
        columns={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adjusted_close",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )

    # Coerce to UTC (yfinance returns tz-aware for intraday, naive for daily)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts

    # Enforce column order (canonical first, extras at the end)
    canonical = ["timestamp", "open", "high", "low", "close", "volume"]
    extras = [c for c in df.columns if c not in canonical]
    df = df[canonical + extras]
    df = df.dropna(subset=canonical).reset_index(drop=True)
    return df
