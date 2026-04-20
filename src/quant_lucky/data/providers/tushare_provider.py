"""Tushare Pro provider for A-share daily bars.

Requires ``TUSHARE_TOKEN`` in ``.env`` (register at https://tushare.pro).
The free tier has per-minute quota; intraday data requires a paid tier,
so this first cut only implements daily / weekly / monthly frequencies.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from quant_lucky.data.base import (
    AuthenticationError,
    DataProvider,
    DataProviderError,
    DownloadRequest,
)
from quant_lucky.data.schema import Frequency, Market
from quant_lucky.utils.config import settings
from quant_lucky.utils.logging import logger


class TushareProvider(DataProvider):
    """A-share daily OHLCV via Tushare Pro."""

    name: ClassVar[str] = "tushare"
    supported_markets: ClassVar[set[Market]] = {Market.A_SHARE}
    supported_frequencies: ClassVar[set[Frequency]] = {
        Frequency.DAILY,
        Frequency.WEEKLY,
        Frequency.MONTHLY,
    }
    requires_credentials: ClassVar[bool] = True

    def __init__(self) -> None:
        if settings.tushare_token is None:
            raise AuthenticationError(
                "TUSHARE_TOKEN not set. Register at https://tushare.pro, "
                "then copy your token into .env"
            )

        import tushare as ts  # heavy; import lazily

        ts.set_token(settings.tushare_token.get_secret_value())
        self._pro = ts.pro_api()

    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        if request.frequency != Frequency.DAILY:
            # weekly/monthly available via different endpoints; future work
            raise DataProviderError(
                f"Only daily frequency implemented in M1 (got {request.frequency})"
            )

        ts_code = self._normalise_symbol(request.symbol)
        start = request.start.strftime("%Y%m%d")
        end = request.end.strftime("%Y%m%d")

        logger.info(
            "tushare: {code} {start} -> {end}",
            code=ts_code,
            start=start,
            end=end,
        )

        try:
            df = self._pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        except Exception as e:
            raise DataProviderError(f"tushare fetch failed: {e}") from e

        if df is None or df.empty:
            raise DataProviderError(f"tushare returned empty for {ts_code}")

        df = df.rename(columns={"trade_date": "timestamp", "vol": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """Accept both ``600000`` and ``600000.SH`` forms."""
        if "." in symbol:
            return symbol.upper()
        # Exchange inference by first digits (good enough for common stocks)
        if symbol.startswith(("60", "68", "90")):
            return f"{symbol}.SH"
        if symbol.startswith(("00", "30", "20")):
            return f"{symbol}.SZ"
        if symbol.startswith(("43", "83", "87", "88")):
            return f"{symbol}.BJ"
        raise ValueError(f"Cannot infer exchange for symbol: {symbol!r}")
