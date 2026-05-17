"""AkShare provider — credential-free A-share / HK / US OHLCV.

AkShare is an open-source Python wrapper that scrapes public financial
data sources (Eastmoney, Sina, Tencent ...). It needs no token, which
makes it a natural cross-check for :class:`TushareProvider` (paid, A-share
only) and a fallback for :class:`YFinanceProvider` (US/HK, often rate
limited).

This provider supports:

* Markets:  ``A_SHARE``, ``HK``, ``US``
* Frequencies:  ``1m / 5m / 15m / 30m / 60m / daily / weekly / monthly``
* Adjustment:  ``qfq`` (前复权, default), ``hfq`` (后复权), ``""`` (raw)

Network proxies are picked up automatically from ``HTTP_PROXY`` /
``HTTPS_PROXY`` because AkShare uses ``requests`` under the hood.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pandas as pd

from quant_lucky.data.base import (
    DataProvider,
    DataProviderError,
    DownloadRequest,
    RateLimitError,
)
from quant_lucky.data.schema import Frequency, Market
from quant_lucky.utils.logging import logger
from quant_lucky.utils.net import bypass_proxy_env

Adjust = Literal["qfq", "hfq", ""]

_DAILY_PERIOD_MAP: dict[Frequency, str] = {
    Frequency.DAILY: "daily",
    Frequency.WEEKLY: "weekly",
    Frequency.MONTHLY: "monthly",
}

_INTRADAY_PERIOD_MAP: dict[Frequency, str] = {
    Frequency.MINUTE_1: "1",
    Frequency.MINUTE_5: "5",
    Frequency.MINUTE_15: "15",
    Frequency.MINUTE_30: "30",
    Frequency.HOUR_1: "60",
}

_COLUMN_RENAME: dict[str, str] = {
    "日期": "timestamp",
    "时间": "timestamp",
    "date": "timestamp",  # sina endpoint
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

_CN_TZ = "Asia/Shanghai"
_US_TZ = "America/New_York"


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient upstream issue worth retrying.

    AkShare wraps every error in plain ``Exception`` and only string-stringifies
    the cause, so we match on substrings of the message rather than exception
    types. Covers: connection reset, 502/503, empty reply, and DNS hiccups.
    """
    text = str(exc).lower()
    needles = (
        "remotedisconnected",
        "remote end closed",
        "connection aborted",
        "connection reset",
        "empty reply",
        "502",
        "503",
        "504",
        "max retries exceeded",
        "read timed out",
        "getaddrinfo failed",
    )
    return any(n in text for n in needles)


class AkshareProvider(DataProvider):
    """OHLCV via the ``akshare`` package."""

    name: ClassVar[str] = "akshare"
    supported_markets: ClassVar[set[Market]] = {Market.A_SHARE, Market.HK, Market.US}
    supported_frequencies: ClassVar[set[Frequency]] = (
        set(_DAILY_PERIOD_MAP) | set(_INTRADAY_PERIOD_MAP)
    )
    requires_credentials: ClassVar[bool] = False

    def __init__(self, adjust: Adjust = "qfq") -> None:
        if adjust not in ("qfq", "hfq", ""):
            raise ValueError(f"adjust must be 'qfq', 'hfq', or '' — got {adjust!r}")
        self._adjust = adjust

        import akshare as ak  # heavy import; defer until instantiation

        self._ak = ak

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        market = request.market or self._infer_market(request.symbol)
        if market not in self.supported_markets:
            raise DataProviderError(f"akshare does not support market: {market}")

        symbol = self._normalise_symbol(request.symbol, market)
        start = request.start.strftime("%Y%m%d")
        end = request.end.strftime("%Y%m%d")
        is_intraday = request.frequency in _INTRADAY_PERIOD_MAP

        logger.info(
            "akshare: {symbol} {start} -> {end} @ {freq} (market={mkt}, adjust={adj!r})",
            symbol=symbol,
            start=request.start.date(),
            end=request.end.date(),
            freq=request.frequency.value,
            mkt=market.value,
            adj=self._adjust,
        )

        try:
            with bypass_proxy_env():
                df = self._dispatch(market, request.frequency, symbol, start, end)
        except (DataProviderError, ValueError):
            raise
        except Exception as e:  # AkShare wraps everything in plain Exception
            text = str(e)
            if "429" in text or "Too Many Requests" in text or "rate" in text.lower():
                raise RateLimitError(f"akshare rate limited: {e}") from e
            # A-share daily: eastmoney is often flaky from mainland; try sina.
            if (
                market is Market.A_SHARE
                and request.frequency is Frequency.DAILY
                and _is_transient_network_error(e)
            ):
                logger.warning(
                    "akshare eastmoney failed ({err}); falling back to sina endpoint",
                    err=type(e).__name__,
                )
                try:
                    with bypass_proxy_env():
                        df = self._fetch_a_share_daily_sina(symbol, start, end)
                except Exception as fallback_err:  # pragma: no cover - belt&braces
                    raise DataProviderError(
                        f"akshare fetch failed (eastmoney: {e}; sina: {fallback_err})"
                    ) from fallback_err
            else:
                raise DataProviderError(f"akshare fetch failed: {e}") from e

        if df is None or df.empty:
            raise DataProviderError(f"akshare returned empty for {symbol}")

        return self._normalise(df, market=market, is_intraday=is_intraday)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _dispatch(
        self,
        market: Market,
        frequency: Frequency,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Route to the correct ``akshare`` endpoint."""
        is_intraday = frequency in _INTRADAY_PERIOD_MAP
        period = (
            _INTRADAY_PERIOD_MAP[frequency]
            if is_intraday
            else _DAILY_PERIOD_MAP[frequency]
        )

        if market is Market.A_SHARE:
            if is_intraday:
                return self._ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period,
                    start_date=f"{start[:4]}-{start[4:6]}-{start[6:8]} 09:00:00",
                    end_date=f"{end[:4]}-{end[4:6]}-{end[6:8]} 15:00:00",
                    adjust=self._adjust,
                )
            return self._ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start,
                end_date=end,
                adjust=self._adjust,
            )

        if market is Market.HK:
            if is_intraday:
                # AkShare's HK intraday endpoint does not accept `adjust`.
                return self._ak.stock_hk_hist_min_em(
                    symbol=symbol,
                    period=period,
                    start_date=f"{start[:4]}-{start[4:6]}-{start[6:8]} 09:00:00",
                    end_date=f"{end[:4]}-{end[4:6]}-{end[6:8]} 16:00:00",
                )
            return self._ak.stock_hk_hist(
                symbol=symbol,
                period=period,
                start_date=start,
                end_date=end,
                adjust=self._adjust,
            )

        if market is Market.US:
            if is_intraday:
                # stock_us_hist_min_em only takes symbol; window is ~last 5 days.
                return self._ak.stock_us_hist_min_em(symbol=symbol)
            return self._ak.stock_us_hist(
                symbol=symbol,
                period=period,
                start_date=start,
                end_date=end,
                adjust=self._adjust,
            )

        raise DataProviderError(f"akshare does not support market: {market}")

    # ------------------------------------------------------------------
    # Sina fallback for A-share daily
    # ------------------------------------------------------------------
    def _fetch_a_share_daily_sina(
        self, ak_symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fall back to the sina endpoint when eastmoney is unavailable.

        ``stock_zh_a_daily`` takes a sina-style symbol like ``sh600519`` or
        ``sz000001``. We infer the prefix from the bare 6-digit code:
        ``6`` / ``9`` → Shanghai, otherwise Shenzhen. (This matches the
        rule used by Tushare's exchange suffix mapping.)
        """
        if not (ak_symbol.isdigit() and len(ak_symbol) == 6):
            raise DataProviderError(
                f"sina fallback requires bare 6-digit A-share code, got {ak_symbol!r}"
            )
        prefix = "sh" if ak_symbol[0] in ("6", "9") else "sz"
        sina_symbol = f"{prefix}{ak_symbol}"

        df = self._ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start,
            end_date=end,
            adjust=self._adjust,
        )
        return df

    # ------------------------------------------------------------------
    # Symbol normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_market(symbol: str) -> Market:
        """Best-effort market detection from symbol shape."""
        s = symbol.upper()
        if s.endswith((".SH", ".SZ", ".BJ")):
            return Market.A_SHARE
        if s.endswith(".HK"):
            return Market.HK
        bare = s.split(".")[-1]
        if bare.isdigit():
            if len(bare) == 6:
                return Market.A_SHARE
            if len(bare) <= 5:
                return Market.HK
        # Anything else (alphabetic ticker) we treat as US.
        return Market.US

    @staticmethod
    def _normalise_symbol(symbol: str, market: Market) -> str:
        """Map the project's symbol form to AkShare's expected form."""
        s = symbol.strip()
        if not s:
            raise ValueError("symbol must not be empty")

        if market is Market.A_SHARE:
            # Project & Tushare style: '600000.SH' / '000001.SZ'.
            # AkShare wants the bare 6-digit code.
            if "." in s:
                bare, suffix = s.split(".", 1)
                if suffix.upper() not in {"SH", "SZ", "BJ"}:
                    raise ValueError(f"Unknown A-share exchange suffix: {symbol!r}")
            else:
                bare = s
            if not (bare.isdigit() and len(bare) == 6):
                raise ValueError(f"Invalid A-share code: {symbol!r}")
            return bare

        if market is Market.HK:
            bare = s.split(".")[0]
            if not bare.isdigit() or len(bare) > 5:
                raise ValueError(f"Invalid HK code: {symbol!r}")
            return bare.zfill(5)

        if market is Market.US:
            # AkShare's stock_us_hist wants '<exchange_prefix>.<TICKER>',
            # where the prefix is 105 (NASDAQ), 106 (NYSE) or 107 (AMEX).
            if "." in s and s.split(".", 1)[0].isdigit():
                return s.upper()
            logger.warning(
                "akshare US symbol {sym!r} has no exchange prefix; "
                "defaulting to 106 (NYSE). Pass '105.{sym}' or '107.{sym}' "
                "to override.",
                sym=s.upper(),
            )
            return f"106.{s.upper()}"

        raise ValueError(f"Unsupported market: {market}")

    # ------------------------------------------------------------------
    # Schema normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(df: pd.DataFrame, *, market: Market, is_intraday: bool) -> pd.DataFrame:
        """Rename columns, coerce timestamps to UTC, sort, dedupe."""
        df = df.rename(columns=_COLUMN_RENAME)
        if "timestamp" not in df.columns:
            raise DataProviderError(
                f"akshare response missing date/time column; got {list(df.columns)!r}"
            )

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any():
            raise DataProviderError("akshare returned unparseable timestamps")

        # AkShare timestamps are naive local time of the underlying exchange.
        # Daily US bars are calendar dates with no tz semantics — treat them
        # the same way Tushare/yfinance do (localize to UTC midnight).
        tz = _US_TZ if (market is Market.US and is_intraday) else _CN_TZ
        if market is Market.US and not is_intraday:
            tz = "UTC"

        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        ts = ts.dt.tz_convert("UTC")
        df["timestamp"] = ts

        canonical = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in canonical if c not in df.columns]
        if missing:
            raise DataProviderError(
                f"akshare response missing canonical columns: {missing}"
            )

        # Cast numerics defensively; AkShare can return object dtype on edges.
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("amount", "turnover"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=canonical)
        extras = [c for c in df.columns if c not in canonical and c in {"amount", "turnover"}]
        df = df[canonical + extras]
        df = (
            df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # Test/debug helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AkshareProvider(adjust={self._adjust!r})"

    # Expose for tests that want to assert dispatch arguments.
    @property
    def adjust(self) -> Adjust:
        return self._adjust

    # Keep mypy happy when callers stash extra state.
    def _set_test_module(self, ak: Any) -> None:  # pragma: no cover - test helper
        """Override the bound akshare module (used by tests for isolation)."""
        self._ak = ak
