"""Offline unit tests for AkshareProvider.

These tests never hit the network. The provider's ``__init__`` does a
real ``import akshare``, which we satisfy by either:

1. Letting the real akshare import succeed (it's a project dependency),
2. Or, when akshare is missing, registering a stub module in
   ``sys.modules`` so the import resolves.

Once instantiated, all dispatch goes through a small in-memory
``_FakeAk`` object that records the call arguments and returns canned
DataFrames with AkShare's Chinese column names.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pytest

# Provide a stub `akshare` module if the real package is not importable,
# so we can still construct the provider in CI without network deps.
if importlib.util.find_spec("akshare") is None:  # pragma: no cover - env-dependent
    sys.modules["akshare"] = types.ModuleType("akshare")

from quant_lucky.data.base import DataProviderError, DownloadRequest, RateLimitError
from quant_lucky.data.providers.akshare_provider import AkshareProvider
from quant_lucky.data.schema import Frequency, Market, validate_ohlcv


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeAk:
    """In-memory replacement for the ``akshare`` module."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Default response for any endpoint: a tiny daily-style frame.
        self._response: pd.DataFrame | Exception = _make_cn_daily_df()

    def set_response(self, df_or_exc: pd.DataFrame | Exception) -> None:
        self._response = df_or_exc

    def _record(self, name: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append((name, kwargs))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response.copy()

    # The exact set of endpoints the provider dispatches to.
    def stock_zh_a_hist(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_zh_a_hist", **kw)

    def stock_zh_a_hist_min_em(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_zh_a_hist_min_em", **kw)

    def stock_hk_hist(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_hk_hist", **kw)

    def stock_hk_hist_min_em(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_hk_hist_min_em", **kw)

    def stock_us_hist(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_us_hist", **kw)

    def stock_us_hist_min_em(self, **kw: Any) -> pd.DataFrame:
        return self._record("stock_us_hist_min_em", **kw)


def _make_cn_daily_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "开盘": [10.0, 10.5, 10.4],
            "收盘": [10.5, 10.4, 10.6],
            "最高": [10.7, 10.6, 10.8],
            "最低": [9.9, 10.2, 10.3],
            "成交量": [1_000_000.0, 1_100_000.0, 1_050_000.0],
            "成交额": [1.05e7, 1.14e7, 1.10e7],
            "振幅": [7.5, 3.8, 4.8],
            "涨跌幅": [5.0, -1.0, 1.9],
            "涨跌额": [0.5, -0.1, 0.2],
            "换手率": [0.5, 0.55, 0.52],
        }
    )


def _make_cn_intraday_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "时间": [
                "2024-01-02 09:30:00",
                "2024-01-02 09:35:00",
                "2024-01-02 09:40:00",
            ],
            "开盘": [10.0, 10.1, 10.2],
            "收盘": [10.1, 10.2, 10.15],
            "最高": [10.15, 10.25, 10.22],
            "最低": [9.95, 10.05, 10.1],
            "成交量": [1000.0, 1200.0, 900.0],
            "成交额": [10000.0, 12200.0, 9150.0],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_ak() -> _FakeAk:
    return _FakeAk()


@pytest.fixture
def provider(fake_ak: _FakeAk) -> AkshareProvider:
    p = AkshareProvider(adjust="qfq")
    # Replace the bound module with our fake; bypasses real akshare entirely.
    p._set_test_module(fake_ak)
    return p


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    return (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_adjust_is_qfq(self, provider: AkshareProvider) -> None:
        assert provider.adjust == "qfq"

    def test_rejects_unknown_adjust(self) -> None:
        with pytest.raises(ValueError, match="adjust"):
            AkshareProvider(adjust="zzz")  # type: ignore[arg-type]

    def test_supports_advertises_correct_capabilities(self) -> None:
        p = AkshareProvider()
        p._set_test_module(_FakeAk())
        req = DownloadRequest(
            symbol="600000",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            frequency=Frequency.DAILY,
            market=Market.A_SHARE,
        )
        assert p.supports(req)

        req_crypto = DownloadRequest(
            symbol="BTC/USDT",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            frequency=Frequency.DAILY,
            market=Market.CRYPTO,
        )
        assert not p.supports(req_crypto)


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------
class TestSymbolNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("600000.SH", "600000"),
            ("000001.SZ", "000001"),
            ("600000", "600000"),
        ],
    )
    def test_a_share_strips_suffix(self, raw: str, expected: str) -> None:
        assert AkshareProvider._normalise_symbol(raw, Market.A_SHARE) == expected

    @pytest.mark.parametrize("bad", ["AAA", "60000", "600000.XX", ""])
    def test_a_share_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            AkshareProvider._normalise_symbol(bad, Market.A_SHARE)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("700", "00700"),
            ("00700", "00700"),
            ("700.HK", "00700"),
        ],
    )
    def test_hk_zero_pads(self, raw: str, expected: str) -> None:
        assert AkshareProvider._normalise_symbol(raw, Market.HK) == expected

    def test_us_with_prefix_passes_through(self) -> None:
        assert AkshareProvider._normalise_symbol("105.AAPL", Market.US) == "105.AAPL"

    def test_us_without_prefix_defaults_to_nyse(self) -> None:
        # Default is 106 (NYSE); caller can override by passing prefix.
        assert AkshareProvider._normalise_symbol("AAPL", Market.US) == "106.AAPL"


# ---------------------------------------------------------------------------
# Market inference (when request.market is None)
# ---------------------------------------------------------------------------
class TestMarketInference:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("600000.SH", Market.A_SHARE),
            ("000001.SZ", Market.A_SHARE),
            ("600000", Market.A_SHARE),
            ("00700", Market.HK),
            ("700.HK", Market.HK),
            ("AAPL", Market.US),
            ("105.AAPL", Market.US),
        ],
    )
    def test_inference(self, symbol: str, expected: Market) -> None:
        assert AkshareProvider._infer_market(symbol) == expected


# ---------------------------------------------------------------------------
# Endpoint dispatch
# ---------------------------------------------------------------------------
class TestDispatch:
    def _req(
        self,
        symbol: str,
        market: Market,
        freq: Frequency,
        date_range: tuple[datetime, datetime],
    ) -> DownloadRequest:
        return DownloadRequest(
            symbol=symbol,
            start=date_range[0],
            end=date_range[1],
            frequency=freq,
            market=market,
        )

    def test_a_share_daily_calls_stock_zh_a_hist(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider.fetch(self._req("600000.SH", Market.A_SHARE, Frequency.DAILY, date_range))
        names = [name for name, _ in fake_ak.calls]
        assert names == ["stock_zh_a_hist"]
        kwargs = fake_ak.calls[0][1]
        assert kwargs["symbol"] == "600000"
        assert kwargs["period"] == "daily"
        assert kwargs["adjust"] == "qfq"

    def test_a_share_intraday_calls_min_em(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(_make_cn_intraday_df())
        provider.fetch(
            self._req("600000.SH", Market.A_SHARE, Frequency.MINUTE_5, date_range)
        )
        assert fake_ak.calls[0][0] == "stock_zh_a_hist_min_em"
        assert fake_ak.calls[0][1]["period"] == "5"

    def test_hk_daily_calls_stock_hk_hist(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        provider.fetch(self._req("00700", Market.HK, Frequency.DAILY, date_range))
        assert fake_ak.calls[0][0] == "stock_hk_hist"
        assert fake_ak.calls[0][1]["symbol"] == "00700"

    def test_hk_intraday_omits_adjust(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(_make_cn_intraday_df())
        provider.fetch(self._req("00700", Market.HK, Frequency.MINUTE_30, date_range))
        assert fake_ak.calls[0][0] == "stock_hk_hist_min_em"
        # HK min_em endpoint signature does not accept `adjust`.
        assert "adjust" not in fake_ak.calls[0][1]

    def test_us_daily_calls_stock_us_hist(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        # AkShare US daily returns dates only; reuse the CN daily fixture
        # which the provider's normalizer is happy with after the rename.
        provider.fetch(self._req("105.AAPL", Market.US, Frequency.DAILY, date_range))
        assert fake_ak.calls[0][0] == "stock_us_hist"
        assert fake_ak.calls[0][1]["symbol"] == "105.AAPL"

    @pytest.mark.parametrize(
        ("freq", "expected_period"),
        [
            (Frequency.MINUTE_1, "1"),
            (Frequency.MINUTE_5, "5"),
            (Frequency.MINUTE_15, "15"),
            (Frequency.MINUTE_30, "30"),
            (Frequency.HOUR_1, "60"),
            (Frequency.DAILY, "daily"),
            (Frequency.WEEKLY, "weekly"),
            (Frequency.MONTHLY, "monthly"),
        ],
    )
    def test_period_string_mapping(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
        freq: Frequency,
        expected_period: str,
    ) -> None:
        if freq in {
            Frequency.MINUTE_1,
            Frequency.MINUTE_5,
            Frequency.MINUTE_15,
            Frequency.MINUTE_30,
            Frequency.HOUR_1,
        }:
            fake_ak.set_response(_make_cn_intraday_df())
        provider.fetch(self._req("600000", Market.A_SHARE, freq, date_range))
        # US intraday path is the only one that does not pass `period`; skip it.
        assert fake_ak.calls[-1][1].get("period") == expected_period


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------
class TestSchemaNormalization:
    def test_daily_response_passes_schema_validator(
        self,
        provider: AkshareProvider,
        date_range: tuple[datetime, datetime],
    ) -> None:
        df = provider.fetch(
            DownloadRequest(
                symbol="600000",
                start=date_range[0],
                end=date_range[1],
                frequency=Frequency.DAILY,
                market=Market.A_SHARE,
            )
        )
        # Column order canonical-first, extras at the end.
        assert list(df.columns[:6]) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert "amount" in df.columns
        assert "turnover" in df.columns
        # Timestamps are UTC.
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) in {"UTC", "Etc/UTC"}
        validate_ohlcv(df)

    def test_intraday_response_passes_schema_validator(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(_make_cn_intraday_df())
        df = provider.fetch(
            DownloadRequest(
                symbol="600000",
                start=date_range[0],
                end=date_range[1],
                frequency=Frequency.MINUTE_5,
                market=Market.A_SHARE,
            )
        )
        assert "timestamp" in df.columns
        # 09:30 Beijing == 01:30 UTC.
        assert df["timestamp"].iloc[0].hour == 1
        validate_ohlcv(df)

    def test_us_daily_treated_as_calendar_date_utc(
        self,
        provider: AkshareProvider,
        date_range: tuple[datetime, datetime],
    ) -> None:
        # The fixture uses 2024-01-02 etc. US daily should map to UTC midnight.
        df = provider.fetch(
            DownloadRequest(
                symbol="105.AAPL",
                start=date_range[0],
                end=date_range[1],
                frequency=Frequency.DAILY,
                market=Market.US,
            )
        )
        first = df["timestamp"].iloc[0]
        assert first.hour == 0
        assert first.minute == 0


# ---------------------------------------------------------------------------
# Adjustment parameter
# ---------------------------------------------------------------------------
class TestAdjustment:
    def test_hfq_propagates_to_endpoint(
        self,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        p = AkshareProvider(adjust="hfq")
        p._set_test_module(fake_ak)
        p.fetch(
            DownloadRequest(
                symbol="600000",
                start=date_range[0],
                end=date_range[1],
                frequency=Frequency.DAILY,
                market=Market.A_SHARE,
            )
        )
        assert fake_ak.calls[0][1]["adjust"] == "hfq"

    def test_raw_adjust_is_empty_string(
        self,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        p = AkshareProvider(adjust="")
        p._set_test_module(fake_ak)
        p.fetch(
            DownloadRequest(
                symbol="600000",
                start=date_range[0],
                end=date_range[1],
                frequency=Frequency.DAILY,
                market=Market.A_SHARE,
            )
        )
        assert fake_ak.calls[0][1]["adjust"] == ""


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
class TestErrors:
    def test_empty_response_raises(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(pd.DataFrame())
        with pytest.raises(DataProviderError, match="empty"):
            provider.fetch(
                DownloadRequest(
                    symbol="600000",
                    start=date_range[0],
                    end=date_range[1],
                    frequency=Frequency.DAILY,
                    market=Market.A_SHARE,
                )
            )

    def test_underlying_exception_wraps_to_data_provider_error(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(RuntimeError("boom"))
        with pytest.raises(DataProviderError, match="fetch failed"):
            provider.fetch(
                DownloadRequest(
                    symbol="600000",
                    start=date_range[0],
                    end=date_range[1],
                    frequency=Frequency.DAILY,
                    market=Market.A_SHARE,
                )
            )

    def test_rate_limit_text_maps_to_rate_limit_error(
        self,
        provider: AkshareProvider,
        fake_ak: _FakeAk,
        date_range: tuple[datetime, datetime],
    ) -> None:
        fake_ak.set_response(RuntimeError("HTTP 429 Too Many Requests"))
        with pytest.raises(RateLimitError):
            provider.fetch(
                DownloadRequest(
                    symbol="600000",
                    start=date_range[0],
                    end=date_range[1],
                    frequency=Frequency.DAILY,
                    market=Market.A_SHARE,
                )
            )


# ---------------------------------------------------------------------------
# Live integration (skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.live
def test_live_a_share_daily_smoke() -> None:
    """Hits eastmoney.com. Run only when explicitly requested:

        QUANT_RUN_LIVE=1 pytest -m live tests/data/providers/test_akshare_provider.py
    """
    if not os.environ.get("QUANT_RUN_LIVE"):
        pytest.skip("Set QUANT_RUN_LIVE=1 to run live network tests")
    p = AkshareProvider(adjust="qfq")
    df = p.fetch(
        DownloadRequest(
            symbol="600000",
            start=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            frequency=Frequency.DAILY,
            market=Market.A_SHARE,
        )
    )
    assert len(df) > 0
    validate_ohlcv(df)
