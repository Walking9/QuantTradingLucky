"""Tests for :class:`CryptoTopNUniverse`."""

from __future__ import annotations

from datetime import date

import pytest

from quant_lucky.universe import CryptoTopNUniverse
from quant_lucky.universe.base import UniverseDataUnavailableError


def _fake_tickers(_: date) -> dict[str, float]:
    # volumes in USDT
    return {
        "BTC/USDT": 5_000_000_000,
        "ETH/USDT": 2_000_000_000,
        "SOL/USDT": 800_000_000,
        "XRP/USDT": 600_000_000,
        "DOGE/USDT": 400_000_000,
        "ADA/USDT": 300_000_000,
        "BNB/USDT": 500_000_000,
        "FOO/BTC": 999_999_999,  # different quote currency - should be excluded
    }


class TestCryptoTopNUniverse:
    def test_top_n_by_volume(self) -> None:
        u = CryptoTopNUniverse(top_n=3, ticker_source=_fake_tickers)
        snap = u.snapshot(date(2024, 6, 1))
        # Highest-volume USDT pairs
        assert set(snap) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}

    def test_anchors_always_included(self) -> None:
        # top_n=1, anchors force BTC+ETH back in
        u = CryptoTopNUniverse(
            top_n=1, ticker_source=_fake_tickers, anchors=("BTC/USDT", "ETH/USDT")
        )
        snap = u.snapshot(date(2024, 6, 1))
        assert {"BTC/USDT", "ETH/USDT"} <= set(snap)

    def test_quote_currency_filter(self) -> None:
        u = CryptoTopNUniverse(top_n=10, ticker_source=_fake_tickers)
        snap = u.snapshot(date(2024, 6, 1))
        # FOO/BTC should never appear since quote_currency is USDT
        assert "FOO/BTC" not in snap

    def test_seed_fallback_when_no_source(self) -> None:
        u = CryptoTopNUniverse(top_n=5)
        snap = u.snapshot(date(2024, 6, 1))
        assert len(snap) == 5
        assert snap.metadata["source"] == "seed"

    def test_source_empty_strict_raises(self) -> None:
        u = CryptoTopNUniverse(
            top_n=5,
            ticker_source=lambda _: {},
            use_seed_on_error=False,
        )
        with pytest.raises(UniverseDataUnavailableError):
            u.snapshot(date(2024, 6, 1))

    def test_source_empty_fallback_to_seed(self) -> None:
        u = CryptoTopNUniverse(top_n=5, ticker_source=lambda _: {})
        snap = u.snapshot(date(2024, 6, 1))
        assert snap.metadata["source"] == "seed"
        assert len(snap) == 5

    def test_invalid_top_n(self) -> None:
        with pytest.raises(ValueError):
            CryptoTopNUniverse(top_n=0)

    def test_duplicated_anchors_deduped(self) -> None:
        u = CryptoTopNUniverse(
            top_n=2,
            ticker_source=_fake_tickers,
            anchors=("BTC/USDT", "BTC/USDT", "ETH/USDT"),
        )
        snap = u.snapshot(date(2024, 6, 1))
        # Uniqueness preserved
        assert len(set(snap)) == len(snap)

    def test_anchor_wrong_quote_currency_is_skipped(self) -> None:
        # anchor in a different quote currency should be filtered out,
        # not forced into a USDT-only universe.
        u = CryptoTopNUniverse(
            top_n=1,
            ticker_source=_fake_tickers,
            anchors=("FOO/BTC",),
        )
        snap = u.snapshot(date(2024, 6, 1))
        assert "FOO/BTC" not in snap
