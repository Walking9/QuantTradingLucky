"""Crypto "Top-N by volume" universe builder.

Crypto does not have index committees; liquidity shifts daily and new
tokens appear constantly. The dominant universe construction rules are:

- **Top-N by 24h quote volume** on a primary exchange (Binance, OKX).
- **Top-N by market cap** (CoinGecko / CoinMarketCap).
- **Anchor pairs** — always include BTC/ETH even when volume drops.

This builder implements Top-N by volume with a pluggable ``ticker_source``
so tests can supply a deterministic mapping without hitting ccxt.

Quote currency handling
-----------------------
``quote_currency`` (default ``"USDT"``) restricts the universe to pairs
quoted in the same currency — otherwise the "volume" is not comparable
across pairs.

The ``anchor`` list guarantees certain symbols always remain in the
universe regardless of volume ranking; pass ``()`` to disable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import ClassVar

from quant_lucky.universe.base import (
    UniverseBuilder,
    UniverseDataUnavailableError,
    UniverseSnapshot,
    coerce_date,
)
from quant_lucky.utils.logging import logger

#: Baseline list used when no ``ticker_source`` is configured.
#: Ordered by 24h volume rank at the time of writing — sufficient for
#: offline demos. Format matches ccxt unified symbol notation.
SEED_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "TRX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "NEAR/USDT",
    "UNI/USDT",
    "ATOM/USDT",
    "APT/USDT",
    "OP/USDT",
    "ARB/USDT",
)

#: A ticker source returns a mapping ``symbol -> quote_volume`` for a
#: single exchange as of ``as_of``. Using a simple mapping (rather than
#: a heavier ccxt-flavoured structure) keeps the builder testable.
TickerSource = Callable[[date], Mapping[str, float]]


@dataclass
class CryptoTopNUniverse(UniverseBuilder):
    """Build the top-N crypto universe by 24h quote volume.

    Parameters
    ----------
    top_n:
        Number of symbols to retain.
    quote_currency:
        Restrict candidates to pairs quoted in this currency.
    anchors:
        Symbols forced into the universe even if they fail the volume
        ranking. BTC/ETH by default.
    ticker_source:
        Optional callable returning a ``symbol -> volume`` mapping.
    use_seed_on_error:
        Fall back to :data:`SEED_UNIVERSE` when the ticker source fails.
    """

    top_n: int = 20
    quote_currency: str = "USDT"
    anchors: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    ticker_source: TickerSource | None = None
    use_seed_on_error: bool = True
    seed: tuple[str, ...] = field(default_factory=lambda: SEED_UNIVERSE)
    exchange_name: str = "binance"
    name: ClassVar[str] = "crypto_topn"

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {self.top_n}")
        self.anchors = tuple(dict.fromkeys(self.anchors))  # dedupe preserving order

    def snapshot(self, as_of: date | datetime | None = None) -> UniverseSnapshot:
        target = coerce_date(as_of)

        if self.ticker_source is None:
            return self._seed_snapshot(target, reason="no ticker_source configured")

        try:
            tickers = dict(self.ticker_source(target))
        except Exception as exc:  # pragma: no cover - defensive
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError(f"Ticker source failed: {exc}") from exc
            logger.warning(
                "Crypto ticker source failed ({err}); using seed list",
                err=exc,
            )
            return self._seed_snapshot(target, reason=f"ticker_source error: {exc}")

        if not tickers:
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError("Ticker source returned empty")
            return self._seed_snapshot(target, reason="ticker_source empty")

        members = self._select(tickers)
        return UniverseSnapshot(
            as_of=target,
            members=members,
            metadata={
                "builder": self.name,
                "source": f"ticker_source({self.exchange_name})",
                "quote_currency": self.quote_currency,
                "top_n": self.top_n,
                "anchors": list(self.anchors),
            },
        )

    def _select(self, tickers: Mapping[str, float]) -> tuple[str, ...]:
        """Pick top-N by volume plus anchors, filtered by quote currency."""
        suffix = f"/{self.quote_currency}"
        quoted = {
            sym: vol
            for sym, vol in tickers.items()
            if sym.endswith(suffix) and vol is not None and vol >= 0
        }

        # Sort by volume desc, then symbol asc for deterministic ties.
        ranked = sorted(quoted.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [sym for sym, _ in ranked[: self.top_n]]

        # Merge anchors (deduplicated, preserving rank order then anchor order).
        # Anchors are only added if they actually match the quote currency.
        anchored = [a for a in self.anchors if a.endswith(suffix)]
        merged: list[str] = []
        seen: set[str] = set()
        for sym in (*top, *anchored):
            if sym not in seen:
                merged.append(sym)
                seen.add(sym)
        return tuple(merged)

    def _seed_snapshot(self, target: date, *, reason: str) -> UniverseSnapshot:
        return UniverseSnapshot(
            as_of=target,
            members=self.seed[: self.top_n],
            metadata={
                "builder": self.name,
                "source": "seed",
                "reason": reason,
                "exchange": self.exchange_name,
            },
        )
