"""S&P 500 universe builder.

The S&P 500 rebalances quarterly. Wikipedia maintains a constantly
updated membership list that several Python libraries scrape
(``yfinance`` included). Point-in-time history is available via
Compustat / CRSP / SPGlobal; free sources only give the **current**
membership.

Like :mod:`.csi300`, this builder supports a pluggable fetcher with a
seed fallback so the test suite and notebook demos do not require
network access.
"""

from __future__ import annotations

from collections.abc import Callable
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

#: Seed S&P 500 universe: top 40 by market cap at Q4 2025.
#: For tests and offline demos only — always inject a real fetcher in
#: production research.
SEED_UNIVERSE: tuple[str, ...] = (
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "GOOGL",  # Alphabet A
    "GOOG",  # Alphabet C
    "AMZN",  # Amazon
    "META",  # Meta
    "BRK-B",  # Berkshire Hathaway B
    "LLY",  # Eli Lilly
    "TSLA",  # Tesla
    "AVGO",  # Broadcom
    "JPM",  # JPMorgan Chase
    "V",  # Visa
    "UNH",  # UnitedHealth
    "XOM",  # ExxonMobil
    "WMT",  # Walmart
    "MA",  # Mastercard
    "PG",  # Procter & Gamble
    "JNJ",  # Johnson & Johnson
    "HD",  # Home Depot
    "ORCL",  # Oracle
    "COST",  # Costco
    "MRK",  # Merck
    "ABBV",  # AbbVie
    "CVX",  # Chevron
    "NFLX",  # Netflix
    "BAC",  # Bank of America
    "KO",  # Coca-Cola
    "ADBE",  # Adobe
    "PEP",  # PepsiCo
    "TMO",  # Thermo Fisher
    "CRM",  # Salesforce
    "CSCO",  # Cisco
    "MCD",  # McDonald's
    "ABT",  # Abbott
    "WFC",  # Wells Fargo
    "ACN",  # Accenture
    "LIN",  # Linde
    "DHR",  # Danaher
    "INTU",  # Intuit
)

Fetcher = Callable[[date], list[str] | tuple[str, ...]]


@dataclass
class SP500Universe(UniverseBuilder):
    """S&P 500 universe builder.

    Parameters are analogous to :class:`.csi300.CSI300Universe`.
    """

    fetcher: Fetcher | None = None
    use_seed_on_error: bool = True
    seed: tuple[str, ...] = field(default_factory=lambda: SEED_UNIVERSE)
    name: ClassVar[str] = "sp500"

    def snapshot(self, as_of: date | datetime | None = None) -> UniverseSnapshot:
        target = coerce_date(as_of)

        if self.fetcher is None:
            return self._seed_snapshot(target, reason="no fetcher configured")

        try:
            members = tuple(self.fetcher(target))
        except Exception as exc:  # pragma: no cover - defensive
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError(f"SP500 fetcher failed: {exc}") from exc
            logger.warning(
                "SP500 fetcher failed ({err}); falling back to seed list",
                err=exc,
            )
            return self._seed_snapshot(target, reason=f"fetcher error: {exc}")

        if not members:
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError("SP500 fetcher returned empty")
            return self._seed_snapshot(target, reason="fetcher returned empty")

        return UniverseSnapshot(
            as_of=target,
            members=members,
            metadata={"builder": self.name, "source": "fetcher"},
        )

    def _seed_snapshot(self, target: date, *, reason: str) -> UniverseSnapshot:
        return UniverseSnapshot(
            as_of=target,
            members=self.seed,
            metadata={"builder": self.name, "source": "seed", "reason": reason},
        )
