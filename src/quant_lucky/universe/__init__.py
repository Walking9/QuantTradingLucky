"""Tradeable universe construction.

Builds and maintains asset pools used by strategies, e.g.:

- CSI 300 / CSI 500 / CSI 1000 constituents (point-in-time).
- S&P 500 constituents with historical add/remove events.
- BTC/ETH + top-N by market cap / volume.

All universes MUST be **point-in-time correct** — a symbol that joined
the index in 2018 should not appear in a 2015 universe snapshot.
This is the primary defence against survivorship bias.

Typical usage
-------------
::

    from quant_lucky.universe import CSI300Universe, SP500Universe, CryptoTopNUniverse

    cn = CSI300Universe()
    print(cn.members())               # uses seed list

    us = SP500Universe(fetcher=my_sp500_loader)
    print(us.snapshot(as_of=date(2024, 6, 30)))

    crypto = CryptoTopNUniverse(
        top_n=10,
        ticker_source=lambda _: {"BTC/USDT": 1e9, "ETH/USDT": 8e8},
    )
    print(crypto.members())
"""

from __future__ import annotations

from quant_lucky.universe.base import (
    UniverseBuilder,
    UniverseDataUnavailableError,
    UniverseError,
    UniverseSnapshot,
    coerce_date,
)
from quant_lucky.universe.crypto_topn import CryptoTopNUniverse
from quant_lucky.universe.csi300 import CSI300Universe
from quant_lucky.universe.sp500 import SP500Universe
from quant_lucky.universe.static import StaticUniverse

__all__ = [
    # Concrete
    "CSI300Universe",
    "CryptoTopNUniverse",
    "SP500Universe",
    "StaticUniverse",
    # Base
    "UniverseBuilder",
    "UniverseDataUnavailableError",
    "UniverseError",
    "UniverseSnapshot",
    "coerce_date",
]
