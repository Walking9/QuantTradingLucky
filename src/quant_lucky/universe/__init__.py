"""Tradeable universe construction.

Builds and maintains asset pools used by strategies, e.g.:

- CSI 300 / CSI 500 / CSI 1000 constituents (point-in-time).
- S&P 500 constituents with historical add/remove events.
- BTC/ETH + top-N by market cap / volume.

All universes MUST be **point-in-time correct** — a symbol that joined
the index in 2018 should not appear in a 2015 universe snapshot.
This is the primary defence against survivorship bias.
"""
