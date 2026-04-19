"""Crypto-specific data and strategies.

Differences from equity/futures justifying a dedicated module:

- 24/7 market, no settlement/halt boundaries.
- Perpetual swaps with funding rates (periodic cash flows).
- Multiple exchanges (CEX) with fragmented liquidity and diverging fees.
- On-chain data (Glassnode, CoinGecko) as an orthogonal signal source.
- Stable-coin depeg and exchange solvency risk.

Contents (progressively filled in months 9+):
- Multi-exchange data collectors via ``ccxt`` (async, rate-limited).
- Funding-rate arbitrage, triangular arb, cash-and-carry.
- On-chain factor extractors (active addresses, MVRV, exchange netflow).
"""
