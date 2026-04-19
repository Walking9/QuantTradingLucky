"""Transaction cost models.

Covers:
- Commission (A-share, US, crypto — different structures).
- Stamp duty / transfer fees (A-share specifics).
- Slippage models (fixed bps, volume-proportional, square-root impact).
- Funding-rate cost for perpetual swaps.

Realistic cost modelling is the difference between a strategy that works
and one that bleeds. Default cost assumptions should err on the pessimistic
side; any "costless" backtest should be explicitly flagged.
"""
