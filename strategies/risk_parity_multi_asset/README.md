# risk_parity_multi_asset

Inverse-volatility **risk parity** across a cross-market basket: BTC, SPY
(US equity), CSI-300 (China equity). Each asset is weighted `∝ 1/σ` so it
contributes roughly equal *risk* rather than equal *capital*, rebalanced
monthly. The benchmark is naive equal weight — risk parity's null hypothesis.

## Run it

```bash
python strategies/risk_parity_multi_asset/run_backtest.py
# weekly rebalance variant:
python strategies/risk_parity_multi_asset/run_backtest.py --set signal.rebalance=W
```

Artifacts go to `reports/`. Full write-up: `reports/strategies/risk_parity_multi.md`.

## Theory

A 60/40 stock/bond book is ~90% equity *risk*. Risk parity fixes that by
down-weighting the volatile leg. The full equal-risk-contribution (ERC)
solution needs a covariance inversion that is ill-conditioned on short
samples; for a small, weakly-correlated basket the inverse-vol approximation
`w_i ∝ 1/σ_i` is within a hair of ERC and far more robust. (A proper ERC
optimiser via `cvxpy` is a natural M7 extension.)

## Data (real)

BTC-USDT (Binance), SPY (yfinance), 000300.SS / CSI-300 (yfinance), daily,
**2021-03 → 2024-12** (717 days), inner-joined on common trading days.

> **Annualisation:** the inner join keeps only days *all three* venues trade,
> dropping crypto weekends → ~252 bars/year, so `periods_per_year=252` (not
> 365). Forward-filling weekends instead would inject stale equity prices —
> a quiet look-ahead/staleness bug we avoid on purpose.

## What to expect — risk parity is a *risk* story, not a return story

Over this window inverse-vol sizing puts only **~12% in BTC** (73%
annualised vol) vs **~46% SPY / ~42% CSI-300** — equal weight would force 33%
each. The result is exactly what the theory predicts:

- **Volatility ~17% vs equal-weight ~29%**, **max drawdown ~−33% vs ~−45%**,
  **beta ~0.5** to the equal-weight basket (half the amplitude).
- Full-sample **Sharpe ~0.5 vs ~0.6** — *slightly below* equal weight,
  because equal weight rode more of BTC's 2023–24 rally. Risk parity gives up
  some upside in exchange for the risk reduction. That trade-off is the point.

The IS/OOS split is starkly regime-dependent (negative through the 2022
crypto bear, strongly positive in the 2023–24 recovery); the Walk-Forward DSR
is well below the 0.95 bar — **suggestive, not statistically significant** on
3 assets over <4 years. Honest framing: the *risk-reduction* mechanism is
robust and visible; the *return* outcome is regime- and window-dependent.

## Limitations

- Three assets, <4 years, one BTC boom-bust cycle → not a population claim.
- Inverse-vol ignores correlations (true ERC accounts for them); fine for a
  weakly-correlated basket, less so as assets co-move in a crisis.
- Holding weights flat between rebalances ignores intra-month drift (a
  documented approximation; a drift-aware NAV mode is future work).
- No leverage; real risk-parity funds lever the low-vol book to a vol target.
- Package regression tests live in `tests/strategies/test_packages.py`.
