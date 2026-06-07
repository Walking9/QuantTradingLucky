# momentum_us_monthly_top20

Cross-sectional **"12-1" momentum** (Jegadeesh & Titman 1993): each month
rank the universe by trailing 12-month return *excluding the most recent
month* (`lookback=252`, `skip=21`), go long the top 20% and short the
bottom 20% equal-weight (dollar-neutral), hold one month.

## Run it

```bash
python strategies/momentum_us_monthly_top20/run_backtest.py
# planted-edge demo (inject time-series persistence):
python strategies/momentum_us_monthly_top20/run_backtest.py --set synthetic.autocorr=0.12
# long-only "buy top 20%" variant:
python strategies/momentum_us_monthly_top20/run_backtest.py --set signal.long_only=true
```

Artifacts (metrics JSON, generated `RESULTS.md`, equity curve, sensitivity
table) are written to `reports/`. Full write-up: `reports/strategies/momentum_us.md`.

## Theory

Winners keep winning and losers keep losing over 3–12 month horizons — one
of the most replicated anomalies in finance. Skipping the most recent month
sidesteps short-term reversal. A dollar-neutral long-short book isolates the
*cross-sectional* signal from market direction (target beta ≈ 0).

## Assumptions

- Monthly rebalance; weights held flat between month-ends (cost ~12×/yr).
- Equal weight within each leg; gross exposure 1.0 (0.5 long / 0.5 short).
- Cost: 10 bps/side on traded notional (liquid US large-caps).
- Shorting is free and frictionless (an idealisation — see limitations).

## ⚠️ Data honesty

A real S&P 500 cross-section is **not available offline** in this repo
(documented yfinance 429 history), and the few cached US names cannot form a
500-name ranking. The run therefore uses a **synthetic** panel and labels
itself `data: synthetic`. The default synthetic config has *homogeneous*
expected returns (`drift_dispersion=0`, `beta=1`), so with `autocorr=0` it is
a genuine **null**: momentum ranks pure noise and earns ≈0 (full Sharpe ~0.08,
**DSR ≈ 0.00**). Setting `autocorr>0` injects real time-series persistence and
the same machinery captures it (Sharpe ~3.4, **DSR ≈ 1.00**), with beta to the
market ≈ 0 in both cases. This validates the *pipeline and its honesty signals*
— it is **not** a discovered alpha.

> Pitfall found while building this: with the generator's *heterogeneous*
> defaults, a momentum sort posts a fake Sharpe ~2.5 even at `autocorr=0` by
> mechanically harvesting static drift/beta dispersion the generator fixed.
> See `docs/pitfalls.md`.

## Limitations

- Synthetic data → no claim about real momentum magnitude/decay.
- No borrow cost, short availability, or short-squeeze risk modelled.
- Equal-weight legs ignore liquidity/capacity; real S&P momentum is capacity
  constrained and crash-prone (2009 momentum crash).
- Package regression tests live in `tests/strategies/test_packages.py` (so CI
  collects them); this package ships no separate `tests/` dir by design.
