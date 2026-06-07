# dual_ma_a_daily_vol_filter

Dual moving-average crossover with a **volatility filter** (trend-following,
A-shares). Hold a name while its fast SMA (20d) is above its slow SMA (60d)
**and** its realised volatility is below a ceiling (40% annualised); sit in
cash otherwise. Long-only — A-share retail cannot easily short.

## Run it

```bash
python strategies/dual_ma_a_daily_vol_filter/run_backtest.py
# turn the volatility filter off (pure dual-MA crossover):
python strategies/dual_ma_a_daily_vol_filter/run_backtest.py --set signal.vol_max=null
```

Artifacts go to `reports/`. Full write-up: `reports/strategies/dual_ma_a.md`.

## Theory

Trend systems profit from sustained directional moves and lose in choppy,
mean-reverting regimes. A dual-MA crossover defines the trend; the volatility
filter is a regime gate that drops a name when it gets too wild (where
crossovers whipsaw most). Weighting is **budget-per-name** (`gross/N` when a
name's gates are open, cash otherwise), so gross exposure *falls* when fewer
names qualify — the defensive behaviour we want.

## Data (real)

Eight liquid, sector-diverse A-share names (akshare daily, **2022-01 →
2023-12**, 484 aligned trading days), inner-joined on common dates. This is a
*small* basket over a *short, difficult* window — enough to exercise the
strategy and the evaluation framework on genuine data, **not** enough to draw
a population-level conclusion. The full CSI-300 history (incl. delisted names)
is an open M6 data task.

## Assumptions

- Cost: **30 bps/side**. A-shares pay ~3.5 bps to buy but ~53.5 bps to sell
  (0.05% stamp duty is sell-side only); 30 bps/side is a conservative
  *symmetric* blend. The true sell-side asymmetry is a modelling limitation.
- Long-only; down-trend / high-vol state is cash, not a short.
- Daily signals, but trades only on gate transitions → modest turnover.

## What to expect

2022–2023 A-shares were choppy and broadly down (the equal-weight basket
returned about **−7%/yr** buy & hold). A long-only trend strategy has no
up-trends to ride, so it **loses a little** — but it loses *calmly*: it cut
the max drawdown to roughly half the basket's and ran at about a third of its
volatility by sitting in cash through the declines. The Walk-Forward DSR is
≈0 — **no statistical edge in this window**, which is the honest finding, not
a strategy to deploy. Trend following needs trends; this sample didn't have
them.

## Limitations

- Tiny basket, short window, one (bad-for-trend) regime → not representative.
- No A-share T+1 settlement, price-limit (±10%) fill constraints, or
  suspension handling modelled (vector engine assumes close-to-close fills).
- Symmetric cost ignores the buy/sell stamp-duty asymmetry (pessimistic on
  buys, optimistic on sells).
- Package regression tests live in `tests/strategies/test_packages.py`.
