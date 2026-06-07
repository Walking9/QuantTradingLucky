# risk_parity_multi_asset — generated results

- Data source: **real** (3 assets, 2021-03-10 → 2024-12-30)
- Cost: **15.0 bps/side**, annualisation **252**

## Headline metrics (net of cost)

| window | ann_return | ann_vol | sharpe | max_dd | hit_rate |
|---|---|---|---|---|---|
| full | 8.91% | 17.38% | 0.51 | -32.62% | 46.44% |
| in-sample | -5.51% | 18.00% | -0.31 | -32.62% | 41.12% |
| out-of-sample | 42.37% | 15.68% | 2.70 | -7.52% | 58.80% |
| benchmark | 16.92% | 28.76% | 0.59 | -45.18% | 51.19% |

- Annualised one-way turnover: **1.48**

## Attribution vs benchmark (single-factor OLS)

- Beta: **0.513**
- Annualised alpha: **0.23%**
- Correlation: **0.849** (R² 0.721)

## Walk-Forward (parameter generalisation)

- Splits: **8**, trials: **32**
- Aggregate OOS Sharpe: **1.51**
- Deflated Sharpe Ratio: **0.210** (P[true Sharpe > 0] after 32 trials)

## Parameter sensitivity (OOS Sharpe)

| param | value | full_sharpe | oos_sharpe | is−oos |
|---|---|---|---|---|
| vol_window | 21 | 0.43 | 2.94 | -3.35 |
| vol_window | 42 | 0.46 | 2.66 | -2.99 |
| vol_window * | 63 | 0.51 | 2.70 | -3.01 |
| vol_window | 126 | 0.43 | 2.66 | -3.11 |
| rebalance | W | 0.51 | 2.64 | -2.92 |
| rebalance * | M | 0.51 | 2.70 | -3.01 |

`*` = base configuration.
