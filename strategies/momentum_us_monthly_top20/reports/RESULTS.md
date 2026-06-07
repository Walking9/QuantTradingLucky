# momentum_us_monthly_top20 — generated results

- Data source: **synthetic** (60 assets, 2018-01-02 → 2023-10-18)
- Cost: **10.0 bps/side**, annualisation **252**

## Headline metrics (net of cost)

| window | ann_return | ann_vol | sharpe | max_dd | hit_rate |
|---|---|---|---|---|---|
| full | 0.32% | 4.04% | 0.08 | -7.81% | 41.47% |
| in-sample | 0.34% | 3.79% | 0.09 | -7.81% | 37.90% |
| out-of-sample | 0.28% | 4.56% | 0.06 | -6.04% | 49.78% |
| benchmark | 13.01% | 16.28% | 0.80 | -26.06% | 50.40% |

- Annualised one-way turnover: **4.38**

## Attribution vs benchmark (single-factor OLS)

- Beta: **-0.002**
- Annualised alpha: **0.34%**
- Correlation: **-0.007** (R² 0.000)

## Walk-Forward (parameter generalisation)

- Splits: **8**, trials: **32**
- Aggregate OOS Sharpe: **0.19**
- Deflated Sharpe Ratio: **0.000** (P[true Sharpe > 0] after 32 trials)

## Parameter sensitivity (OOS Sharpe)

| param | value | full_sharpe | oos_sharpe | is−oos |
|---|---|---|---|---|
| lookback | 126.0 | -0.04 | -0.25 | 0.31 |
| lookback | 189.0 | 0.10 | 0.45 | -0.53 |
| lookback * | 252.0 | 0.08 | 0.06 | 0.03 |
| lookback | 315.0 | -0.22 | -0.69 | 0.73 |
| skip | 0.0 | 0.60 | 0.51 | 0.14 |
| skip * | 21.0 | 0.08 | 0.06 | 0.03 |
| skip | 42.0 | 0.26 | -0.01 | 0.42 |
| top_quantile | 0.1 | 0.41 | 0.90 | -0.77 |
| top_quantile * | 0.2 | 0.08 | 0.06 | 0.03 |
| top_quantile | 0.3 | 0.38 | -0.25 | 0.95 |

`*` = base configuration.
