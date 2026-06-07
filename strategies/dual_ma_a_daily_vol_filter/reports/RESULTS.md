# dual_ma_a_daily_vol_filter — generated results

- Data source: **real** (8 assets, 2022-01-03 → 2023-12-28)
- Cost: **30.0 bps/side**, annualisation **252**

## Headline metrics (net of cost)

| window | ann_return | ann_vol | sharpe | max_dd | hit_rate |
|---|---|---|---|---|---|
| full | -6.18% | 7.23% | -0.85 | -14.49% | 33.68% |
| in-sample | -3.91% | 7.07% | -0.55 | -8.66% | 29.88% |
| out-of-sample | -11.42% | 7.60% | -1.50 | -9.02% | 42.47% |
| benchmark | -7.29% | 19.77% | -0.37 | -30.79% | 43.18% |

- Annualised one-way turnover: **4.95**

## Attribution vs benchmark (single-factor OLS)

- Beta: **0.210**
- Annualised alpha: **-4.65%**
- Correlation: **0.574** (R² 0.330)

## Walk-Forward (parameter generalisation)

- Splits: **4**, trials: **12**
- Aggregate OOS Sharpe: **-1.93**
- Deflated Sharpe Ratio: **0.000** (P[true Sharpe > 0] after 12 trials)

## Parameter sensitivity (OOS Sharpe)

| param | value | full_sharpe | oos_sharpe | is−oos |
|---|---|---|---|---|
| fast | 10.0 | -0.95 | -2.40 | 2.02 |
| fast * | 20.0 | -0.85 | -1.50 | 0.95 |
| fast | 30.0 | -1.06 | -1.29 | 0.35 |
| slow | 40.0 | -1.04 | -1.70 | 0.91 |
| slow * | 60.0 | -0.85 | -1.50 | 0.95 |
| slow | 90.0 | -0.65 | -1.13 | 0.69 |
| vol_max | 0.3 | -1.35 | -1.56 | 0.30 |
| vol_max * | 0.4 | -0.85 | -1.50 | 0.95 |
| vol_max | 0.5 | -0.99 | -1.56 | 0.79 |
| vol_max | nan | -0.91 | -1.55 | 0.89 |

`*` = base configuration.
