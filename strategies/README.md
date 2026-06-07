# Strategies

具体策略实现目录。每个策略一个子目录，自成一个可独立运行的包。

## 目录约定

```
strategies/<strategy_name>/
├── README.md          # 策略说明：理论来源、假设、限制
├── config.yaml        # 可调参数（回测区间、交易池、费率、参数）
├── strategy.py        # 策略主体：信号生成 + 仓位规则
├── run_backtest.py    # CLI 入口，调用 quant_lucky.backtest
├── tests/             # 策略级测试（回测结果回归、关键参数敏感性）
└── reports/           # 本策略产出的报告、图表
```

## 策略命名

`<类型>_<市场>_<频率>_<主题>`，例如：

- `momentum_us_monthly_top20`
- `dual_ma_a_daily_vol_filter`
- `risk_parity_multi_asset_weekly`
- `funding_arb_crypto_8h`

## 交付标准（每个策略必须有）

1. **可复现**：clone 仓库后 `python run_backtest.py` 即可跑出相同结果。
2. **可解释**：README 说清楚为什么应该有效、在什么环境下会失效。
3. **有样本外**：参数不能在全样本上调优；必须有 IS/OOS 划分或 Walk-Forward。
4. **有稳健性**：参数敏感性扫描 + 不同市场区间测试。
5. **有归因**：收益来源拆解（市场 Beta / 风格 / Alpha）。

## 已实现策略（M6）

| 包 | 策略 | 数据 | 报告 |
|---|---|---|---|
| `momentum_us_monthly_top20/` | 横截面 12-1 动量（J&T 1993） | 合成（离线无标普横截面） | [`reports/strategies/momentum_us.md`](../reports/strategies/momentum_us.md) |
| `dual_ma_a_daily_vol_filter/` | 双均线 + 波动率过滤趋势 | **真实** A 股 8 名 | [`reports/strategies/dual_ma_a.md`](../reports/strategies/dual_ma_a.md) |
| `risk_parity_multi_asset/` | 逆波动率风险平价 | **真实** BTC+SPY+CSI300 | [`reports/strategies/risk_parity_multi.md`](../reports/strategies/risk_parity_multi.md) |

共享评测层在 `quant_lucky.strategies.evaluation`（`run_research` / `evaluate_strategy` /
`parameter_sensitivity` / `attribution` / `write_artifacts`），CLI 辅助在
`quant_lucky.strategies.cli`。每个包 `python run_backtest.py` 即可复现并把
`metrics.json` / `RESULTS.md` / `equity_curve.csv` / `sensitivity.csv` 写入其 `reports/`。

> **关于 `tests/` 的一处刻意偏离**：上面的目录约定列了 `tests/`，但本仓库的包级回归
> 测试集中放在主测试树 `tests/strategies/test_packages.py`（按文件路径加载各包
> `strategy.py`），以便 CI（`testpaths=["tests"]`）能收集到。各包因此不再单独建 `tests/`。
