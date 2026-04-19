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

策略还没开始写？这很正常。至少到 M6 才会有第一个完整策略。
