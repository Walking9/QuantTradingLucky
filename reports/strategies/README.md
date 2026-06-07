# M6 策略研究报告

M6「经典策略复现与评价」的三份完整研究报告。每个策略都跑同一套**诚实交付包**
（M5 月报定下的标准）：full / IS / OOS 对比、Walk-Forward + Deflated Sharpe Ratio
反过拟合数、参数敏感性扫描、对基准的收益归因。共享评测层在
`quant_lucky.strategies.evaluation`，可运行包在 `strategies/<name>/`。

| 报告 | 市场 | 数据 | 头条结论 | DSR |
|---|---|---|---|---:|
| [`momentum_us`](momentum_us.md) | 美股（合成） | synthetic | 零假设面板上动量赚不到钱（Sharpe 0.08）；植入真边后同一套机器抓住它（Sharpe 3.45） | 0.00 → 1.00 |
| [`dual_ma_a`](dual_ma_a.md) | A 股 | **真实** 8 名 2022–23 | 诚实负结果：震荡市趋势策略小亏，但回撤/波动压到基准一半以下 | 0.00 |
| [`risk_parity_multi`](risk_parity_multi.md) | BTC+SPY+CSI300 | **真实** 2021–24 | 风险平价兑现降风险承诺（vol 17% vs 29%，MDD -33% vs -45%），但非收益增强 | 0.21 |

## 共同方法学

- **No-look-ahead**：所有权重 `weights[t]` 仅用 ≤ t 的信息；引擎内部 `shift(1)` 赚 (t, t+1]。
- **成本显式**：每个策略用统一的 `cost_bps`（per-side），full/IS/OOS/WF 数字可直接比较。
- **样本外纪律**：固定 IS/OOS 切分看"不同区间"，Walk-Forward 看"重调参后是否泛化"，
  DSR 把多重检验的水分挤干（`n_trials` 越大、尾部越肥，折扣越狠）。
- **归因**：对市场/等权基准做单因子 OLS，beta 衡量市场敞口，alpha 衡量残差。

## 三条贯穿 M6 的诚实信号

1. **DSR 是把 IS Sharpe 请回现实的工具**：momentum 零假设 DSR=0.00、有真边 DSR=1.00，
   是"为什么论文要报告试验次数 N"的代码级答案。
2. **回测可以因不可交易的理由而好看**：合成 momentum 在 `autocorr=0` 下若用异质 generator
   会跑出假 Sharpe 2.5（收割写死的静态离散度）——见 `docs/pitfalls.md`。
3. **负结果也是结果**：dual_ma 在 2022–23 A 股诚实地亏钱，但如实展示了趋势策略的防御性；
   不去 curve-fit 成正收益，才是 M6 想练的肌肉。

## 复现

```bash
python strategies/momentum_us_monthly_top20/run_backtest.py     # 合成，离线可跑
python strategies/dual_ma_a_daily_vol_filter/run_backtest.py    # 需 akshare 缓存
python strategies/risk_parity_multi_asset/run_backtest.py       # 需 yfinance+ccxt 缓存
```

每次运行把机器生成的 `metrics.json` / `RESULTS.md` / `equity_curve.csv` / `sensitivity.csv`
写入对应包的 `reports/`；本目录的 `.md` 是据此手写的研究解读。
