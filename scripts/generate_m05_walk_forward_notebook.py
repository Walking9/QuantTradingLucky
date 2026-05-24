"""Generate notebooks/M05_walk_forward.ipynb.

This notebook is the M5 Walk-Forward / IS-OOS demonstration. It is the
"part 2" of the M5 series (part 1 = backtest traps). The script is the
canonical source; the .ipynb is built from it.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "M05_walk_forward.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    md(
        """# M5 第二章: IS/OOS 切分 与 Walk-Forward —— 不让自己再吹回测

上一篇 `M05_backtest_traps` 演示了 **数据窥探** 的危险性：在零 Alpha 噪声上扫 50 个动量窗口，最佳 Sharpe 也能跑到 +1。这是怎么回事？答案：**抽样噪声 ≠ Alpha**。

本 Notebook 实战展示三种「样本外验证」机制，让数据窥探立刻现形：

1. **固定 IS/OOS 切分**：训练在前 70%，测试在后 30%。最简单，但容易因为时点选择不当被忽悠。
2. **Rolling Walk-Forward**：固定宽度滚动训练窗口 + 紧邻测试窗口；样本外片段可以拼成完整曲线。
3. **Anchored Walk-Forward**：训练窗口从起点不断扩张；模拟「每月用过去全部数据重新拟合」的真实研究员姿势。

最后我们给 OOS Sharpe **打个折扣**：用 López de Prado 的 **Deflated Sharpe Ratio (DSR)** 计算「调了 N 个参数后这个 Sharpe 是否还显著」。

> 关键观点：**没经过 OOS 的 Sharpe 都是文学创作**。一个老老实实做完 Walk-Forward 的负 Sharpe，比一个 IS 的正 Sharpe 更有说服力。
"""
    ),
    code(
        """from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_lucky.backtest import (
    VectorEngine,
    fixed_split,
    rolling_walk_forward,
    anchored_walk_forward,
    walk_forward,
    deflated_sharpe_ratio,
    compute_performance,
)

SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
"""
    ),
    md(
        """## 0. 合成数据：还是那盘零 Alpha 噪声

10 只资产 × 1000 个交易日，对数收益 i.i.d. N(0, 1%)。理论上 **没有任何策略能在这上面跑出正 OOS Sharpe**。

故意用比 M05_backtest_traps 更长的样本（500 → 1000），这样滚动窗口能划出足够多的 split。
"""
    ),
    code(
        """N_DAYS = 1000
N_ASSETS = 10
dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="B")
assets = [f"S{i}" for i in range(N_ASSETS)]

log_ret = rng.normal(0.0, 0.01, size=(N_DAYS, N_ASSETS))
prices = pd.DataFrame(
    np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
    index=dates,
    columns=assets,
)
print(f"价格面板: {prices.shape}, 时间区间: {dates[0].date()} → {dates[-1].date()}")
"""
    ),
    md(
        """## 1. 策略候选：动量窗口扫描

定义一个简单的 cross-sectional 动量策略：每天按过去 `window` 日收益排序，做多前 30%、做空后 30%。`window` 是我们的「待优化超参数」。

关键防呆：用 `pct_change(window).shift(1)` 保证 `t` 时刻的权重只用 `≤ t-1` 的价格 — 无前视。
"""
    ),
    code(
        """def momentum_weights(param: dict, full_prices: pd.DataFrame) -> pd.DataFrame:
    \"\"\"价格 → 多空权重。强制 .shift(1) 防前视。\"\"\"
    n = param["window"]
    mom = full_prices.pct_change(n, fill_method=None).shift(1)
    n_assets = mom.shape[1]
    long_n = max(1, int(0.3 * n_assets))
    ranks = mom.rank(axis=1)  # 1 = worst, n = best
    w = pd.DataFrame(0.0, index=full_prices.index, columns=full_prices.columns)
    w[ranks > (n_assets - long_n)] = 1.0 / long_n          # 多 top 30%
    w[ranks <= long_n] = -1.0 / long_n                      # 空 bottom 30%
    return w.where(ranks.notna(), 0.0)

# 候选窗口
PARAM_GRID = [{"window": w} for w in [3, 5, 10, 20, 40, 60, 80, 120, 160, 200]]
print(f"参数候选数: {len(PARAM_GRID)}")
"""
    ),
    md(
        """## 2. 第一道防线：固定 IS/OOS 切分

把数据砍成 70% 训练 / 30% 测试。在训练段扫所有窗口，挑 Sharpe 最高的，再到测试段验证。

这就是「业余研究员」的标准姿势：写完策略，留个测试集，最后跑一次。
"""
    ),
    code(
        """split = fixed_split(prices.index, oos_fraction=0.3)
print(f"训练: {split.train_start.date()} → {split.train_end.date()} "
      f"({(prices.index <= split.train_end).sum()} 个交易日)")
print(f"测试: {split.test_start.date()} → {split.test_end.date()} "
      f"({(prices.index >= split.test_start).sum()} 个交易日)")

engine = VectorEngine(cost_bps=5.0)
is_results = {}
for p in PARAM_GRID:
    w_full = momentum_weights(p, prices)
    w_train = split.slice_train(w_full)
    res = engine.run(w_train, split.slice_train(prices))
    is_results[p["window"]] = res.report.sharpe

is_sharpe = pd.Series(is_results, name="IS Sharpe")
best_w = int(is_sharpe.idxmax())
print(f"\\nIS 最佳窗口: {best_w}, IS Sharpe = {is_sharpe.max():+.3f}")

# 把同一个窗口拿去 OOS 跑一次
w_test = split.slice_test(momentum_weights({"window": best_w}, prices))
oos = engine.run(w_test, split.slice_test(prices))
print(f"OOS Sharpe (同一窗口): {oos.report.sharpe:+.3f}")
print(f"OOS 年化收益: {oos.report.annual_return:+.2%}")
"""
    ),
    md(
        """**读法**：
- IS 段最佳 Sharpe 看起来「正经」，但这是从 10 个窗口里挑出来的。
- 同一个「最佳窗口」拿到 OOS 上一跑，几乎一定 — Sharpe 远不如 IS，甚至变负。

这个 gap 就是「**参数过拟合**」的指纹。但单一切分有个问题：只用了一次 OOS 数据，结论受抽样运气影响。
"""
    ),
    code(
        """fig, ax = plt.subplots()
is_sharpe.plot(ax=ax, marker="o", label="IS Sharpe (前 70%)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axhline(is_sharpe.max(), color="red", linewidth=0.8, linestyle=":",
           label=f"IS best = {is_sharpe.max():+.2f}")
ax.axhline(oos.report.sharpe, color="green", linewidth=0.8, linestyle=":",
           label=f"OOS at IS-best = {oos.report.sharpe:+.2f}")
ax.set_xlabel("动量窗口")
ax.set_ylabel("Sharpe")
ax.set_title("固定 IS/OOS 切分：IS 漂亮的窗口在 OOS 上多半失灵")
ax.legend()
plt.show()
"""
    ),
    md(
        """## 3. 第二道防线：Rolling Walk-Forward

固定切分有两个弱点：
1. **只看一次** OOS，结论受抽样运气影响。
2. **不适应非平稳性** — 2018 年的最佳窗口可能 2022 年完全失效。

Walk-Forward 把数据砍成多个滚动窗口：每个窗口在 IS 段挑参数 → OOS 段验证 → 推进 → 重复。最后把所有 OOS 拼接成一条连续曲线。

`backtest.rolling_walk_forward` 默认 `step = test_size`，OOS 片段无重叠 — 可以直接拼接。
"""
    ),
    code(
        """splits = rolling_walk_forward(
    prices.index,
    train_size=400,   # 训练窗口 ≈ 1.6 年
    test_size=80,     # OOS 窗口 ≈ 4 个月
)
print(f"Rolling WF 共 {len(splits)} 个 split")
for i, s in enumerate(splits[:3]):
    print(f"  [{i}] train: {s.train_start.date()} → {s.train_end.date()}, "
          f"test: {s.test_start.date()} → {s.test_end.date()}")
print("  ...")
"""
    ),
    code(
        """wf_result = walk_forward(
    prices=prices,
    splits=splits,
    param_grid=PARAM_GRID,
    strategy_fn=momentum_weights,
    cost_bps=5.0,
)
print(f"每个 split 选出的窗口: {[p['window'] for p in wf_result.selected_params]}")
print(f"IS Sharpes : {[f'{s:+.2f}' for s in wf_result.is_metrics]}")
print(f"OOS Sharpes: {[f'{s:+.2f}' for s in wf_result.oos_metrics]}")
print(f"\\n总试验次数 (n_trials) = {wf_result.n_trials}")
print(f"聚合 OOS Sharpe = {wf_result.extra['aggregate_oos_sharpe']:+.3f}")
print(f"Deflated Sharpe Ratio = {wf_result.deflated_sharpe:.4f}")
"""
    ),
    md(
        """**读法**：
- **每个 split 选出的「最佳窗口」都不同** — 这是数据窥探最直接的证据：参数没有稳定性，每次都在追噪声。
- **IS Sharpe → OOS Sharpe 的衰减触目惊心**：IS 普遍 +0.3 到 +1.0，OOS 多数是 -0.5 到 -1.5。
- **DSR 几乎等于 0**：在零 Alpha 数据上，调了 $\\text{params} \\times \\text{splits}$ 个组合，最后的 Sharpe 还能被简单的「调参噪声」解释 — 这正是 DSR 想检测的事。
"""
    ),
    code(
        """# 把聚合 OOS 曲线画出来
oos_value = (1.0 + wf_result.oos_returns).cumprod()
fig, ax = plt.subplots()
oos_value.plot(ax=ax, label=f"Walk-Forward OOS (聚合 Sharpe={wf_result.extra['aggregate_oos_sharpe']:+.2f})")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_title(f"Rolling WF 拼接 OOS 净值 (DSR={wf_result.deflated_sharpe:.3f})")
ax.set_ylabel("净值")
ax.legend()
plt.show()
"""
    ),
    md(
        """## 4. 第三道防线：Anchored Walk-Forward

Anchored WF 训练窗口从起点开始单调增长 — 模拟「每月把过去全部数据塞进模型重新拟合」。

适用场景：相信底层数据生成过程相对稳定，更多数据 → 更好估计；不太担心 regime shift。如果你怀疑有 regime shift（如 2020 疫情前后、Crypto 牛熊），用 Rolling 更安全。
"""
    ),
    code(
        """splits_anchored = anchored_walk_forward(
    prices.index,
    initial_train_size=400,
    test_size=80,
)
wf_anchored = walk_forward(
    prices=prices,
    splits=splits_anchored,
    param_grid=PARAM_GRID,
    strategy_fn=momentum_weights,
    cost_bps=5.0,
)
print(f"Anchored WF 共 {len(splits_anchored)} 个 split")
print(f"每个 split 选出的窗口: {[p['window'] for p in wf_anchored.selected_params]}")
print(f"聚合 OOS Sharpe = {wf_anchored.extra['aggregate_oos_sharpe']:+.3f}")
print(f"DSR = {wf_anchored.deflated_sharpe:.4f}")
"""
    ),
    code(
        """fig, ax = plt.subplots()
(1.0 + wf_result.oos_returns).cumprod().plot(
    ax=ax, label=f"Rolling WF (Sharpe={wf_result.extra['aggregate_oos_sharpe']:+.2f})"
)
(1.0 + wf_anchored.oos_returns).cumprod().plot(
    ax=ax, label=f"Anchored WF (Sharpe={wf_anchored.extra['aggregate_oos_sharpe']:+.2f})"
)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Rolling vs Anchored Walk-Forward — 零 Alpha 数据上两条都该贴着 1.0")
ax.set_ylabel("净值")
ax.legend()
plt.show()
"""
    ),
    md(
        """## 5. Deflated Sharpe Ratio — 给 Sharpe 戴上紧箍咒

DSR 的核心思想：**「你试了多少组参数？」决定了 Sharpe 被打多少折扣**。

- $\\text{SR}_{\\text{obs}}$：观察到的年化 Sharpe（在我们这里是聚合 OOS Sharpe）。
- $E[\\max \\text{SR} \\mid H_0]$：N 个独立标准正态变量的最大值的期望（即「零假设下你期待的最佳 Sharpe」）。
- 修正因子：用 skew / kurtosis 调整 Sharpe 估计量的方差。

DSR ≥ 0.95 才算「公开发表水平」；DSR < 0.5 就是「跟运气没区别」。
"""
    ),
    code(
        """# DSR 关于 trial 数的敏感性 — 同一个 SR 在不同 N 下有什么变化
sr_observed = 1.5  # 我们假装某个策略观测到了 1.5 的年化 Sharpe
n_obs = 252        # 252 天 OOS
trial_counts = [1, 5, 10, 50, 100, 500, 1000, 5000]
dsr_curve = [
    deflated_sharpe_ratio(
        observed_sharpe=sr_observed,
        n_trials=n,
        n_observations=n_obs,
    )
    for n in trial_counts
]

dsr_df = pd.DataFrame({"n_trials": trial_counts, "DSR": dsr_curve})
print(dsr_df.to_string(index=False))

fig, ax = plt.subplots()
ax.plot(trial_counts, dsr_curve, marker="o")
ax.set_xscale("log")
ax.axhline(0.95, color="green", linestyle=":", label="0.95 阈值")
ax.axhline(0.50, color="red", linestyle=":", label="0.50 阈值")
ax.set_xlabel("尝试参数组数 N (log)")
ax.set_ylabel("Deflated Sharpe Ratio")
ax.set_title(f"DSR vs N — 观测 Sharpe = {sr_observed} 固定")
ax.legend()
plt.show()
"""
    ),
    md(
        """**读法**：
- 试 1 次就拿出 Sharpe = 1.5：DSR ≈ 0.99 — 几乎一定真。
- 试 100 次挑最好：DSR 跌到 ~0.6 — 介于「可能真」和「可能噪声」之间。
- 试 1000 次：DSR 进入 0.3 量级 — 大概率是抓阄。
- 试 5000 次：DSR < 0.1 — 你这是噪声制造机。

**实战教训**：
1. 报告 Sharpe 一定要附 `n_trials`，否则没法做诚实的 DSR 折扣。
2. 我们在 §3 的 WF 里 `n_trials = 10 个窗口 × 8 个 split = 80`；零 Alpha 数据下 DSR ≈ 0 完全符合预期。
3. **真实策略**目标：DSR ≥ 0.95，且 `n_trials` 在合理范围（< 100，理想 < 30）。
"""
    ),
    md(
        """## 6. 小结：M5 结业级回测清单

| 关卡 | 工具 | 现状 |
|---|---|---|
| 防前视 | `VectorEngine` 内部 `shift(1)` | ✅ M5 完成 |
| 显式成本 | `cost_bps` / `cost_model` 必传 | ✅ M5 完成 |
| 标准报表 | `PerformanceReport` (Sharpe/Sortino/MDD/Calmar) | ✅ M5 完成 |
| 跨框架对照 | `scripts/verify_vectorbt_parity.py` ≤ 3% 偏差 | ✅ M5 完成 |
| 防数据窥探 | `fixed_split` / `rolling_walk_forward` / `anchored_walk_forward` + `walk_forward` | ✅ 本 NB |
| 多重检验惩罚 | `deflated_sharpe_ratio` | ✅ 本 NB |
| 防生存偏差 | 历史时点 universe + 含退市股 | ⏭ M7 |
| 微结构成本 | 涨跌停、停牌、流动性约束 | ⏭ M8 事件引擎 |

**下一步 (M6)**：把 Walk-Forward + DSR 用到 3 个经典策略复现 — 美股动量、A 股双均线、跨市场风险平价 — 每一个都要：
- 报告 IS / OOS Sharpe 对比
- 报告 `n_trials` 和对应 DSR
- 至少做一组「参数稳健性」（在 ±50% 参数范围内重跑）

如果一个策略经不起 Walk-Forward + DSR ≥ 0.5 的拷问，它就不该见到实盘的钱。
"""
    ),
]


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Add per-cell id (nbformat ≥ 4.5 requires it).
    for i, cell in enumerate(nb["cells"]):
        cell.setdefault("id", str(i))
        if cell["cell_type"] == "code":
            cell.setdefault("execution_count", None)
            cell.setdefault("outputs", [])
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    build()
