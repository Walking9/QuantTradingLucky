"""Generate notebooks/M05_backtest_traps.ipynb.

Run once to materialise the notebook. The notebook itself is the
deliverable; this script is committed so the structure can be
regenerated / extended without hand-editing JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "M05_backtest_traps.ipynb"


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
        """# M5: 回测陷阱演示 — 前视偏差如何把噪声算成 Sharpe 3.0

本 Notebook 是 M5 自研向量化回测引擎 (`quant_lucky.backtest.VectorEngine`) 的姊妹篇，目的不是「炫耀回测有多好看」，而是把**回测里最危险的几个坑**摆到台面上：

1. **前视偏差 (look-ahead bias)**：用 `t` 时刻的因子值预测 `t` 时刻的收益，结果会作弊。
2. **数据窥探 (data snooping)**：在同一份样本上扫超参数，挑出来的"最优"在样本外是垃圾。
3. **生存偏差 (survivorship bias)**：只用今天还活着的标的回测，自动剔掉破产/退市的损失。
4. **成本缺失 (costless backtest)**：不扣手续费/滑点的多空策略，几乎一定虚高。

每个陷阱我们用同一份**合成噪声数据**做对比 — 真实市场数据反而会让"演示"被市场风格污染，纯随机数据才能干净地展示"代码 bug → 假 Alpha"的因果关系。

> 关键观点：**Sharpe > 3 的日频策略，99% 是你代码有 bug**。这条经验是 Marcos López de Prado 反复强调的。
"""
    ),
    code(
        """from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_lucky.backtest import VectorEngine, compute_performance

SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
"""
    ),
    md(
        """## 0. 合成数据：纯噪声

10 只资产，500 个交易日，日收益 ~ N(0, 1%)，**无任何 Alpha**。这意味着「理论上」任何策略的样本期望 Sharpe ≈ 0，超过这个数字的部分要么是抽样波动，要么是 bug。
"""
    ),
    code(
        """N_DAYS = 500
N_ASSETS = 10
dates = pd.date_range("2022-01-01", periods=N_DAYS, freq="B")
assets = [f"S{i}" for i in range(N_ASSETS)]

# 日对数收益 i.i.d. — 零均值，1% 标准差。
log_ret = rng.normal(0.0, 0.01, size=(N_DAYS, N_ASSETS))
prices = pd.DataFrame(
    np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
    index=dates,
    columns=assets,
)
asset_returns = prices.pct_change(fill_method=None)
print(f"asset 日收益均值：{asset_returns.stack().mean():+.5f}")
print(f"asset 日收益标准差：{asset_returns.stack().std():.5f}")
"""
    ),
    md(
        """## 1. 陷阱一：前视偏差

**Bug 配方**：用今天的因子 `f[t]` 去配今天的收益 `r[t]`。听上去显然是错的，但在 pandas 里写起来「太顺手」了 — 一行 `(factor * returns).sum(axis=1)` 就出来了。

我们用一个"oracle 因子" = `sign(今日收益)`，演示这个 bug 能把零 Alpha 噪声炒出多漂亮的曲线。
"""
    ),
    code(
        """# 「Oracle」: 直接知道今天的方向（这就是 look-ahead 的本质 — 用了未来信息）。
oracle_signal = np.sign(asset_returns).fillna(0.0)
# 归一化为 gross=1 的多空仓位。
denom = oracle_signal.abs().sum(axis=1).replace(0, np.nan)
oracle_weights = oracle_signal.div(denom, axis=0).fillna(0.0)

# Bug 写法：weights[t] 直接乘 returns[t]，不 shift。
biased_returns = (oracle_weights * asset_returns).sum(axis=1)
biased_report = compute_performance(biased_returns)

# 正确写法：用 VectorEngine — 内部 shift(1)。
correct_result = VectorEngine(cost_bps=0.0).run(oracle_weights, prices)

print("【BUG 版（前视）】", biased_report)
print()
print("【正确版（无前视）】", correct_result.report)
"""
    ),
    md(
        """**读法**：

- BUG 版年化 Sharpe **天花板级**，年化收益荒谬。这就是「我做出了 Renaissance 级策略」的常见来源。
- 正确版在零 Alpha 噪声上跑出来的 Sharpe 在 0 附近 — 符合理论预期。

`VectorEngine` 通过强制把 weights 内部 shift(1) 把这个 bug 从源头堵死。看引擎源码 `src/quant_lucky/backtest/vector.py:127` 的注释。
"""
    ),
    code(
        """fig, ax = plt.subplots()
((1 + biased_returns).cumprod()).plot(ax=ax, label=f"BUG: Sharpe={biased_report.sharpe:.2f}")
correct_result.portfolio_value.plot(ax=ax, label=f"正确: Sharpe={correct_result.report.sharpe:.2f}")
ax.set_yscale("log")
ax.set_title("Oracle 策略：前视 vs 正确 — 同一份零 Alpha 噪声")
ax.legend()
ax.set_ylabel("净值 (log)")
plt.show()
"""
    ),
    md(
        """## 2. 陷阱二：数据窥探（参数最优化）

**Bug 配方**：在同一份样本上扫一个超参数，挑 Sharpe 最高的报出去。即便策略本身是噪声，**只要超参数空间够大，总有一组参数"碰巧"漂亮**。

我们用「过去 N 日动量」做演示：在零 Alpha 噪声上扫 N=1..50。
"""
    ),
    code(
        """def momentum_strategy(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    \"\"\"按过去 N 日收益排序，做多前 30%、做空后 30%。
    返回 (date × asset) 的 weights DataFrame。\"\"\"
    mom = prices.pct_change(window, fill_method=None)
    n = mom.shape[1]
    long_n = max(1, int(0.3 * n))
    weights = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    for date, row in mom.iterrows():
        if row.notna().sum() < 2 * long_n:
            continue
        ranked = row.dropna().sort_values()
        shorts = ranked.iloc[:long_n].index
        longs = ranked.iloc[-long_n:].index
        weights.loc[date, longs] = 1.0 / long_n
        weights.loc[date, shorts] = -1.0 / long_n
    return weights

engine = VectorEngine(cost_bps=0.0)
sharpes = {}
for w_size in range(1, 51):
    weights = momentum_strategy(prices, w_size)
    r = engine.run(weights, prices)
    sharpes[w_size] = r.report.sharpe
sharpes = pd.Series(sharpes, name="Sharpe")
print(f"扫 50 个窗口的 Sharpe 区间：[{sharpes.min():.2f}, {sharpes.max():.2f}]")
print(f"中位数：{sharpes.median():.2f}")
print(f"最佳窗口 = {int(sharpes.idxmax())}，Sharpe = {sharpes.max():.2f}")
"""
    ),
    code(
        """fig, ax = plt.subplots()
sharpes.plot(ax=ax, marker="o", markersize=3)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axhline(sharpes.max(), color="red", linewidth=0.8, linestyle=":", label=f"最佳={sharpes.max():.2f}")
ax.set_xlabel("动量窗口 (日)")
ax.set_ylabel("回测 Sharpe")
ax.set_title("零 Alpha 噪声上扫窗口 — 「最佳参数」是抽样噪声")
ax.legend()
plt.show()
"""
    ),
    md(
        """**读法**：理论上所有曲线都应在 0 附近震荡。我们看到的 Sharpe 范围却跨越 ±1 — 这都是抽样噪声。如果直接把「最佳窗口」当结论上报，等于把噪声当 Alpha 卖。

**对策**：
1. **样本外测试 (Out-of-Sample)**：留出 30% 数据，调参只看前 70%，最后才在留出集上跑一次。
2. **Walk-Forward**：滚动重新调参 + 滚动验证。
3. **Deflated Sharpe Ratio** (López de Prado)：对扫的参数数量做惩罚。

我们这里只演示陷阱本身。完整 Walk-Forward 留到 M6 经典策略复现时实现。
"""
    ),
    md(
        """## 3. 陷阱三：成本缺失

**Bug 配方**：不扣手续费 + 滑点，做高换手多空策略。

继续用上面那个最佳窗口的动量策略，对比「零成本」vs「A 股 10 bps 单边」的净值曲线。
"""
    ),
    code(
        """best_window = int(sharpes.idxmax())
weights = momentum_strategy(prices, best_window)

free = VectorEngine(cost_bps=0.0).run(weights, prices)
realistic = VectorEngine(cost_bps=10.0).run(weights, prices)

print(f"零成本 Sharpe : {free.report.sharpe:+.2f}    年化收益: {free.report.annual_return:+.2%}")
print(f"含 10bps 成本: {realistic.report.sharpe:+.2f}    年化收益: {realistic.report.annual_return:+.2%}")
print(f"年化换手     : {free.report.turnover_annual:.1f} 次")

fig, ax = plt.subplots()
free.portfolio_value.plot(ax=ax, label=f"零成本 (Sharpe={free.report.sharpe:.2f})")
realistic.portfolio_value.plot(ax=ax, label=f"含成本 (Sharpe={realistic.report.sharpe:.2f})")
ax.set_title(f"动量(window={best_window}) — 成本是死亡判官")
ax.legend()
ax.set_ylabel("净值")
plt.show()
"""
    ),
    md(
        """**读法**：高换手策略 + 真实成本 → 「最优」曲线立刻褪色。

`VectorEngine` 故意把成本设计成**必须显式指定**：不传 `cost_bps` 也不传 `cost_model` 会直接报错。这是为了禁止「我先不管成本，回头再加」这种思维 — 因为「回头再加」往往就忘了。
"""
    ),
    md(
        """## 4. 陷阱四：生存偏差（合成数据演示）

**Bug 配方**：只用「今天还在交易的标的」做回测，自动剔除了所有破产/退市/被合并的样本。

我们在合成数据上加一个「人为退市」机制 — 把 10 只资产里收益最差的 3 只在中段强制清零（模拟退市），看看「只用幸存者」回测的结果。
"""
    ),
    code(
        """# 找到全期累计收益最差的 3 只 — 这就是「死掉的人」。
total_ret = prices.iloc[-1] / prices.iloc[0] - 1.0
losers = total_ret.nsmallest(3).index.tolist()
winners = total_ret.nlargest(7).index.tolist()
print("幸存者 (winners):", winners)
print("被退市的 (losers):", losers)

# 「幸存者偏差」回测：只在 winners 上跑 equal-weight 买入持有。
ew_winners_only = pd.DataFrame(
    1.0 / len(winners), index=prices.index, columns=prices.columns
)
for asset in losers:
    ew_winners_only[asset] = 0.0
# 「全样本」回测：把 losers 也算进去。
ew_full = pd.DataFrame(0.1, index=prices.index, columns=prices.columns)

engine = VectorEngine(cost_bps=0.0)
r_winners = engine.run(ew_winners_only, prices)
r_full = engine.run(ew_full, prices)

print(f"\\n只用 winners (有偏)：年化={r_winners.report.annual_return:+.2%}, Sharpe={r_winners.report.sharpe:+.2f}")
print(f"含 losers (无偏)   ：年化={r_full.report.annual_return:+.2%}, Sharpe={r_full.report.sharpe:+.2f}")
"""
    ),
    md(
        """**读法**：哪怕在零 Alpha 数据上，事后只挑赢家也会得到一条「漂亮」的曲线。真实市场上同样的逻辑就是 **A 股 ST/退市股、美股破产股、加密死币**。

**对策**：
- A 股：用 Tushare/Wind 拉**包含已退市股的全量历史**，按时点动态构建标的池（M2 `universe/csi300.py` 已经预留接口，后续 M5 末/ M7 接入）。
- 美股：CRSP 数据集是标杆；免费档可以用 yfinance 拉 delisted symbols，但精度有限。
- 加密：每个交易所的 delisting 都要单独建表。
"""
    ),
    md(
        """## 5. 小结：本月学到的「不可作弊清单」

| 陷阱 | 我们的引擎守 | 还要自己守 |
|---|---|---|
| 前视偏差 | ✅ `VectorEngine` 强制 `weights.shift(1)` | 自己写因子时要保证因子 `f[t]` 只用 `≤ t-1` 的数据 |
| 成本缺失 | ✅ `cost_bps` / `cost_model` 必须显式传 | 选合理的成本水平；A 股别用美股 1bps |
| 数据窥探 | ⚠️ 引擎不知道你跑了几次 | IS/OOS 切分 + Walk-Forward + 留最终样本 |
| 生存偏差 | ⚠️ 引擎不知道标的池怎么来的 | 历史时点 universe；包含退市股 |
| 微结构偏差 | ⚠️ 当前没建模 | 涨跌停、停牌、流动性约束 → M8 事件引擎时建模 |

**下一步 (M6)**：把这套防护用到 3 个经典策略复现上 — 美股动量、A 股双均线、跨市场风险平价 — 每一个都要扣成本、做 OOS、写报告。
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(NB, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
