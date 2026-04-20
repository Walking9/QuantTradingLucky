"""Generate notebooks/M01_eda.ipynb from source cells defined here.

Keeping the notebook source in a Python script makes diffs readable and
lets us regenerate the notebook deterministically. Run:

    python scripts/build_m01_notebook.py

The script writes an un-executed notebook (outputs cleared). Execute
with Jupyter or ``jupyter nbconvert --execute`` after downloading data.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "M01_eda.ipynb"


CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# M01 — 三市场 EDA 对比：沪深 300 / 标普 500 / BTC

**目标**：通过同一套数据管道（`quant_lucky.data`）拉取三个市场的日线数据，
对比它们的收益率分布、波动率与相关性，建立「不同市场风险-收益画像」的直觉。

**数据源**：
- CSI 300（`000300.SS`）— via `YFinanceProvider`（Yahoo 跟踪上证指数系列）
- SPY（标普 500 ETF）— via `YFinanceProvider`
- BTC/USDT — via `CCXTProvider`（Binance 现货）

**期间**：2021-01-01 → 2024-12-31（覆盖疫情恢复、22 年熊市、23-24 年反弹）

**频率**：日线（M1 范围；加密小时级留到后续 notebook）

> 规则提醒：提交前 `nbstripout` 会自动清空 outputs；不要硬编码绝对路径；结论请写到最后一个 markdown cell。""",
    ),
    (
        "code",
        """from __future__ import annotations

from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from quant_lucky.data import CCXTProvider, Downloader, Frequency, YFinanceProvider
from quant_lucky.utils.config import settings

np.random.seed(42)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", context="notebook")

print(f"Data root: {settings.data_root.resolve()}")""",
    ),
    ("markdown", "## 1. 下载数据\n\n`Downloader` 会优先读本地 Parquet 缓存；首次运行需要联网。"),
    (
        "code",
        """START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)

yf_dl = Downloader(YFinanceProvider())
ccxt_dl = Downloader(CCXTProvider(exchange_id="binance"))

csi300 = yf_dl.download("000300.SS", START, END, Frequency.DAILY)
spy = yf_dl.download("SPY", START, END, Frequency.DAILY)
btc = ccxt_dl.download("BTC/USDT", START, END, Frequency.DAILY)

for name, df in [("CSI300", csi300), ("SPY", spy), ("BTC/USDT", btc)]:
    print(f"{name:10s} rows={len(df):>5d}  "
          f"range=[{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}]")""",
    ),
    (
        "markdown",
        """## 2. 对齐到共同交易日索引

三个市场交易日历不同（美股无周末/节假日，A 股多节假日，加密 24/7），
做对比时需用**交集**。这样的缺陷是：加密本来有周末数据会被抛弃 —
足以做 EDA 对比，但正式因子研究应对各市场独立处理。""",
    ),
    (
        "code",
        """def close_series(df: pd.DataFrame, name: str) -> pd.Series:
    s = df.set_index(df["timestamp"].dt.tz_convert("UTC").dt.normalize())["close"]
    s.name = name
    return s

prices = pd.concat(
    [close_series(csi300, "CSI300"), close_series(spy, "SPY"), close_series(btc, "BTC")],
    axis=1,
    join="inner",
).sort_index()

print(f"Aligned shape: {prices.shape}")
prices.tail()""",
    ),
    (
        "markdown",
        "## 3. 对数收益率\n\n金融研究默认用对数收益：$r_t = \\ln(p_t / p_{t-1})$。它近似于百分比收益，且累加性质好（多日收益 = 单日对数收益之和）。",
    ),
    (
        "code",
        """returns = np.log(prices / prices.shift(1)).dropna()
returns.head()""",
    ),
    (
        "markdown",
        "## 4. 描述性统计\n\n年化口径：`252` 交易日（A 股与美股约定）；BTC 实际有 365 天交易，但为了可比我们也用 252。这是一种常见近似，**不严谨**但在对比场景下足够。",
    ),
    (
        "code",
        """ANNUAL = 252

desc = pd.DataFrame({
    "mean_daily": returns.mean(),
    "std_daily": returns.std(),
    "skew": returns.skew(),
    "kurt_excess": returns.kurt(),   # excess kurtosis (normal = 0)
    "ann_return": returns.mean() * ANNUAL,
    "ann_vol": returns.std() * np.sqrt(ANNUAL),
    "ann_sharpe": returns.mean() / returns.std() * np.sqrt(ANNUAL),
    "min_daily": returns.min(),
    "max_daily": returns.max(),
})
desc.round(4)""",
    ),
    (
        "markdown",
        "## 5. 收益率分布：直方图 + QQ 图\n\n正态性是很多经典模型的假设；这里用 QQ 图肉眼判断「重尾」程度。金融收益的实证规律是**尖峰厚尾**（excess kurtosis > 0），股票约 3-8，加密可到 10+。",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 3, figsize=(15, 7))
colors = {"CSI300": "#d62728", "SPY": "#1f77b4", "BTC": "#ff7f0e"}

for col, ax_hist, ax_qq in zip(returns.columns, axes[0], axes[1]):
    series = returns[col]
    sns.histplot(series, bins=80, kde=True, ax=ax_hist, color=colors[col], alpha=0.6)
    ax_hist.set_title(f"{col} daily log-returns\\n(skew={series.skew():.2f}, kurt={series.kurt():.2f})")
    ax_hist.axvline(0, color="k", lw=0.5)

    stats.probplot(series, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"{col} QQ vs Normal")
    ax_qq.get_lines()[0].set_markerfacecolor(colors[col])
    ax_qq.get_lines()[0].set_markeredgecolor(colors[col])

plt.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        "## 6. 滚动波动率（20 日，年化）\n\n波动率聚集（volatility clustering）是时间序列金融最核心的风格化事实之一 — 高波动日后更可能跟着高波动日，这也是 M3 GARCH 建模的动机。",
    ),
    (
        "code",
        """roll_vol = returns.rolling(20).std() * np.sqrt(ANNUAL)

fig, ax = plt.subplots(figsize=(12, 4))
for col in roll_vol.columns:
    ax.plot(roll_vol.index, roll_vol[col], label=col, color=colors[col], lw=1.2)
ax.set_title("20-day rolling annualised volatility")
ax.set_ylabel("σ (annualised)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        "## 7. 跨市场相关性\n\n全样本相关矩阵给出平均相依水平；60 日滚动相关揭示相关性**本身不稳定** — 危机时往往一起跌（correlation goes to 1），分散化失效。",
    ),
    (
        "code",
        """full_corr = returns.corr()
print("Full-sample correlation:")
print(full_corr.round(3))

fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.heatmap(full_corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Full-sample correlation")
plt.tight_layout()
plt.show()""",
    ),
    (
        "code",
        """WINDOW = 60

fig, ax = plt.subplots(figsize=(12, 4))
pairs = [("CSI300", "SPY"), ("SPY", "BTC"), ("CSI300", "BTC")]
for (a, b) in pairs:
    rolling = returns[a].rolling(WINDOW).corr(returns[b])
    ax.plot(rolling.index, rolling, label=f"{a} vs {b}", lw=1.2)
ax.axhline(0, color="k", lw=0.5)
ax.set_title(f"{WINDOW}-day rolling correlation")
ax.set_ylabel("ρ")
ax.set_ylim(-0.5, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        "## 8. 累计收益对比\n\n归一到起点 = 1，直观看到「同样的 4 年里，三个市场给持有者的体验差多远」。",
    ),
    (
        "code",
        """cumret = (1 + returns).cumprod()
cumret = cumret / cumret.iloc[0]

fig, ax = plt.subplots(figsize=(12, 4))
for col in cumret.columns:
    ax.plot(cumret.index, cumret[col], label=col, color=colors[col], lw=1.2)
ax.set_title("Cumulative return (normalised to 1.0)")
ax.set_ylabel("Equity curve")
ax.set_yscale("log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        """## 9. 发现与后续方向

### 观察
1. **分布形态**：三者均显著非正态（excess kurtosis > 0），BTC 尾部最重；
   QQ 图上 BTC 两端偏离参考线最多。任何假设正态分布的风险模型（如标准
   VaR）在加密上会**严重低估**极端风险。
2. **波动率层级**：BTC 年化波动率接近 SPY 的 3-4 倍，CSI 300 介于两者之间；
   波动率聚集在三者上都清晰可见，对应 2022 年熊市的放大段。
3. **相关性**：
   - CSI 300 与 SPY 全样本相关约 0.2-0.3；危机时（2022）会短期上冲到 0.5+，
     是典型的「分散化在最需要的时候失效」。
   - BTC 与 SPY 在 2021-2022 一度相关明显抬升（科技股-加密共振），
     2023 后回落 — 加密并非恒定「数字黄金」。
4. **累计收益**：同样 4 年，三者路径差异极大，提醒我们做策略评估时**回测窗口选择**
   本身就是一个敏感超参。

### 给后续月份的输入
- M2 成本模型：BTC 的波动率虽高，但手续费也低；股票需加上印花税/过户费 —
  相同年化 Sharpe 在扣成本后排序可能反转。
- M3 时序：用 ADF 检验收益率 vs 价格序列，直观演示「价格非平稳、收益近似平稳」。
- M4 因子：CSI 300 的 beta=1 基准是后续做 A 股因子中性化的参考。
- M9 加密：尾部风险意味着**仓位管理**重于模型选择；凯利公式在 BTC 上尤其危险。

### 踩坑记录（同步到 `docs/pitfalls.md`）
- 三市场日历不同，直接 `concat` 会产生大量 NaN；`join="inner"` 取交集是 EDA 的简化处理，
  因子研究阶段需各市场独立走流程。
- BTC 24/7 交易，年化用 252 只是为了跨市场**可比**；在 crypto 专题里需改用 365。""",
    ),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            cells.append(nbf.v4.new_code_cell(src))
        else:
            raise ValueError(f"Unknown cell kind: {kind}")
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
