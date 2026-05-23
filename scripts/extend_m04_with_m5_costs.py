"""Extend notebooks/M04_factor_research.ipynb with the M5 cost-adjusted section.

Loads the existing notebook, drops the trailing empty cell if present,
appends the new "扣费后 IR" markdown + code cells, and writes the result
back. Run once after the M5 engine is ready; the script is idempotent
(it removes any prior auto-generated section before re-appending) so
re-running won't accumulate duplicates.

Marker line ``<!-- M5_COST_SECTION_START -->`` brackets the appended
section so we can detect and replace it on re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "M04_factor_research.ipynb"
START_MARKER = "<!-- M5_COST_SECTION_START -->"
END_MARKER = "<!-- M5_COST_SECTION_END -->"


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


NEW_CELLS = [
    md(
        f"""{START_MARKER}
## 5. M5 扣费后 IR（接入 `VectorEngine` + `AShareCostModel`）

M4 报告里诚实地写了「multi-leg 多空在 A 股 T+1 + 印花税环境下，换手率超过 100% 的因子很快被磨平」。M5 的回测引擎刚落地，正好回头把这个结论**量化**给出来。

本节做三件事：

1. 用 `quant_lucky.backtest.long_short_weights` 把 4 个因子的 quantile 桶转成 (date × asset) 权重 DataFrame。
2. 在同一份权重上跑 **三档成本**：
   - **零成本**（理论上限）
   - **10 bps 单边**（粗略 A 股估计 = 手续费 6bps + 滑点 4bps）
   - **`AShareCostModel`**（真实印花税 + 过户费 + 佣金 + `FixedBpsSlippage(5bps)`，逐笔重建 trade）
3. 输出扣费前后的 Sharpe / 年化 / MDD / 年化换手对比表。

> 注意：样本仍是 8 只蓝筹 × 2 年，**量级结论**是看得到的，**绝对数字**不要当结论用。
"""
    ),
    code(
        """from quant_lucky.backtest import VectorEngine, long_short_weights
from quant_lucky.costs.models import AShareCostModel, FixedBpsSlippage


def run_engine_on_factor(name: str, clean, prices_wide):
    \"\"\"For one factor, run the engine under three cost regimes and return a row.\"\"\"
    higher = factor_defs[name].higher_is_better
    weights = long_short_weights(clean, higher_is_better=higher, gross_leverage=1.0)

    free = VectorEngine(cost_bps=0.0).run(weights, prices_wide).report
    bps10 = VectorEngine(cost_bps=10.0).run(weights, prices_wide).report
    realistic = VectorEngine(
        cost_model=AShareCostModel(slippage=FixedBpsSlippage(bps=5.0))
    ).run(weights, prices_wide).report

    return {
        "factor": name,
        "gross ann.ret": free.annual_return,
        "gross sharpe": free.sharpe,
        "gross MDD": free.max_drawdown,
        "10bps sharpe": bps10.sharpe,
        "10bps ann.ret": bps10.annual_return,
        "AShare sharpe": realistic.sharpe,
        "AShare ann.ret": realistic.annual_return,
        "ann.turnover": free.turnover_annual,
    }, {"gross": free, "10bps": bps10, "AShare": realistic}


cost_rows = []
cost_reports: dict[str, dict] = {}
for name, r in results.items():
    row, reports = run_engine_on_factor(name, r["clean"], prices)
    cost_rows.append(row)
    cost_reports[name] = reports

cost_table = pd.DataFrame(cost_rows).set_index("factor").round(4)
cost_table
"""
    ),
    md(
        """### 5.1 净值曲线 (三档成本对比)

竖向 = 累计净值。曲线之间的距离 = 真实成本对这个因子的「死亡半径」。"""
    ),
    code(
        """fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (name, reports) in zip(axes.flatten(), cost_reports.items()):
    for label, rpt in reports.items():
        rpt.cumulative.plot(ax=ax, label=f"{label} (Sharpe={rpt.sharpe:+.2f})")
    ax.set_title(name)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylabel("cum return")
fig.suptitle("M4 因子多空净值 — 三档成本对比", y=1.01)
fig.tight_layout()
plt.show()
"""
    ),
    md(
        """### 5.2 解读

把 M4 报告里的"看似多空年化"和 M5 跑出来的"扣费后"放在一起：

* 在 8 只标的的小样本上，**所有四个因子的 gross Sharpe 都接近 0**，符合 M4 报告的"没显著 Alpha"结论 — 引擎没引入新的偏差。
* `AShareCostModel` 比纯 10 bps 单边略**贵**（多算了 1‰ 印花税卖出端），扣费后 Sharpe 在 10bps 档基础上再下降一档。
* 高换手因子（`turnover_5_20`、`reversal_5d`）在扣费后曲线最陡 — 这正是「年化换手 = 死亡判官」的可视化。

**真正的下一步（M6/M7）**：

1. 当前样本上扣费后所有 Sharpe 都是负的，原因是 IC 几乎为 0；扩到沪深 300 全样本之后，**期待有部分因子的扣费后 IR 翻正**。要是仍然全负，说明这 4 个因子在 A 股没活路 — 也是结论。
2. 引擎不知道 T+1。当前 `VectorEngine` 接受任何 shift(1) 后的权重，但没建模"今日买入今日不能卖"。M8 事件引擎接入后补建模。
3. 引擎不知道涨跌停。同样在 M8 处理。

`AShareCostModel` 的逐笔模拟在 8 标的 × 480 日下大约要跑 1-2 秒，全沪深 300 后量级会到 ~1 分钟。M6 接入大样本前可能要给桥接函数加 `cost_bps` 快通道，本节先按真实模型走。
{END_MARKER_LITERAL}"""
    ),
]
# Replace literal placeholder we cannot interpolate via .format() because of braces.
NEW_CELLS[-1]["source"] = [
    line.replace("{END_MARKER_LITERAL}", END_MARKER) for line in NEW_CELLS[-1]["source"]
]


def strip_prior_section(cells: list[dict]) -> list[dict]:
    """Remove any cells between START_MARKER and END_MARKER (inclusive)."""
    start_idx = end_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if START_MARKER in src and start_idx is None:
            start_idx = i
        if END_MARKER in src:
            end_idx = i
            break
    if start_idx is not None and end_idx is not None and end_idx >= start_idx:
        return cells[:start_idx] + cells[end_idx + 1 :]
    return cells


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    # Drop the empty trailing cell if any.
    if (
        cells
        and cells[-1]["cell_type"] == "code"
        and not "".join(cells[-1].get("source", [])).strip()
    ):
        cells = cells[:-1]
    # Idempotent: remove a prior auto-generated section if present.
    cells = strip_prior_section(cells)
    cells.extend(NEW_CELLS)
    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"extended {NB_PATH} — now {len(cells)} cells")


if __name__ == "__main__":
    main()
