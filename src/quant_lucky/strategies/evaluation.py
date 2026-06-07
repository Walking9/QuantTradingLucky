"""Strategy research harness — the shared evaluation layer behind the
runnable ``strategies/<name>/`` packages.

M6 asks every classic-strategy reproduction to ship the *same* honesty
bundle (the standard the M5 review wrote down): an IS/OOS comparison, a
Walk-Forward + Deflated-Sharpe-Ratio anti-overfit number, a parameter-
sensitivity sweep, and a return attribution against a market benchmark.
The Walk-Forward + DSR machinery already lives in
:mod:`quant_lucky.backtest.validation`; this module supplies the three
pieces that are *not* there yet and would otherwise be copy-pasted into
each package's ``run_backtest.py``:

* :func:`attribution` — single-factor (market) OLS: how much of the
  return is benchmark beta vs residual alpha.
* :func:`is_oos_split` — score one *fixed* no-look-ahead weight book on
  an in-sample and an out-of-sample era. This answers "did the edge
  survive into the later period?", a different question from
  Walk-Forward's "did *re-tuned* parameters generalise?".
* :func:`parameter_sensitivity` — re-run a strategy across a grid of
  parameter perturbations (the M5 review's "±50% rerun") and tabulate
  how IS/OOS metrics move. A strategy whose Sharpe collapses when a
  window changes by one notch is overfit to that window.

:func:`evaluate_strategy` ties the full-sample backtest, the IS/OOS
split, the benchmark, and the attribution into one
:class:`StrategyEvaluation` bundle with ``metrics_dict`` (JSON) and
``to_markdown`` (a generated results block) so the three packages stay
thin and consistent.

Cost convention
---------------
Every metric here is computed with the **same** ``cost_bps`` so full /
IS / OOS / sensitivity / Walk-Forward numbers are directly comparable
(Walk-Forward only accepts a scalar ``cost_bps``, so we standardise on
it everywhere rather than mixing in a per-fill ``CostModel``). ``cost_bps``
is per-side, charged on ``Σ|Δw|`` — see :class:`~quant_lucky.backtest.VectorEngine`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lucky.backtest.report import PerformanceReport, compute_performance
from quant_lucky.backtest.validation import (
    Split,
    StrategyFn,
    WalkForwardResult,
    fixed_split,
    rolling_walk_forward,
    walk_forward,
)
from quant_lucky.backtest.vector import BacktestResult, VectorEngine

__all__ = [
    "FactorAttribution",
    "ResearchSpec",
    "SplitEvaluation",
    "StrategyEvaluation",
    "attribution",
    "evaluate_strategy",
    "is_oos_split",
    "parameter_sensitivity",
    "run_research",
    "write_artifacts",
]


# ---------------------------------------------------------------------------
# Single-factor (market) attribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorAttribution:
    """Decomposition of a return stream against one benchmark factor.

    A single-factor OLS ``r_t = α + β · b_t + ε_t`` where ``b`` is the
    market/benchmark return. ``alpha_annual`` is the intercept scaled to
    a year; ``beta`` is the market loading; ``r_squared`` is the share of
    variance the benchmark explains. For a *market-neutral* book (e.g. a
    dollar-neutral long-short) ``beta`` should sit near zero — that *is*
    the attribution story, not a bug.
    """

    beta: float
    alpha_annual: float
    correlation: float
    r_squared: float
    n_obs: int


def attribution(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> FactorAttribution:
    """Single-factor OLS of ``returns`` on ``benchmark_returns``.

    The two series are inner-joined on their index and NaNs dropped
    before the regression, so misaligned calendars degrade gracefully
    rather than throwing.

    Args:
        returns: Strategy per-period (simple) returns.
        benchmark_returns: Market/benchmark per-period returns.
        periods_per_year: Annualisation factor for the alpha intercept.

    Returns:
        A :class:`FactorAttribution`. All scalars are ``nan`` if fewer
        than 3 overlapping observations or a degenerate (zero-variance)
        benchmark make the regression undefined.
    """
    df = pd.concat({"r": returns, "b": benchmark_returns}, axis=1).dropna()
    n = len(df)
    if n < 3:
        return FactorAttribution(float("nan"), float("nan"), float("nan"), float("nan"), n)

    r = df["r"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)

    b_var = float(b.var(ddof=1))
    if b_var <= 0.0:
        return FactorAttribution(float("nan"), float("nan"), float("nan"), float("nan"), n)

    cov = float(np.cov(r, b, ddof=1)[0, 1])
    beta = cov / b_var
    alpha_pp = float(r.mean()) - beta * float(b.mean())
    alpha_annual = alpha_pp * periods_per_year

    r_std = float(r.std(ddof=1))
    corr = float("nan") if r_std <= 0.0 else float(np.corrcoef(r, b)[0, 1])
    r_squared = corr**2 if np.isfinite(corr) else float("nan")

    return FactorAttribution(
        beta=beta,
        alpha_annual=alpha_annual,
        correlation=corr,
        r_squared=r_squared,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Fixed IS / OOS split of one weight book
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitEvaluation:
    """In-sample vs out-of-sample performance of one fixed weight book."""

    split: Split
    is_report: PerformanceReport
    oos_report: PerformanceReport


def is_oos_split(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float,
    periods_per_year: int = 252,
    oos_fraction: float = 0.3,
) -> SplitEvaluation:
    """Score one *fixed* weight book on an IS and an OOS era.

    The weights are assumed no-look-ahead (the strategy primitives in
    this package guarantee that). We run the engine **once** over the
    full panel, then slice the resulting net-return series at
    ``1 - oos_fraction`` and recompute the report on each side. Slicing
    the already-costed series — rather than re-running the engine on the
    OOS price window — is deliberate: it avoids charging a spurious entry
    cost at the OOS boundary (the position carried over from the IS era
    is real, not a fresh buy).

    Args:
        weights: ``date × asset`` weight book.
        prices: ``date × asset`` close prices.
        cost_bps: Per-side cost in basis points.
        periods_per_year: Annualisation factor.
        oos_fraction: Fraction of the timeline reserved for OOS.

    Returns:
        A :class:`SplitEvaluation` with the IS and OOS reports.
    """
    engine = VectorEngine(cost_bps=cost_bps, periods_per_year=periods_per_year)
    full = engine.run(weights, prices)
    idx = pd.DatetimeIndex(full.net_returns.index)
    split = fixed_split(idx, oos_fraction=oos_fraction)

    is_ret = full.net_returns.loc[split.train_start : split.train_end]
    oos_ret = full.net_returns.loc[split.test_start : split.test_end]
    is_report = compute_performance(is_ret, periods_per_year=periods_per_year)
    oos_report = compute_performance(oos_ret, periods_per_year=periods_per_year)
    return SplitEvaluation(split=split, is_report=is_report, oos_report=oos_report)


# ---------------------------------------------------------------------------
# Parameter-sensitivity sweep
# ---------------------------------------------------------------------------
def parameter_sensitivity(
    strategy_fn: StrategyFn,
    prices: pd.DataFrame,
    *,
    base_param: Mapping[str, Any],
    sweep: Mapping[str, Sequence[Any]],
    cost_bps: float,
    periods_per_year: int = 252,
    oos_fraction: float = 0.3,
) -> pd.DataFrame:
    """One-at-a-time parameter perturbation around a base configuration.

    For each ``(name, values)`` in ``sweep`` and each value, the base
    parameters are copied with that one key overridden, the strategy is
    re-run over the full panel, and full-sample plus OOS metrics are
    recorded. This is the M6 robustness check: a healthy strategy moves
    *smoothly* as a parameter changes and keeps a positive OOS Sharpe
    across the neighbourhood; a knife-edge peak is the fingerprint of an
    overfit window.

    Args:
        strategy_fn: ``(param_dict, prices) -> weights`` (no-look-ahead).
        prices: Full ``date × asset`` close panel.
        base_param: The base configuration to perturb from.
        sweep: Mapping of parameter name → list of values to try.
        cost_bps: Per-side cost in basis points.
        periods_per_year: Annualisation factor.
        oos_fraction: OOS fraction forwarded to :func:`is_oos_split`.

    Returns:
        A tidy DataFrame, one row per ``(param, value)`` with columns:
        ``param, value, is_base, full_sharpe, full_annual_return,
        full_max_drawdown, is_sharpe, oos_sharpe, oos_annual_return,
        oos_max_drawdown, is_minus_oos_sharpe``.
    """
    engine = VectorEngine(cost_bps=cost_bps, periods_per_year=periods_per_year)
    rows: list[dict[str, Any]] = []
    for pname, values in sweep.items():
        for value in values:
            param = dict(base_param)
            is_base = param.get(pname) == value
            param[pname] = value
            weights = strategy_fn(param, prices)
            full = engine.run(weights, prices)
            split_eval = is_oos_split(
                weights,
                prices,
                cost_bps=cost_bps,
                periods_per_year=periods_per_year,
                oos_fraction=oos_fraction,
            )
            is_sr = split_eval.is_report.sharpe
            oos_sr = split_eval.oos_report.sharpe
            rows.append(
                {
                    "param": pname,
                    "value": value,
                    "is_base": is_base,
                    "full_sharpe": full.report.sharpe,
                    "full_annual_return": full.report.annual_return,
                    "full_max_drawdown": full.report.max_drawdown,
                    "is_sharpe": is_sr,
                    "oos_sharpe": oos_sr,
                    "oos_annual_return": split_eval.oos_report.annual_return,
                    "oos_max_drawdown": split_eval.oos_report.max_drawdown,
                    "is_minus_oos_sharpe": is_sr - oos_sr,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full strategy evaluation bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyEvaluation:
    """Everything one strategy run produces, in one serialisable bundle.

    ``walk_forward`` and ``sensitivity`` are optional and filled by the
    caller (they need the strategy's parameter grid, which lives in the
    package, not here). The rest is computed by :func:`evaluate_strategy`.
    """

    name: str
    data_source: str
    span: tuple[pd.Timestamp, pd.Timestamp]
    n_assets: int
    cost_bps: float
    periods_per_year: int
    full: BacktestResult
    split: SplitEvaluation
    benchmark: BacktestResult | None = None
    attribution: FactorAttribution | None = None
    walk_forward: WalkForwardResult | None = None
    sensitivity: pd.DataFrame | None = None

    # -- serialisation ----------------------------------------------------
    def metrics_dict(self) -> dict[str, Any]:
        """A flat, JSON-serialisable summary of the headline numbers."""
        rep = self.full.report
        out: dict[str, Any] = {
            "name": self.name,
            "data_source": self.data_source,
            "span_start": str(self.span[0].date()),
            "span_end": str(self.span[1].date()),
            "n_assets": self.n_assets,
            "cost_bps": self.cost_bps,
            "periods_per_year": self.periods_per_year,
            "full": _report_metrics(rep),
            "in_sample": _report_metrics(self.split.is_report),
            "out_of_sample": _report_metrics(self.split.oos_report),
            "split": {
                "train_start": str(self.split.split.train_start.date()),
                "train_end": str(self.split.split.train_end.date()),
                "test_start": str(self.split.split.test_start.date()),
                "test_end": str(self.split.split.test_end.date()),
            },
        }
        if self.benchmark is not None:
            out["benchmark"] = _report_metrics(self.benchmark.report)
        if self.attribution is not None:
            out["attribution"] = {
                "beta": _f(self.attribution.beta),
                "alpha_annual": _f(self.attribution.alpha_annual),
                "correlation": _f(self.attribution.correlation),
                "r_squared": _f(self.attribution.r_squared),
                "n_obs": self.attribution.n_obs,
            }
        if self.walk_forward is not None:
            wf = self.walk_forward
            out["walk_forward"] = {
                "n_splits": len(wf.splits),
                "n_trials": wf.n_trials,
                "aggregate_oos_sharpe": _f(wf.extra.get("aggregate_oos_sharpe", float("nan"))),
                "deflated_sharpe": _f(wf.deflated_sharpe),
                "mean_is_sharpe": _f(float(np.nanmean(wf.is_metrics)) if wf.is_metrics else np.nan),
                "mean_oos_sharpe": _f(
                    float(np.nanmean(wf.oos_metrics)) if wf.oos_metrics else np.nan
                ),
            }
        return out

    # -- rendering --------------------------------------------------------
    def to_markdown(self) -> str:
        """A machine-generated results block for ``reports/RESULTS.md``."""
        lines: list[str] = []
        lines.append(f"# {self.name} — generated results")
        lines.append("")
        lines.append(
            f"- Data source: **{self.data_source}** "
            f"({self.n_assets} assets, {self.span[0].date()} → {self.span[1].date()})"
        )
        lines.append(
            f"- Cost: **{self.cost_bps:.1f} bps/side**, annualisation **{self.periods_per_year}**"
        )
        lines.append("")
        lines.append("## Headline metrics (net of cost)")
        lines.append("")
        cols = ["window", "ann_return", "ann_vol", "sharpe", "max_dd", "hit_rate"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        lines.append(_md_row("full", self.full.report))
        lines.append(_md_row("in-sample", self.split.is_report))
        lines.append(_md_row("out-of-sample", self.split.oos_report))
        if self.benchmark is not None:
            lines.append(_md_row("benchmark", self.benchmark.report))
        ta = self.full.report.turnover_annual
        if ta is not None:
            lines.append("")
            lines.append(f"- Annualised one-way turnover: **{ta:.2f}**")

        if self.attribution is not None:
            a = self.attribution
            lines.append("")
            lines.append("## Attribution vs benchmark (single-factor OLS)")
            lines.append("")
            lines.append(f"- Beta: **{a.beta:.3f}**")
            lines.append(f"- Annualised alpha: **{a.alpha_annual:.2%}**")
            lines.append(f"- Correlation: **{a.correlation:.3f}** (R² {a.r_squared:.3f})")

        if self.walk_forward is not None:
            wf = self.walk_forward
            lines.append("")
            lines.append("## Walk-Forward (parameter generalisation)")
            lines.append("")
            lines.append(f"- Splits: **{len(wf.splits)}**, trials: **{wf.n_trials}**")
            lines.append(
                f"- Aggregate OOS Sharpe: "
                f"**{wf.extra.get('aggregate_oos_sharpe', float('nan')):.2f}**"
            )
            lines.append(
                f"- Deflated Sharpe Ratio: **{wf.deflated_sharpe:.3f}** "
                f"(P[true Sharpe > 0] after {wf.n_trials} trials)"
            )

        if self.sensitivity is not None and not self.sensitivity.empty:
            lines.append("")
            lines.append("## Parameter sensitivity (OOS Sharpe)")
            lines.append("")
            lines.append("| param | value | full_sharpe | oos_sharpe | is−oos |")
            lines.append("|---|---|---|---|---|")
            for _, r in self.sensitivity.iterrows():
                star = " *" if r["is_base"] else ""
                lines.append(
                    f"| {r['param']}{star} | {r['value']} | "
                    f"{r['full_sharpe']:.2f} | {r['oos_sharpe']:.2f} | "
                    f"{r['is_minus_oos_sharpe']:.2f} |"
                )
            lines.append("")
            lines.append("`*` = base configuration.")
        lines.append("")
        return "\n".join(lines)


def evaluate_strategy(
    *,
    name: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float,
    periods_per_year: int = 252,
    oos_fraction: float = 0.3,
    data_source: str = "real",
    benchmark_weights: pd.DataFrame | None = None,
    walk_forward_result: WalkForwardResult | None = None,
    sensitivity: pd.DataFrame | None = None,
) -> StrategyEvaluation:
    """Run the full-sample backtest, IS/OOS split, benchmark, attribution.

    Walk-Forward and sensitivity are not computed here (they need the
    strategy's parameter grid) — pass them in if you have them and they
    will be carried on the bundle for reporting.

    Args:
        name: Strategy identifier (used in headings / metrics).
        prices: ``date × asset`` close panel.
        weights: ``date × asset`` strategy weight book.
        cost_bps: Per-side cost in basis points.
        periods_per_year: Annualisation factor.
        oos_fraction: OOS fraction for the fixed split.
        data_source: ``"real"`` or ``"synthetic"`` (carried for honesty).
        benchmark_weights: Optional benchmark weight book (e.g. equal
            weight) run through the same engine; enables attribution.
        walk_forward_result: Optional pre-computed Walk-Forward result.
        sensitivity: Optional pre-computed sensitivity table.

    Returns:
        A populated :class:`StrategyEvaluation`.
    """
    engine = VectorEngine(cost_bps=cost_bps, periods_per_year=periods_per_year)
    full = engine.run(weights, prices)
    split = is_oos_split(
        weights,
        prices,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        oos_fraction=oos_fraction,
    )

    benchmark: BacktestResult | None = None
    attr: FactorAttribution | None = None
    if benchmark_weights is not None:
        benchmark = engine.run(benchmark_weights, prices)
        attr = attribution(
            full.net_returns,
            benchmark.net_returns,
            periods_per_year=periods_per_year,
        )

    span = (pd.Timestamp(full.net_returns.index[0]), pd.Timestamp(full.net_returns.index[-1]))
    return StrategyEvaluation(
        name=name,
        data_source=data_source,
        span=span,
        n_assets=int(prices.shape[1]),
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        full=full,
        split=split,
        benchmark=benchmark,
        attribution=attr,
        walk_forward=walk_forward_result,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _f(x: float) -> float:
    """Coerce to a plain float (so numpy scalars serialise to JSON)."""
    return float(x)


def _report_metrics(rep: PerformanceReport) -> dict[str, float]:
    return {
        "annual_return": _f(rep.annual_return),
        "annual_vol": _f(rep.annual_vol),
        "sharpe": _f(rep.sharpe),
        "sortino": _f(rep.sortino),
        "calmar": _f(rep.calmar),
        "max_drawdown": _f(rep.max_drawdown),
        "hit_rate": _f(rep.hit_rate),
        "n_periods": int(rep.n_periods),
    }


def _md_row(label: str, rep: PerformanceReport) -> str:
    return (
        f"| {label} | {rep.annual_return:.2%} | {rep.annual_vol:.2%} | "
        f"{rep.sharpe:.2f} | {rep.max_drawdown:.2%} | {rep.hit_rate:.2%} |"
    )


# ---------------------------------------------------------------------------
# One-call research driver (keeps each package's run_backtest.py trivial)
# ---------------------------------------------------------------------------
@dataclass
class ResearchSpec:
    """Declarative description of one strategy research run.

    A package's ``strategy.py`` builds this from its config and hands it
    to :func:`run_research`, which assembles the full
    :class:`StrategyEvaluation` (full-sample backtest + IS/OOS split +
    benchmark attribution + Walk-Forward/DSR + parameter sensitivity).
    Keeping the orchestration here means the three ``run_backtest.py``
    entry points stay ~15 lines each and share one tested code path.

    The Walk-Forward and sensitivity blocks are computed only when the
    fields they need are present; an over-short panel that yields zero
    Walk-Forward splits degrades to "no Walk-Forward" rather than raising.
    """

    name: str
    prices: pd.DataFrame
    weights: pd.DataFrame
    cost_bps: float
    periods_per_year: int = 252
    oos_fraction: float = 0.3
    data_source: str = "real"
    benchmark_weights: pd.DataFrame | None = None
    # Walk-Forward (all four required to run it).
    strategy_fn: StrategyFn | None = None
    param_grid: Sequence[Mapping[str, Any]] | None = None
    wf_train_size: int | None = None
    wf_test_size: int | None = None
    # Parameter sensitivity (both required to run it; reuses strategy_fn).
    base_param: Mapping[str, Any] | None = None
    sensitivity_sweep: Mapping[str, Sequence[Any]] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def run_research(spec: ResearchSpec) -> StrategyEvaluation:
    """Execute a :class:`ResearchSpec` into a full :class:`StrategyEvaluation`."""
    wf: WalkForwardResult | None = None
    if (
        spec.strategy_fn is not None
        and spec.param_grid is not None
        and spec.wf_train_size is not None
        and spec.wf_test_size is not None
    ):
        splits = rolling_walk_forward(
            pd.DatetimeIndex(spec.prices.index),
            train_size=spec.wf_train_size,
            test_size=spec.wf_test_size,
        )
        if splits:
            wf = walk_forward(
                prices=spec.prices,
                splits=splits,
                param_grid=spec.param_grid,
                strategy_fn=spec.strategy_fn,
                cost_bps=spec.cost_bps,
                periods_per_year=spec.periods_per_year,
            )

    sensitivity: pd.DataFrame | None = None
    if (
        spec.strategy_fn is not None
        and spec.base_param is not None
        and spec.sensitivity_sweep is not None
    ):
        sensitivity = parameter_sensitivity(
            spec.strategy_fn,
            spec.prices,
            base_param=spec.base_param,
            sweep=spec.sensitivity_sweep,
            cost_bps=spec.cost_bps,
            periods_per_year=spec.periods_per_year,
            oos_fraction=spec.oos_fraction,
        )

    return evaluate_strategy(
        name=spec.name,
        prices=spec.prices,
        weights=spec.weights,
        cost_bps=spec.cost_bps,
        periods_per_year=spec.periods_per_year,
        oos_fraction=spec.oos_fraction,
        data_source=spec.data_source,
        benchmark_weights=spec.benchmark_weights,
        walk_forward_result=wf,
        sensitivity=sensitivity,
    )


def write_artifacts(evaluation: StrategyEvaluation, out_dir: Path) -> list[Path]:
    """Persist a strategy run's reproducible artifacts under ``out_dir``.

    Writes ``metrics.json`` (the headline numbers), ``RESULTS.md`` (the
    generated results block), ``equity_curve.csv`` (net portfolio value
    of the strategy and, if present, the benchmark) and — when computed —
    ``sensitivity.csv``. Returns the list of paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation.metrics_dict(), indent=2), encoding="utf-8")
    written.append(metrics_path)

    results_path = out_dir / "RESULTS.md"
    results_path.write_text(evaluation.to_markdown(), encoding="utf-8")
    written.append(results_path)

    curves = {"strategy_net": evaluation.full.portfolio_value}
    if evaluation.benchmark is not None:
        curves["benchmark_net"] = evaluation.benchmark.portfolio_value
    curve_path = out_dir / "equity_curve.csv"
    pd.DataFrame(curves).to_csv(curve_path)
    written.append(curve_path)

    if evaluation.sensitivity is not None and not evaluation.sensitivity.empty:
        sens_path = out_dir / "sensitivity.csv"
        evaluation.sensitivity.to_csv(sens_path, index=False)
        written.append(sens_path)

    return written
