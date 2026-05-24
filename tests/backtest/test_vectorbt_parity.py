"""Integration test: VectorEngine parity with vectorbt.

This is an **integration test** in two senses:

* It imports a heavy reference library (``vectorbt``) that pulls in
  numba and the rest of the stack. Skipped automatically if vectorbt
  isn't installed (you'd need the ``backtest`` extra: ``uv sync --extra
  backtest``).
* It tests the *behaviour contract* between our engine and a known-good
  reference, not the internals of either. Failures here usually mean
  semantics drifted (look-ahead policy, cost convention, turnover
  accounting) — not a typo.

Why test what an external library says
--------------------------------------
Our unit tests in ``test_vector.py`` verify the engine by construction
(handcrafted expected returns). That guards correctness within OUR
worldview. The vectorbt parity test guards *against* the world-view
itself drifting: if I ever change ``weights.shift(1)`` policy without
realising, vectorbt's terminal value will diverge and this test rings
the bell.

The M5 learning plan target is ≤ 3 % deviation. We assert 1 % here to
catch drift earlier than the formal target; if our reference scenarios
genuinely require a wider tolerance, bump it explicitly with a note.
"""

from __future__ import annotations

import importlib.util

import pytest

# Skip the whole module if vectorbt isn't available. Keeps the
# fast-feedback unit suite hermetic.
vbt_spec = importlib.util.find_spec("vectorbt")
if vbt_spec is None:
    pytest.skip(
        "vectorbt not installed; install with `uv sync --extra backtest`",
        allow_module_level=True,
    )


import numpy as np  # noqa: E402
from scripts.verify_vectorbt_parity import (  # noqa: E402
    EngineMetrics,
    ScenarioResult,
    make_synthetic_prices,
    run_all,
    scenario_buy_and_hold,
    scenario_daily_long_short,
    scenario_equal_weight_rebalance,
    scenario_monthly_rebalance_drifted,
)

from quant_lucky.backtest import VectorEngine  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Top-level parity contract
# ---------------------------------------------------------------------------
def test_all_parity_scenarios_within_3pct() -> None:
    """The M5 learning target: ≤ 3 % terminal-value deviation across all
    parity scenarios. If this fails, we lost back-to-back parity with the
    industry reference — fix before merging.
    """
    parity, _ = run_all()
    assert parity, "Expected at least one parity scenario"

    deviations: list[tuple[str, float]] = []
    for r in parity:
        rel = r.deviations["terminal_value_rel"]
        # Skip NaN: would mean ours.terminal_value == 0 (no data); a
        # separate test would catch that.
        if not np.isnan(rel):
            deviations.append((r.scenario, abs(rel)))

    worst = max(deviations, key=lambda kv: kv[1])
    assert worst[1] <= 0.03, (
        f"Parity deviation {worst[1]:.4%} on scenario {worst[0]!r} exceeds 3 %. "
        f"All deviations: {deviations}"
    )


def test_drifted_buy_and_hold_is_essentially_exact() -> None:
    """Strongest sanity check: with no rebalancing (only drift), the two
    engines should agree to float precision. If this drifts, our return-
    compounding math is wrong, not just our cost or shift semantics.
    """
    prices = make_synthetic_prices()
    weights = scenario_buy_and_hold(prices)
    ours = _run_ours(weights["ours"], prices, cost_bps=0.0)
    vbt = _run_vbt(weights["vbt"], prices, cost_bps=0.0)
    rel = abs(vbt.terminal_value / ours.terminal_value - 1.0)
    assert rel < 1e-6, f"BNH parity drifted: {rel:.2e}"


def test_costfree_daily_rebalance_within_25bps_terminal() -> None:
    """Cost-free daily rebalance: any difference is float / rounding only.
    Allow 25 bps of terminal value drift — generous to insulate this test
    from harmless numerical noise.
    """
    prices = make_synthetic_prices()
    weights = scenario_equal_weight_rebalance(prices)
    ours = _run_ours(weights["ours"], prices, cost_bps=0.0)
    vbt = _run_vbt(weights["vbt"], prices, cost_bps=0.0)
    rel = abs(vbt.terminal_value / ours.terminal_value - 1.0)
    assert rel < 0.0025, f"Cost-free rebalance parity: {rel:.4%}"


def test_high_turnover_with_costs_within_1pct() -> None:
    """Daily long-short + 10bps is the hardest scenario. The cost-formula
    NAV-floating-vs-fixed gap shows here. Still under 1 % terminal.
    """
    prices = make_synthetic_prices()
    weights = scenario_daily_long_short(prices)
    ours = _run_ours(weights["ours"], prices, cost_bps=10.0)
    vbt = _run_vbt(weights["vbt"], prices, cost_bps=10.0)
    rel = abs(vbt.terminal_value / ours.terminal_value - 1.0)
    assert rel < 0.01, f"High-turnover parity: {rel:.4%}"


def test_monthly_rebalance_drifted_within_50bps() -> None:
    """Monthly rebalance with drifted weights between rebalances. Both
    engines model the same physical strategy; difference is rounding.
    """
    prices = make_synthetic_prices()
    weights = scenario_monthly_rebalance_drifted(prices)
    ours = _run_ours(weights["ours"], prices, cost_bps=10.0)
    vbt = _run_vbt(weights["vbt"], prices, cost_bps=10.0)
    rel = abs(vbt.terminal_value / ours.terminal_value - 1.0)
    assert rel < 0.005, f"Monthly-rebalance parity: {rel:.4%}"


# ---------------------------------------------------------------------------
# Sharpe & MDD sanity (catch sign / annualisation drift)
# ---------------------------------------------------------------------------
def test_sharpe_signs_agree_across_scenarios() -> None:
    """Looser metric-level check: Sharpe sign must agree on every
    scenario (catches accidental sign-flipping in cost or return).
    """
    parity, _ = run_all()
    for r in parity:
        if np.isnan(r.ours.sharpe) or np.isnan(r.vbt.sharpe):
            continue
        # Allow tiny noise around zero — only complain on opposite signs
        # when both have magnitude > 0.05.
        if abs(r.ours.sharpe) > 0.05 and abs(r.vbt.sharpe) > 0.05:
            assert (r.ours.sharpe > 0) == (r.vbt.sharpe > 0), (
                f"Sharpe signs disagree on {r.scenario}: "
                f"ours={r.ours.sharpe:+.3f} vs vbt={r.vbt.sharpe:+.3f}"
            )


def test_mdd_within_30bps_absolute_across_scenarios() -> None:
    """Max drawdown should match closely too. We use absolute delta
    because MDD can be near zero and relative blows up.
    """
    parity, _ = run_all()
    for r in parity:
        delta = abs(r.deviations["max_drawdown_abs"])
        assert delta < 0.003, (
            f"MDD deviates {delta:.4%} on {r.scenario}; "
            f"ours={r.ours.max_drawdown:+.4%}, vbt={r.vbt.max_drawdown:+.4%}"
        )


# ---------------------------------------------------------------------------
# Internals — re-import the helpers without importing the whole script.
# We don't reuse the script's _our_engine/_vbt_engine to avoid coupling
# the tests to its internal naming, but we DO use its scenarios so the
# CI path is exactly what the report shows.
# ---------------------------------------------------------------------------
def _run_ours(weights, prices, cost_bps: float) -> EngineMetrics:
    res = VectorEngine(cost_bps=cost_bps).run(weights, prices)
    return EngineMetrics(
        name="VectorEngine",
        terminal_value=float(res.portfolio_value.iloc[-1]),
        annual_return=res.report.annual_return,
        annual_vol=res.report.annual_vol,
        sharpe=res.report.sharpe,
        max_drawdown=res.report.max_drawdown,
    )


def _run_vbt(weights, prices, cost_bps: float) -> EngineMetrics:
    import vectorbt as vbt  # local import: only when test actually runs

    init_cash = 1_000_000.0
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        fees=cost_bps / 10_000.0,
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        freq="D",
    )
    value = pf.value()
    rets = pf.returns()
    norm_value = value / init_cash
    n = len(rets)
    if n > 1:
        mean = float(rets.mean())
        std = float(rets.std(ddof=1))
        ann_ret = mean * 252.0
        ann_vol = std * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    else:
        ann_ret = float("nan")
        ann_vol = float("nan")
        sharpe = float("nan")
    running_max = norm_value.cummax()
    dd = float((norm_value / running_max - 1.0).min())
    return EngineMetrics(
        name="vectorbt",
        terminal_value=float(norm_value.iloc[-1]),
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=dd,
    )


# Suppress unused-import flake (used only to check it imports cleanly).
_ = ScenarioResult
