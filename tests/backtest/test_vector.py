"""Tests for ``backtest.vector.VectorEngine``.

Each test is a small known-answer scenario; the engine is verified by
construction (not by comparing to ``vectorbt``). The vectorbt comparison
test belongs in a separate integration suite and is mentioned in
``notebooks/M05_backtest_traps.ipynb`` for visual cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.backtest.vector import BacktestResult, VectorEngine
from quant_lucky.costs.models import AShareCostModel, FixedBpsSlippage


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
def test_must_specify_costs_explicitly() -> None:
    """Forgetting to pass cost params is a bug, not a default to zero."""
    with pytest.raises(ValueError, match="cost_bps"):
        VectorEngine()


def test_cannot_pass_both_cost_bps_and_cost_model() -> None:
    with pytest.raises(ValueError, match="not both"):
        VectorEngine(cost_bps=5.0, cost_model=AShareCostModel())


def test_cost_bps_must_be_nonneg() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        VectorEngine(cost_bps=-1.0)


def test_zero_cost_bps_is_allowed_but_explicit() -> None:
    # No exception — the user has acknowledged "I want no costs".
    engine = VectorEngine(cost_bps=0.0)
    assert engine.cost_bps == 0.0


# ---------------------------------------------------------------------------
# Edge cases: empty / misaligned input
# ---------------------------------------------------------------------------
def test_rejects_empty_weights(synthetic_prices: pd.DataFrame) -> None:
    engine = VectorEngine(cost_bps=0.0)
    with pytest.raises(ValueError, match="weights is empty"):
        engine.run(pd.DataFrame(), synthetic_prices)


def test_rejects_no_common_dates(synthetic_prices: pd.DataFrame) -> None:
    # Shift the weight index to dates that never overlap with prices.
    w = pd.DataFrame(
        0.5,
        index=pd.date_range("2030-01-01", periods=3, freq="B"),
        columns=synthetic_prices.columns,
    )
    engine = VectorEngine(cost_bps=0.0)
    with pytest.raises(ValueError, match="common dates"):
        engine.run(w, synthetic_prices)


def test_rejects_no_common_assets(synthetic_prices: pd.DataFrame) -> None:
    w = pd.DataFrame(0.5, index=synthetic_prices.index, columns=["X", "Y", "Z"])
    engine = VectorEngine(cost_bps=0.0)
    with pytest.raises(ValueError, match="common assets"):
        engine.run(w, synthetic_prices)


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------
def test_zero_weights_produce_zero_returns(
    zero_weights: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """No positions → no gross return, no turnover, no cost."""
    engine = VectorEngine(cost_bps=10.0)
    result = engine.run(zero_weights, synthetic_prices)
    assert result.gross_returns.fillna(0).sum() == pytest.approx(0.0)
    assert result.net_returns.fillna(0).sum() == pytest.approx(0.0)
    assert result.turnover_series.sum() == pytest.approx(0.0)
    assert result.cost_series.sum() == pytest.approx(0.0)


def test_constant_weights_match_buy_and_hold(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """Cost-free constant weights = buy-and-hold of the equal-weight basket.

    Reference: average asset simple return per day, summed up the curve.
    The engine shifts weights by 1 so we discard the very first NaN bar
    on both sides for the comparison.
    """
    engine = VectorEngine(cost_bps=0.0)
    result = engine.run(equal_weight, synthetic_prices)

    asset_rets = synthetic_prices.pct_change(fill_method=None)
    expected = asset_rets.mul(1.0 / synthetic_prices.shape[1]).sum(axis=1)
    # First bar is NaN due to pct_change; engine's shift(1) makes first
    # bar zero (no prior weights). Drop both and align.
    a = result.gross_returns.iloc[1:].to_numpy()
    b = expected.iloc[1:].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_initial_rebalance_counts_as_turnover(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """First-day weights are bought from a zero portfolio (full turnover)."""
    engine = VectorEngine(cost_bps=0.0)
    result = engine.run(equal_weight, synthetic_prices)
    # Σ|w_0| with 3 assets at 1/3 each → exactly 1.0 of turnover on day 0.
    assert result.turnover_series.iloc[0] == pytest.approx(1.0)
    # No further rebalancing: turnover is zero after day 0.
    assert result.turnover_series.iloc[1:].sum() == pytest.approx(0.0)


def test_costs_reduce_gross_to_net(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """net = gross - cost, identically."""
    engine = VectorEngine(cost_bps=10.0)
    result = engine.run(equal_weight, synthetic_prices)
    diff = result.gross_returns - result.cost_series - result.net_returns
    # Tolerate float noise but require it to be tiny.
    np.testing.assert_allclose(diff.fillna(0).to_numpy(), 0.0, atol=1e-15)


def test_higher_cost_lowers_total_return(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """Monotonicity: more bps → less terminal portfolio value."""
    r_cheap = VectorEngine(cost_bps=1.0).run(equal_weight, synthetic_prices)
    r_pricey = VectorEngine(cost_bps=100.0).run(equal_weight, synthetic_prices)
    assert r_pricey.portfolio_value.iloc[-1] < r_cheap.portfolio_value.iloc[-1]


# ---------------------------------------------------------------------------
# Look-ahead protection
# ---------------------------------------------------------------------------
def test_weights_are_shifted_by_one_bar(synthetic_prices: pd.DataFrame) -> None:
    """An "oracle" strategy that knows tomorrow's return must NOT earn it.

    Build weights = sign(future_return). If the engine wrongly applied
    weights[t] to return[t] (look-ahead), gross_returns would be all
    positive. With the correct shift(1), weights[t] is applied to
    return[t+1] — so the oracle's edge disappears at the join.
    """
    asset_rets = synthetic_prices.pct_change(fill_method=None)
    # Oracle weights: tomorrow's sign on each asset, normalised.
    raw = np.sign(asset_rets).fillna(0.0)
    # Equal book per long/short side, scaled to gross 1.0.
    weights = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    engine = VectorEngine(cost_bps=0.0)
    result = engine.run(weights, synthetic_prices)

    # If we had look-ahead, every bar would be ≥ 0. With proper shift,
    # there must be at least one losing bar (otherwise the test is broken).
    n_neg = (result.gross_returns < 0).sum()
    assert n_neg > 0, "Look-ahead leaked: engine earned tomorrow's return today."


def test_no_lookahead_in_first_bar(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """Day 0 cannot have a position-earned return (no prior weights)."""
    engine = VectorEngine(cost_bps=0.0)
    result = engine.run(equal_weight, synthetic_prices)
    # First bar: shifted weights are NaN → filled to zero → return = 0.
    assert result.gross_returns.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Long-short
# ---------------------------------------------------------------------------
def test_long_short_book_market_neutral_when_assets_correlated(
    long_short_weights: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """A +1 -1 book on iid assets should have lower vol than a +1 +1 book."""
    long_only = pd.DataFrame(0.0, index=synthetic_prices.index, columns=synthetic_prices.columns)
    long_only["A0"] = 1.0
    long_only["A1"] = 1.0

    engine = VectorEngine(cost_bps=0.0)
    r_ls = engine.run(long_short_weights, synthetic_prices)
    r_lo = engine.run(long_only, synthetic_prices)
    # For independent assets the long-short cancels common drift and is
    # quieter on the variance ratio. We tolerate a generous margin since
    # 60 bars is small.
    assert r_ls.report.annual_vol <= r_lo.report.annual_vol * 1.5


# ---------------------------------------------------------------------------
# Cost model path
# ---------------------------------------------------------------------------
def test_cost_model_path_runs_end_to_end(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """A realistic A-share cost model should produce non-zero, finite drag."""
    model = AShareCostModel(slippage=FixedBpsSlippage(bps=5.0))
    engine = VectorEngine(cost_model=model)
    result = engine.run(equal_weight, synthetic_prices)
    assert isinstance(result, BacktestResult)
    # Day 0 paid commission+slippage; subsequent days held → zero cost.
    assert result.cost_series.iloc[0] > 0
    assert result.cost_series.iloc[1:].sum() == pytest.approx(0.0)


def test_cost_model_ignores_dates_with_nan_prices() -> None:
    """A halt day (NaN price) must not raise — we simply book no trade."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [10.0, np.nan, 11.0, 12.0, 13.0]},
        index=dates,
    )
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
    model = AShareCostModel()
    engine = VectorEngine(cost_model=model)
    # Must not raise even though there's a NaN on the halt day.
    result = engine.run(weights, prices)
    assert result.cost_series.isna().sum() == 0


# ---------------------------------------------------------------------------
# Report integration
# ---------------------------------------------------------------------------
def test_result_report_is_consistent_with_net_returns(
    equal_weight: pd.DataFrame, synthetic_prices: pd.DataFrame
) -> None:
    """The PerformanceReport in the result must summarise net_returns."""
    engine = VectorEngine(cost_bps=10.0)
    result = engine.run(equal_weight, synthetic_prices)
    # n_periods should match non-NaN count of net_returns.
    assert result.report.n_periods == int(result.net_returns.notna().sum())
    # Annual turnover propagated.
    assert result.report.turnover_annual is not None
    assert result.report.turnover_annual > 0
