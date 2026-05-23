"""Tests for ``backtest.factor_bridge.long_short_weights``.

The bridge has narrow but important invariants:
  * top quantile is bought equal-weight, bottom is sold equal-weight
  * Σ|w| == gross_leverage on healthy dates, 0 on degenerate dates
  * higher_is_better=False flips the sign of the entire book
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.backtest.factor_bridge import long_short_weights
from quant_lucky.factors.tester import (
    CleanFactorData,
    get_clean_factor_and_forward_returns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def planted_factor() -> tuple[pd.Series, pd.DataFrame, CleanFactorData]:
    """A factor with a hand-built ranking so the bucketing is deterministic.

    6 assets, 30 business days. Factor value = a constant per asset
    (A0 lowest, A5 highest), so every cross-section bucketises into the
    same 3 buckets:
      * bucket 1 (bottom): A0, A1
      * bucket 2 (middle): A2, A3
      * bucket 3 (top):    A4, A5
    """
    rng = np.random.default_rng(7)
    n_days = 30
    assets = [f"A{i}" for i in range(6)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Random walk prices, just so forward returns exist.
    log_ret = rng.normal(0.0005, 0.01, size=(n_days, 6))
    prices = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
        index=dates,
        columns=assets,
    )
    prices.index.name = "date"
    prices.columns.name = "asset"

    # Constant per-asset factor: same ranking every day.
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    factor_vals = np.tile(base, (n_days, 1))
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = pd.Series(factor_vals.ravel(), index=idx, name="planted")

    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1, 5], quantiles=3)
    return factor, prices, clean


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_rejects_nonpositive_leverage(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    _, _, clean = planted_factor
    with pytest.raises(ValueError, match="gross_leverage"):
        long_short_weights(clean, gross_leverage=0.0)


def test_rejects_single_quantile(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    # Reach in and corrupt the quantile count; the function should reject.
    _, _, clean = planted_factor
    bad = CleanFactorData(
        factor=clean.factor,
        forward_returns=clean.forward_returns,
        quantile=clean.quantile,
        quantiles=1,
        periods=clean.periods,
    )
    with pytest.raises(ValueError, match="at least 2"):
        long_short_weights(bad)


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------
def test_gross_exposure_equals_target_on_healthy_dates(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """Σ|w| should equal gross_leverage where both buckets are populated."""
    _, _, clean = planted_factor
    weights = long_short_weights(clean, gross_leverage=1.0)
    abs_sum = weights.abs().sum(axis=1)
    # Drop any all-zero (degenerate) rows; the others must hit 1.0 exactly.
    healthy = abs_sum[abs_sum > 0]
    np.testing.assert_allclose(healthy.to_numpy(), 1.0, rtol=1e-12, atol=1e-12)


def test_book_is_dollar_neutral(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """Long-short by construction → row sum (signed) should be 0."""
    _, _, clean = planted_factor
    weights = long_short_weights(clean, gross_leverage=1.0)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_top_assets_are_long_bottom_assets_are_short(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """Constant-ranking factor → A4/A5 always long, A0/A1 always short."""
    _, _, clean = planted_factor
    weights = long_short_weights(clean, higher_is_better=True)
    # Drop rows where the factor was masked out (head of sample, no forward
    # returns yet) — those are all-zero by design.
    active = weights.loc[(weights != 0).any(axis=1)]
    assert (active["A4"] > 0).all()
    assert (active["A5"] > 0).all()
    assert (active["A0"] < 0).all()
    assert (active["A1"] < 0).all()
    # Middle assets must be exactly zero.
    np.testing.assert_array_equal(active["A2"].to_numpy(), 0.0)
    np.testing.assert_array_equal(active["A3"].to_numpy(), 0.0)


def test_higher_is_better_false_flips_all_weights(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """When the factor is "lower is better", we long the bottom bucket."""
    _, _, clean = planted_factor
    long_top = long_short_weights(clean, higher_is_better=True)
    long_bottom = long_short_weights(clean, higher_is_better=False)
    np.testing.assert_allclose(long_bottom.to_numpy(), -long_top.to_numpy(), rtol=1e-12, atol=1e-12)


def test_gross_leverage_scales_linearly(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """A 2x leverage book is exactly 2x the unit book."""
    _, _, clean = planted_factor
    w1 = long_short_weights(clean, gross_leverage=1.0)
    w2 = long_short_weights(clean, gross_leverage=2.0)
    np.testing.assert_allclose(w2.to_numpy(), 2.0 * w1.to_numpy(), atol=1e-12)


def test_within_side_weights_are_equal(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """Equal-weight within each bucket: A4 weight == A5 weight."""
    _, _, clean = planted_factor
    weights = long_short_weights(clean, gross_leverage=1.0)
    active = weights.loc[(weights != 0).any(axis=1)]
    np.testing.assert_allclose(active["A4"].to_numpy(), active["A5"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(active["A0"].to_numpy(), active["A1"].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------
def test_weights_run_through_vector_engine(
    planted_factor: tuple[pd.Series, pd.DataFrame, CleanFactorData],
) -> None:
    """Smoke test: produced weights are directly consumable by VectorEngine."""
    from quant_lucky.backtest.vector import VectorEngine

    _, prices, clean = planted_factor
    weights = long_short_weights(clean)
    engine = VectorEngine(cost_bps=10.0)
    result = engine.run(weights, prices)
    # Just verify shape / non-empty result; numerical accuracy is the
    # engine's own test suite.
    assert result.report.n_periods > 0
    assert result.report.turnover_annual is not None and result.report.turnover_annual > 0
