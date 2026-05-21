"""Tests for ``backtest.report``: PerformanceReport + compute_performance.

These exercise the metric definitions on hand-crafted return series
where the answer is known by construction. The point is to verify our
formulas, not to test pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.backtest.report import PerformanceReport, compute_performance


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match=r"pd\.Series"):
        compute_performance([0.01, 0.02, 0.03])  # type: ignore[arg-type]


def test_rejects_zero_periods_per_year() -> None:
    s = pd.Series([0.01, 0.0, -0.01])
    with pytest.raises(ValueError, match="periods_per_year"):
        compute_performance(s, periods_per_year=0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_series_returns_nan_report() -> None:
    s = pd.Series(dtype=float)
    report = compute_performance(s)
    assert report.n_periods == 0
    assert np.isnan(report.annual_return)
    assert np.isnan(report.sharpe)
    assert report.max_drawdown == 0.0


def test_all_nan_series_treated_as_empty() -> None:
    s = pd.Series([np.nan, np.nan, np.nan])
    report = compute_performance(s)
    assert report.n_periods == 0


def test_single_observation_has_undefined_vol() -> None:
    s = pd.Series([0.01])
    report = compute_performance(s)
    assert report.n_periods == 1
    # std with n=1 → NaN → annual_vol NaN → Sharpe NaN.
    assert np.isnan(report.annual_vol)
    assert np.isnan(report.sharpe)


# ---------------------------------------------------------------------------
# Known-answer cases
# ---------------------------------------------------------------------------
def test_constant_positive_return_zero_volatility() -> None:
    """A perfectly constant return has zero std, Sharpe is undefined."""
    s = pd.Series([0.001] * 252)
    report = compute_performance(s, periods_per_year=252)
    # Annual return: 0.001 * 252 = 0.252
    assert report.annual_return == pytest.approx(0.252)
    assert report.annual_vol == pytest.approx(0.0, abs=1e-12)
    # Sharpe undefined when vol == 0.
    assert np.isnan(report.sharpe)
    assert report.hit_rate == 1.0


def test_max_drawdown_on_falling_then_recovering_series() -> None:
    """Drawdown is computed from peak; a recovery does not erase it."""
    # +10%, -50%, +0% → peak = 1.10, trough = 0.55, mdd = -50%.
    s = pd.Series([0.10, -0.50, 0.0])
    report = compute_performance(s)
    assert report.max_drawdown == pytest.approx(-0.50)


def test_drawdown_is_zero_on_monotonically_rising_series() -> None:
    s = pd.Series([0.01] * 30)
    report = compute_performance(s)
    assert report.max_drawdown == pytest.approx(0.0, abs=1e-12)
    # Calmar undefined when there is no drawdown.
    assert np.isnan(report.calmar)


def test_hit_rate_counts_strictly_positive_periods() -> None:
    s = pd.Series([0.01, -0.01, 0.0, 0.02, -0.005])
    report = compute_performance(s)
    # 2 of 5 strictly positive.
    assert report.hit_rate == pytest.approx(0.4)


def test_sharpe_sign_matches_mean_return() -> None:
    """A negative-mean series must have negative Sharpe."""
    s = pd.Series([-0.001, 0.002, -0.003, 0.001, -0.002, 0.0005])
    report = compute_performance(s)
    assert report.annual_return < 0
    assert report.sharpe < 0


def test_sortino_geq_sharpe_when_upside_volatile() -> None:
    """If upside dominates volatility, Sortino > Sharpe (downside std smaller)."""
    # Mostly positive returns with a few small losses.
    s = pd.Series([0.02, 0.03, -0.001, 0.025, -0.002, 0.02, 0.015])
    report = compute_performance(s)
    assert report.sortino > report.sharpe


def test_rf_subtraction_lowers_sharpe() -> None:
    """Risk-free rate subtracts from excess return → lower Sharpe."""
    s = pd.Series([0.001] * 252 + [0.0] * 1)  # tiny std, makes test stable
    no_rf = compute_performance(s, rf_annual=0.0)
    with_rf = compute_performance(s, rf_annual=0.10)
    # 10% rf eats most of the 25.2% annualised return.
    assert with_rf.annual_return < no_rf.annual_return


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def test_to_summary_returns_series_with_required_fields() -> None:
    s = pd.Series([0.01, -0.02, 0.015, 0.005])
    report = compute_performance(s)
    summary = report.to_summary()
    expected_keys = {
        "annual_return",
        "annual_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "n_periods",
    }
    assert expected_keys.issubset(set(summary.index))


def test_repr_renders_without_error() -> None:
    s = pd.Series([0.01, -0.02, 0.015])
    report = compute_performance(s, turnover_annual=2.5)
    text = repr(report)
    assert "PerformanceReport" in text
    assert "sharpe=" in text
    assert "turnover_annual=" in text


def test_report_is_immutable() -> None:
    """frozen=True dataclass — assignment must raise FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    s = pd.Series([0.01, 0.02])
    report = compute_performance(s)
    with pytest.raises(FrozenInstanceError):
        report.sharpe = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------
def test_periods_per_year_scales_annual_return_linearly() -> None:
    """Same per-period mean → annual_return scales with periods_per_year."""
    s = pd.Series([0.001] * 100)
    daily = compute_performance(s, periods_per_year=252)
    weekly = compute_performance(s, periods_per_year=52)
    assert daily.annual_return == pytest.approx(0.001 * 252)
    assert weekly.annual_return == pytest.approx(0.001 * 52)


def test_isinstance_returns_performance_report() -> None:
    s = pd.Series([0.01, 0.02])
    report = compute_performance(s)
    assert isinstance(report, PerformanceReport)
