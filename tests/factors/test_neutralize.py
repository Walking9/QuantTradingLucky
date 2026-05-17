"""Tests for cross-sectional pre-processing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.factors.neutralize import (
    neutralize_by_group,
    neutralize_by_size,
    standardize_rank,
    standardize_zscore,
    winsorize_mad,
)


def _make_series(values: np.ndarray, n_assets: int) -> pd.Series:
    """Build a (date, asset) Series of given values, ``n_assets`` per date."""
    n_dates = values.size // n_assets
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    return pd.Series(values, index=idx, name="f")


# ---------------------------------------------------------------------------
# winsorize_mad
# ---------------------------------------------------------------------------
def test_winsorize_mad_clips_outliers() -> None:
    vals = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0] * 4)
    s = _make_series(vals, n_assets=5)
    out = winsorize_mad(s, k=3.0)
    # Outliers (1000 / -1000) should be clipped towards zero.
    out_by_date = out.groupby(level="date").apply(list)
    for row in out_by_date:
        # Both extremes must now be within a finite range, not 1000.
        assert max(row) < 100, f"row not clipped: {row}"
        assert min(row) > -100, f"row not clipped: {row}"


def test_winsorize_mad_rejects_non_positive_k() -> None:
    s = _make_series(np.arange(10.0), n_assets=5)
    with pytest.raises(ValueError, match=">"):
        winsorize_mad(s, k=0.0)


def test_winsorize_mad_is_no_op_when_mad_is_zero() -> None:
    # All identical -> MAD=0 -> function should leave values untouched.
    s = _make_series(np.array([5.0] * 10), n_assets=5)
    out = winsorize_mad(s)
    np.testing.assert_array_equal(s.values, out.values)


# ---------------------------------------------------------------------------
# standardize_zscore
# ---------------------------------------------------------------------------
def test_zscore_mean_zero_per_date() -> None:
    rng = np.random.default_rng(0)
    n = 100
    vals = rng.normal(50, 10, n * 6)
    s = _make_series(vals, n_assets=6)
    out = standardize_zscore(s)
    by_date_mean = out.groupby(level="date").mean()
    np.testing.assert_allclose(by_date_mean.values, 0.0, atol=1e-12)


def test_zscore_unit_std_per_date() -> None:
    rng = np.random.default_rng(1)
    n = 80
    vals = rng.normal(0, 5, n * 6)
    s = _make_series(vals, n_assets=6)
    out = standardize_zscore(s)
    by_date_std = out.groupby(level="date").std(ddof=0)
    np.testing.assert_allclose(by_date_std.values, 1.0, atol=1e-12)


def test_zscore_zero_std_falls_back_to_demean() -> None:
    s = _make_series(np.array([2.0] * 10), n_assets=5)
    out = standardize_zscore(s)
    # All values equal -> std=0 -> output is (x - mean) = 0
    np.testing.assert_allclose(out.values, 0.0)


# ---------------------------------------------------------------------------
# standardize_rank
# ---------------------------------------------------------------------------
def test_rank_in_minus_half_to_half() -> None:
    rng = np.random.default_rng(2)
    s = _make_series(rng.normal(0, 1, 60), n_assets=6)
    out = standardize_rank(s)
    assert out.min() >= -0.5
    assert out.max() <= 0.5


# ---------------------------------------------------------------------------
# neutralize_by_group
# ---------------------------------------------------------------------------
def test_neutralize_by_group_removes_group_mean() -> None:
    # Two groups, each with a known mean. After demean, per-date per-group
    # mean should be ~0.
    n_dates = 30
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    assets = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    # A,B in group "g1" (mean 10); C,D in group "g2" (mean -5).
    factor_values = np.tile([10.0, 10.0, -5.0, -5.0], n_dates)
    group_values = np.tile(["g1", "g1", "g2", "g2"], n_dates)
    factor = pd.Series(factor_values, index=idx, name="f")
    group = pd.Series(group_values, index=idx, name="g")

    out = neutralize_by_group(factor, group, standardize=False)
    # Within each (date, group), mean must be ~0.
    df = pd.concat([out.rename("v"), group.rename("g")], axis=1)
    grouped_means = df.groupby([df.index.get_level_values("date"), "g"])["v"].mean()
    np.testing.assert_allclose(grouped_means.values, 0.0, atol=1e-12)


def test_neutralize_by_group_raises_on_total_misalignment() -> None:
    n = 12
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    assets = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = pd.Series(np.arange(n, dtype=float), index=idx, name="f")

    # Group series indexed with disjoint assets and dates.
    bad_idx = pd.MultiIndex.from_product(
        [pd.date_range("2030-01-01", periods=3, freq="B"), ["X", "Y", "Z", "W"]],
        names=["date", "asset"],
    )
    group = pd.Series(["g"] * n, index=bad_idx, name="g")
    with pytest.raises(ValueError, match="share no"):
        neutralize_by_group(factor, group)


# ---------------------------------------------------------------------------
# neutralize_by_size
# ---------------------------------------------------------------------------
def test_neutralize_by_size_residual_uncorrelated_with_logcap() -> None:
    """After residualising against log(cap), the residual should have
    near-zero correlation with log(cap) by OLS construction."""
    rng = np.random.default_rng(5)
    n_dates = 40
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    assets = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    cap = pd.Series(np.tile(np.geomspace(1e8, 1e11, 10), n_dates), index=idx, name="cap")
    # Construct a factor that is genuinely correlated with log(cap):
    #   factor = 2 * log(cap) + noise
    log_cap = np.log(cap.values)
    factor = pd.Series(
        2 * log_cap + rng.normal(0, 0.5, len(idx)),
        index=idx,
        name="f",
    )

    resid = neutralize_by_size(factor, cap, standardize=False)

    df = pd.concat([resid.rename("r"), pd.Series(log_cap, index=idx).rename("lc")], axis=1).dropna()
    corr = df["r"].corr(df["lc"])
    assert abs(corr) < 0.1, f"residual correlated with log(cap): {corr}"


def test_neutralize_by_size_rejects_non_positive_cap() -> None:
    n = 8
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    assets = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = pd.Series(np.arange(n, dtype=float), index=idx)
    cap = pd.Series([1.0, 2.0, 3.0, 0.0] * 2, index=idx)
    with pytest.raises(ValueError, match="positive"):
        neutralize_by_size(factor, cap)
