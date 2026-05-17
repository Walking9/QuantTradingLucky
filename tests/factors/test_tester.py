"""Tests for the single-factor tester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.factors.tester import (
    CleanFactorData,
    compute_ic,
    compute_long_short,
    compute_mean_returns_by_quantile,
    compute_turnover,
    get_clean_factor_and_forward_returns,
    ic_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_constant_factor(prices: pd.DataFrame, values: list[float]) -> pd.Series:
    """A factor that is constant per-asset (e.g. the oracle drift). Index aligns to prices."""
    long_idx = prices.stack(future_stack=True).index
    n_assets = len(prices.columns)
    n_days = len(prices.index)
    repeated = np.tile(values, n_days)
    assert len(repeated) == n_days * n_assets
    return pd.Series(repeated, index=long_idx, name="oracle")


# ---------------------------------------------------------------------------
# get_clean_factor_and_forward_returns
# ---------------------------------------------------------------------------
def test_clean_factor_validates_factor_type(synthetic_panel: tuple) -> None:
    _, prices = synthetic_panel
    with pytest.raises(TypeError, match=r"pd\.Series"):
        get_clean_factor_and_forward_returns(np.array([1, 2, 3]), prices, periods=[1, 5])


def test_clean_factor_validates_factor_index(synthetic_panel: tuple) -> None:
    _, prices = synthetic_panel
    bad = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match="MultiIndex"):
        get_clean_factor_and_forward_returns(bad, prices, periods=[1, 5])


def test_clean_factor_validates_prices_type(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    factor = df["close"]  # placeholder factor
    with pytest.raises(TypeError, match="wide DataFrame"):
        get_clean_factor_and_forward_returns(factor, factor, periods=[1, 5])


def test_clean_factor_periods_must_be_positive(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")
    with pytest.raises(ValueError, match=r">= 1"):
        get_clean_factor_and_forward_returns(factor, prices, periods=[0, 5])


def test_clean_factor_returns_aligned_bundle(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")  # any well-defined factor
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1, 5], quantiles=3)
    assert isinstance(clean, CleanFactorData)
    assert clean.factor.index.equals(clean.quantile.index)
    assert clean.factor.index.equals(clean.forward_returns.index)
    # Columns for each requested horizon.
    assert list(clean.forward_returns.columns) == ["period_1", "period_5"]


def test_clean_factor_drops_all_nan_factor_rows(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = _momentum_factor_window_5(df)
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1, 5], quantiles=3)
    # NaN factor rows should be excluded.
    assert clean.factor.notna().all()


def _momentum_factor_window_5(df: pd.DataFrame) -> pd.Series:
    """Helper: 5-day momentum which has NaN heads — used to test NaN dropping."""
    from quant_lucky.factors.base import MomentumFactor

    return MomentumFactor(window=5).compute(df)


# ---------------------------------------------------------------------------
# Quantile bucketisation
# ---------------------------------------------------------------------------
def test_quantile_labels_are_1_to_q(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("price")
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=4)
    labels = set(clean.quantile.dropna().unique().tolist())
    assert labels.issubset({1, 2, 3, 4})


def test_quantile_rejects_q_lt_2(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("price")
    with pytest.raises(ValueError, match="quantiles"):
        get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=1)


def test_quantile_handles_all_ties() -> None:
    """If every asset has the same factor value, qcut will fail — we should
    return NaN buckets rather than raising."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    assets = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = pd.Series(1.0, index=idx, name="f")  # constant
    prices = pd.DataFrame(np.arange(40).reshape(10, 4), index=dates, columns=assets, dtype=float)
    prices.index.name = "date"
    prices.columns.name = "asset"
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=3)
    assert clean.quantile.isna().all()


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------
def test_ic_method_validation(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=3)
    with pytest.raises(ValueError, match="method"):
        compute_ic(clean, method="kendall")  # type: ignore[arg-type]


def test_ic_returns_one_column_per_horizon(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1, 5, 20], quantiles=3)
    ic = compute_ic(clean)
    assert list(ic.columns) == ["period_1", "period_5", "period_20"]


def test_oracle_factor_has_strong_positive_ic(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    """The oracle factor (= true drift per asset) must have a strongly
    positive average IC against forward returns. If this test fails, the
    tester is broken — there's no other way to fail it on planted signal."""
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5, 20], quantiles=3)
    ic = compute_ic(clean, method="spearman")
    summary = ic_summary(ic)
    # Both horizons should have clearly positive mean IC (not just > 0).
    assert summary.loc["period_5", "ic_mean"] > 0.3
    assert summary.loc["period_20", "ic_mean"] > 0.5


def test_ic_summary_handles_empty_column() -> None:
    ic = pd.DataFrame({"period_1": [np.nan, np.nan]}, index=pd.RangeIndex(2))
    summary = ic_summary(ic)
    assert pd.isna(summary.loc["period_1", "ic_mean"])
    assert summary.loc["period_1", "n"] == 0


# ---------------------------------------------------------------------------
# Quantile returns
# ---------------------------------------------------------------------------
def test_mean_returns_by_quantile_shape(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[1, 5], quantiles=3)
    pooled = compute_mean_returns_by_quantile(clean)
    assert pooled.shape == (3, 2)  # 3 quantiles x 2 periods


def test_mean_returns_by_date_returns_multiindex(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5], quantiles=3)
    by_date = compute_mean_returns_by_quantile(clean, by_date=True)
    assert isinstance(by_date.index, pd.MultiIndex)
    assert by_date.index.names == ["date", "quantile"]


def test_oracle_top_minus_bottom_is_positive(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    """Top quantile (positive drift) must outperform bottom quantile."""
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5, 20], quantiles=3)
    pooled = compute_mean_returns_by_quantile(clean)
    assert pooled.loc[3, "period_5"] > pooled.loc[1, "period_5"]
    assert pooled.loc[3, "period_20"] > pooled.loc[1, "period_20"]


# ---------------------------------------------------------------------------
# Long-short
# ---------------------------------------------------------------------------
def test_long_short_default_period_is_first(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5, 20], quantiles=3)
    ls_default = compute_long_short(clean)
    ls_explicit = compute_long_short(clean, period=5)
    np.testing.assert_allclose(
        ls_default["series"].dropna().values,
        ls_explicit["series"].dropna().values,
    )


def test_long_short_rejects_unknown_period(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5], quantiles=3)
    with pytest.raises(ValueError, match="not in computed periods"):
        compute_long_short(clean, period=10)


def test_long_short_flips_sign_when_higher_is_better_false(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5], quantiles=3)
    ls_up = compute_long_short(clean, period=5, higher_is_better=True)
    ls_down = compute_long_short(clean, period=5, higher_is_better=False)
    # Series should be exact negatives where both are non-NaN.
    aligned = pd.concat(
        [ls_up["series"].rename("u"), ls_down["series"].rename("d")], axis=1
    ).dropna()
    np.testing.assert_allclose(aligned["u"].values, -aligned["d"].values)


def test_long_short_oracle_is_profitable(
    panel_with_signal: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, prices, oracle = panel_with_signal
    clean = get_clean_factor_and_forward_returns(oracle, prices, periods=[5], quantiles=3)
    ls = compute_long_short(clean, period=5, higher_is_better=True)
    assert ls["annualised_return"] > 0
    assert ls["sharpe"] > 0


def test_long_short_handles_too_few_obs() -> None:
    """Synthesise a clean bundle with very few rows so the sharpe path
    returns NaN gracefully."""
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    assets = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factor = pd.Series([1.0, 2.0, 3.0, 4.0] * 4, index=idx, name="f")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, (4, 4)), axis=0) * 100,
        index=dates,
        columns=assets,
    )
    prices.index.name = "date"
    prices.columns.name = "asset"
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1, 5], quantiles=2)
    # period=5 with only 4 rows gives few or no valid observations.
    ls = compute_long_short(clean, period=5)
    # Either NaN or a finite number — never an exception.
    assert isinstance(ls["sharpe"], float)


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------
def test_turnover_returns_one_value_per_quantile(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=3)
    tov = compute_turnover(clean)
    assert sorted(tov.dropna().index.tolist()) == [1, 2, 3]


def test_turnover_in_zero_one_range(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = df["close"].rename("f")
    clean = get_clean_factor_and_forward_returns(factor, prices, periods=[1], quantiles=3)
    tov = compute_turnover(clean).dropna()
    assert (tov >= 0).all()
    assert (tov <= 1).all()
