"""Tests for factor base class and concrete factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.factors.base import (
    MomentumFactor,
    PriceVolumeFactor,
    ReversalFactor,
    TurnoverFactor,
    VolatilityFactor,
    validate_panel,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_panel_rejects_single_level_index() -> None:
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_panel(df)


def test_validate_panel_rejects_missing_columns(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    df_missing = df.drop(columns=["close"])
    with pytest.raises(ValueError, match="missing required"):
        validate_panel(df_missing)


def test_validate_panel_accepts_clean_input(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    # Should not raise.
    validate_panel(df)


# ---------------------------------------------------------------------------
# MomentumFactor
# ---------------------------------------------------------------------------
def test_momentum_window_must_be_positive(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    with pytest.raises(ValueError, match="window"):
        MomentumFactor(window=0).compute(df)


def test_momentum_matches_manual_pct_change(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    factor = MomentumFactor(window=10).compute(df)

    # Direct computation on the wide price frame should match.
    expected = prices.pct_change(10, fill_method=None).stack(future_stack=True)
    # Both should be NaN in the same places.
    aligned = pd.concat([factor.rename("got"), expected.rename("want")], axis=1).dropna()
    np.testing.assert_allclose(aligned["got"].values, aligned["want"].values, rtol=1e-10)


def test_momentum_with_skip_lags_window(synthetic_panel: tuple) -> None:
    df, prices = synthetic_panel
    skip = 3
    window = 10
    factor = MomentumFactor(window=window, skip=skip).compute(df)
    expected = prices.shift(skip).pct_change(window, fill_method=None).stack(future_stack=True)
    aligned = pd.concat([factor.rename("got"), expected.rename("want")], axis=1).dropna()
    np.testing.assert_allclose(aligned["got"].values, aligned["want"].values, rtol=1e-10)


def test_momentum_first_window_rows_are_nan(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    factor = MomentumFactor(window=20).compute(df)
    # First 20 dates of each asset must be NaN (no history yet).
    first_dates = factor.index.get_level_values("date").unique()[:20]
    head = factor[factor.index.get_level_values("date").isin(first_dates)]
    assert head.isna().all(), "head of momentum should be NaN before window fills"


# ---------------------------------------------------------------------------
# ReversalFactor
# ---------------------------------------------------------------------------
def test_reversal_is_negative_momentum(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    rev = ReversalFactor(window=5).compute(df)
    mom = MomentumFactor(window=5).compute(df)
    aligned = pd.concat([rev.rename("rev"), mom.rename("mom")], axis=1).dropna()
    np.testing.assert_allclose(aligned["rev"].values, -aligned["mom"].values, rtol=1e-10)


# ---------------------------------------------------------------------------
# VolatilityFactor
# ---------------------------------------------------------------------------
def test_volatility_strictly_non_negative(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    vol = VolatilityFactor(window=20).compute(df).dropna()
    assert (vol >= 0).all()


def test_volatility_rejects_window_one(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    with pytest.raises(ValueError, match="window"):
        VolatilityFactor(window=1).compute(df)


def test_volatility_higher_is_better_false() -> None:
    assert VolatilityFactor().higher_is_better is False


# ---------------------------------------------------------------------------
# TurnoverFactor
# ---------------------------------------------------------------------------
def test_turnover_proxy_no_inf_when_baseline_zero(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    # Force the baseline volume to be zero for a stretch to trigger the
    # divide-by-zero guard.
    df = df.copy()
    target_asset = df.index.get_level_values("asset")[0]
    mask = df.index.get_level_values("asset") == target_asset
    df.loc[mask, "volume"] = 0.0
    tov = TurnoverFactor(window=5, baseline_window=20).compute(df)
    # No infinities, only NaNs where the proxy is undefined.
    assert not np.isinf(tov.replace([np.inf, -np.inf], np.nan)).any()


def test_turnover_use_provider_column_requires_it(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel  # has no 'turnover' column
    with pytest.raises(ValueError, match="provider_column"):
        TurnoverFactor(use_provider_column=True).compute(df)


def test_turnover_use_provider_column_path(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    df = df.copy()
    rng = np.random.default_rng(3)
    df["turnover"] = rng.uniform(0, 0.1, len(df))
    tov = TurnoverFactor(window=5, use_provider_column=True).compute(df)
    assert tov.notna().any()


# ---------------------------------------------------------------------------
# PriceVolumeFactor
# ---------------------------------------------------------------------------
def test_price_volume_requires_amount(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    df = df.drop(columns=["amount"])
    with pytest.raises(ValueError, match="missing required"):
        PriceVolumeFactor(window=20).compute(df)


def test_price_volume_finite_when_amount_positive(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    pv = PriceVolumeFactor(window=20).compute(df).dropna()
    assert np.isfinite(pv).all()


# ---------------------------------------------------------------------------
# Index handling
# ---------------------------------------------------------------------------
def test_factor_handles_unnamed_levels(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    df = df.copy()
    df.index = df.index.set_names([None, None])
    # Should still work — base class renames silently.
    factor = MomentumFactor(window=10).compute(df)
    assert factor.index.names == ["date", "asset"]


def test_factor_output_index_matches_input(synthetic_panel: tuple) -> None:
    df, _ = synthetic_panel
    factor = MomentumFactor(window=10).compute(df)
    assert factor.index.equals(df.sort_index().index)
