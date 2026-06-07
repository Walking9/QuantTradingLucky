"""Tests for cross-sectional momentum weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.momentum import (
    cross_sectional_momentum_weights,
    momentum_strategy_fn,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_price_panel(n_assets=12, n_days=400, seed=7)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            cross_sectional_momentum_weights([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cross_sectional_momentum_weights(pd.DataFrame())

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_lookback(self, panel: pd.DataFrame, bad: int) -> None:
        with pytest.raises(ValueError, match="lookback"):
            cross_sectional_momentum_weights(panel, lookback=bad)

    def test_rejects_negative_skip(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="skip"):
            cross_sectional_momentum_weights(panel, skip=-1)

    @pytest.mark.parametrize("bad", [0.0, 0.6, 1.0, -0.1])
    def test_rejects_bad_quantile(self, panel: pd.DataFrame, bad: float) -> None:
        with pytest.raises(ValueError, match="top_quantile"):
            cross_sectional_momentum_weights(panel, top_quantile=bad)

    def test_rejects_bad_gross(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="gross_leverage"):
            cross_sectional_momentum_weights(panel, gross_leverage=0.0)


# ---------------------------------------------------------------------------
# Shape & exposure invariants
# ---------------------------------------------------------------------------
class TestExposure:
    def test_output_shape_and_axes(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(panel, lookback=60, skip=5)
        assert w.shape == panel.shape
        assert list(w.columns) == list(panel.columns)
        assert w.index.equals(panel.index)
        assert w.index.name == "date"
        assert w.columns.name == "asset"

    def test_long_short_dollar_neutral(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(panel, lookback=60, skip=5, top_quantile=0.25)
        # Net exposure is ~0 on every populated date (dollar-neutral book).
        assert w.sum(axis=1).abs().max() < 1e-9

    def test_long_short_gross_capped(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(
            panel, lookback=60, skip=5, top_quantile=0.25, gross_leverage=1.0
        )
        gross = w.abs().sum(axis=1)
        # Populated dates hit exactly the target gross; never exceed it.
        assert gross.max() <= 1.0 + 1e-9
        assert np.allclose(gross[gross > 0], 1.0)

    def test_long_only_non_negative_and_full(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(
            panel, lookback=60, skip=5, top_quantile=0.25, long_only=True, gross_leverage=1.0
        )
        assert (w >= -1e-12).all().all()
        invested = w.sum(axis=1)
        assert np.allclose(invested[invested > 0], 1.0)

    def test_gross_leverage_scales(self, panel: pd.DataFrame) -> None:
        w1 = cross_sectional_momentum_weights(panel, lookback=60, skip=5, gross_leverage=1.0)
        w2 = cross_sectional_momentum_weights(panel, lookback=60, skip=5, gross_leverage=2.0)
        pd.testing.assert_frame_equal(2.0 * w1, w2)

    def test_head_is_flat_until_window_fills(self, panel: pd.DataFrame) -> None:
        lookback, skip = 60, 5
        w = cross_sectional_momentum_weights(panel, lookback=lookback, skip=skip)
        warmup = lookback + skip
        # No position can form before the trailing window exists.
        assert w.iloc[:warmup].abs().to_numpy().sum() == 0.0


# ---------------------------------------------------------------------------
# Economic correctness
# ---------------------------------------------------------------------------
class TestSignal:
    def test_longs_the_winner_shorts_the_loser(self) -> None:
        # 4 assets, monotone ranking by trailing return. With top_quantile
        # 0.25 → 1 name each side: long the strongest, short the weakest.
        dates = pd.bdate_range("2021-01-01", periods=40)
        base = np.linspace(0, 1, len(dates))
        prices = pd.DataFrame(
            {
                "A": 100 * (1 + 0.0010) ** np.arange(len(dates)),  # strongest up
                "B": 100 * (1 + 0.0005) ** np.arange(len(dates)),
                "C": 100 * (1 - 0.0005) ** np.arange(len(dates)),
                "D": 100 * (1 - 0.0010) ** np.arange(len(dates)),  # strongest down
            },
            index=dates,
        )
        del base
        w = cross_sectional_momentum_weights(
            prices, lookback=20, skip=0, top_quantile=0.25, gross_leverage=1.0
        )
        last = w.iloc[-1]
        assert last["A"] == pytest.approx(0.5)
        assert last["D"] == pytest.approx(-0.5)
        assert last["B"] == pytest.approx(0.0)
        assert last["C"] == pytest.approx(0.0)

    def test_thin_cross_section_is_flat(self) -> None:
        # A single asset cannot form a cross-section → all-zero weights.
        dates = pd.bdate_range("2021-01-01", periods=80)
        prices = pd.DataFrame({"ONLY": 100 * (1.001) ** np.arange(len(dates))}, index=dates)
        w = cross_sectional_momentum_weights(prices, lookback=20, skip=0, top_quantile=0.25)
        assert w.to_numpy().sum() == 0.0

    def test_skip_excludes_recent_window(self) -> None:
        # An asset that surged in the last few days but lagged before should
        # rank differently with skip>0 than with skip=0.
        dates = pd.bdate_range("2021-01-01", periods=60)
        flat_then_spike = np.concatenate(
            [np.full(50, 100.0), np.array([100, 110, 121, 133, 146, 160, 176, 194, 213, 234])]
        )
        steady = 100 * (1.002) ** np.arange(60)
        prices = pd.DataFrame({"SPIKE": flat_then_spike, "STEADY": steady}, index=dates)
        w_no_skip = cross_sectional_momentum_weights(
            prices, lookback=20, skip=0, top_quantile=0.5, gross_leverage=1.0
        )
        w_skip = cross_sectional_momentum_weights(
            prices, lookback=20, skip=10, top_quantile=0.5, gross_leverage=1.0
        )
        # With no skip the recent spike makes SPIKE the winner; with a 10-day
        # skip the spike is excluded and SPIKE is no longer favoured.
        assert w_no_skip.iloc[-1]["SPIKE"] > w_no_skip.iloc[-1]["STEADY"]
        assert w_skip.iloc[-1]["SPIKE"] <= w_skip.iloc[-1]["STEADY"]


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------
class TestNoLookAhead:
    def test_future_prices_do_not_change_past_weights(self, panel: pd.DataFrame) -> None:
        cut = 250
        w_full = cross_sectional_momentum_weights(panel, lookback=60, skip=5)
        perturbed = panel.copy()
        perturbed.iloc[cut + 1 :] *= 1.5  # mangle the future
        w_pert = cross_sectional_momentum_weights(perturbed, lookback=60, skip=5)
        pd.testing.assert_frame_equal(w_full.iloc[: cut + 1], w_pert.iloc[: cut + 1])


# ---------------------------------------------------------------------------
# walk_forward adapter
# ---------------------------------------------------------------------------
class TestStrategyFn:
    def test_adapter_matches_direct_call(self, panel: pd.DataFrame) -> None:
        direct = cross_sectional_momentum_weights(
            panel, lookback=120, skip=21, top_quantile=0.3, gross_leverage=1.5
        )
        via_fn = momentum_strategy_fn(
            {"lookback": 120, "skip": 21, "top_quantile": 0.3, "gross_leverage": 1.5}, panel
        )
        pd.testing.assert_frame_equal(direct, via_fn)

    def test_adapter_defaults(self, panel: pd.DataFrame) -> None:
        # Empty param dict → library defaults, no crash, sane shape.
        w = momentum_strategy_fn({}, panel)
        assert w.shape == panel.shape
