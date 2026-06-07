"""Tests for the dual-MA + volatility-filter trend strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.dual_ma import (
    dual_ma_strategy_fn,
    dual_ma_vol_filter_weights,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_price_panel(n_assets=8, n_days=400, seed=11)


def _uptrend(n: int = 200, rate: float = 0.002) -> pd.Series:
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(100 * (1 + rate) ** np.arange(n), index=dates)


def _downtrend(n: int = 200, rate: float = 0.002) -> pd.Series:
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(100 * (1 - rate) ** np.arange(n), index=dates)


class TestValidation:
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            dual_ma_vol_filter_weights(pd.DataFrame())

    def test_slow_must_exceed_fast(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="slow must be > fast"):
            dual_ma_vol_filter_weights(panel, fast=30, slow=30)

    def test_bad_fast(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fast"):
            dual_ma_vol_filter_weights(panel, fast=0, slow=10)

    def test_bad_vol_window(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="vol_window"):
            dual_ma_vol_filter_weights(panel, vol_window=1)

    def test_bad_vol_max(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="vol_max"):
            dual_ma_vol_filter_weights(panel, vol_max=-0.1)


class TestTrendLogic:
    def test_pure_uptrend_is_long(self) -> None:
        prices = pd.DataFrame({"UP": _uptrend()})
        w = dual_ma_vol_filter_weights(
            prices, fast=10, slow=30, vol_window=20, vol_max=None, gross_leverage=1.0
        )
        # Once the slow MA fills and fast>slow, the single name holds its
        # full 1/N = 1.0 budget.
        assert w["UP"].iloc[-1] == pytest.approx(1.0)
        assert w["UP"].iloc[:29].abs().sum() == 0.0  # warmup flat (slow MA fills at iloc 29)

    def test_pure_downtrend_is_cash_when_long_only(self) -> None:
        prices = pd.DataFrame({"DN": _downtrend()})
        w = dual_ma_vol_filter_weights(
            prices, fast=10, slow=30, vol_window=20, vol_max=None, long_only=True
        )
        assert w["DN"].abs().sum() == 0.0  # never holds a falling name

    def test_downtrend_is_short_when_shorting_allowed(self) -> None:
        prices = pd.DataFrame({"DN": _downtrend()})
        w = dual_ma_vol_filter_weights(
            prices, fast=10, slow=30, vol_window=20, vol_max=None, long_only=False
        )
        assert w["DN"].iloc[-1] == pytest.approx(-1.0)


class TestVolFilter:
    def test_high_vol_name_is_gated_off(self) -> None:
        # Two up-trending names; one calm, one violently noisy. The vol
        # ceiling should keep the calm one and drop the noisy one.
        dates = pd.bdate_range("2021-01-01", periods=200)
        rng = np.random.default_rng(0)
        calm = 100 * np.cumprod(1 + rng.normal(0.001, 0.005, 200))
        noisy = 100 * np.cumprod(1 + rng.normal(0.001, 0.05, 200))
        prices = pd.DataFrame({"CALM": calm, "NOISY": noisy}, index=dates)
        w = dual_ma_vol_filter_weights(
            prices, fast=10, slow=30, vol_window=20, vol_max=0.40, gross_leverage=1.0
        )
        # Budget-per-name: CALM can be held (≤ ceiling), NOISY gated to cash.
        assert w["CALM"].iloc[-1] > 0.0
        assert w["NOISY"].iloc[-1] == pytest.approx(0.0)

    def test_filter_off_holds_more(self) -> None:
        panel = synthetic_price_panel(n_assets=8, n_days=400, seed=3, idio_vol=0.5)
        gated = dual_ma_vol_filter_weights(panel, fast=10, slow=30, vol_window=20, vol_max=0.3)
        ungated = dual_ma_vol_filter_weights(panel, fast=10, slow=30, vol_window=20, vol_max=None)
        # Removing the filter can only add exposure, never remove it.
        assert ungated.abs().sum().sum() >= gated.abs().sum().sum()


class TestWeighting:
    def test_budget_per_name_leaves_cash(self, panel: pd.DataFrame) -> None:
        w = dual_ma_vol_filter_weights(panel, fast=10, slow=30, vol_window=20, normalise=False)
        gross = w.abs().sum(axis=1)
        # With cash allowed, gross is usually below full investment.
        assert gross.max() <= 1.0 + 1e-9
        assert (gross < 1.0 - 1e-9).any()

    def test_normalise_is_fully_invested_when_any_active(self, panel: pd.DataFrame) -> None:
        w = dual_ma_vol_filter_weights(
            panel, fast=10, slow=30, vol_window=20, vol_max=None, normalise=True
        )
        gross = w.abs().sum(axis=1)
        active = gross > 0
        assert np.allclose(gross[active], 1.0)


class TestNoLookAhead:
    def test_future_prices_do_not_change_past_weights(self, panel: pd.DataFrame) -> None:
        cut = 250
        w_full = dual_ma_vol_filter_weights(panel, fast=10, slow=30, vol_window=20, vol_max=0.5)
        perturbed = panel.copy()
        perturbed.iloc[cut + 1 :] *= 1.3
        w_pert = dual_ma_vol_filter_weights(perturbed, fast=10, slow=30, vol_window=20, vol_max=0.5)
        pd.testing.assert_frame_equal(w_full.iloc[: cut + 1], w_pert.iloc[: cut + 1])


class TestStrategyFn:
    def test_adapter_matches_direct(self, panel: pd.DataFrame) -> None:
        direct = dual_ma_vol_filter_weights(
            panel, fast=15, slow=45, vol_window=30, vol_max=0.5, normalise=True
        )
        via = dual_ma_strategy_fn(
            {"fast": 15, "slow": 45, "vol_window": 30, "vol_max": 0.5, "normalise": True}, panel
        )
        pd.testing.assert_frame_equal(direct, via)

    def test_adapter_handles_none_vol_max(self, panel: pd.DataFrame) -> None:
        via = dual_ma_strategy_fn({"vol_max": None}, panel)
        direct = dual_ma_vol_filter_weights(panel, vol_max=None)
        pd.testing.assert_frame_equal(direct, via)
