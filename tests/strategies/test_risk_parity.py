"""Tests for the risk-parity (inverse-vol) allocation and its benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.risk_parity import (
    equal_weight_weights,
    inverse_vol_weights,
    rebalance_dates,
    risk_parity_strategy_fn,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_price_panel(n_assets=4, n_days=400, seed=5)


class TestRebalanceDates:
    def test_daily_returns_full_index(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=50)
        assert rebalance_dates(idx, "D").equals(idx)

    def test_monthly_picks_last_trading_day(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=90)  # ~Jan–Apr 2021
        rb = rebalance_dates(idx, "M")
        # One rebalance per distinct (year, month) in the index.
        assert len(rb) == len({(d.year, d.month) for d in idx})
        # Each rebalance date is the max trading day of its month.
        for d in rb:
            same_month = idx[(idx.year == d.year) & (idx.month == d.month)]
            assert d == same_month.max()

    def test_weekly_more_frequent_than_monthly(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=120)
        assert len(rebalance_dates(idx, "W")) > len(rebalance_dates(idx, "M"))

    def test_aliases(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=60)
        assert rebalance_dates(idx, "MONTHLY").equals(rebalance_dates(idx, "M"))
        assert rebalance_dates(idx, "weekly").equals(rebalance_dates(idx, "W"))

    def test_bad_freq(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=10)
        with pytest.raises(ValueError, match="freq"):
            rebalance_dates(idx, "Q")


class TestInverseVol:
    def test_validation(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="empty"):
            inverse_vol_weights(pd.DataFrame())
        with pytest.raises(ValueError, match="vol_window"):
            inverse_vol_weights(panel, vol_window=1)
        with pytest.raises(ValueError, match="gross_leverage"):
            inverse_vol_weights(panel, gross_leverage=0.0)

    def test_weights_non_negative_and_sum_to_gross(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=40, rebalance="M", gross_leverage=1.0)
        assert (w >= -1e-12).all().all()
        invested = w.sum(axis=1)
        assert np.allclose(invested[invested > 0], 1.0)

    def test_lower_vol_gets_higher_weight(self) -> None:
        # Construct two assets: one calm, one volatile, same drift.
        dates = pd.bdate_range("2021-01-01", periods=200)
        rng = np.random.default_rng(0)
        calm = 100 * np.cumprod(1 + rng.normal(0.0, 0.005, 200))
        wild = 100 * np.cumprod(1 + rng.normal(0.0, 0.03, 200))
        prices = pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates)
        w = inverse_vol_weights(prices, vol_window=40, rebalance="D")
        last = w.iloc[-1]
        assert last["CALM"] > last["WILD"]

    def test_weights_only_change_on_rebalance_dates(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=40, rebalance="M")
        rb = set(rebalance_dates(panel.index, "M"))
        changed = w.index[w.diff().abs().sum(axis=1) > 1e-12]
        # Every weight change must land on a rebalance date.
        assert set(changed).issubset(rb)

    def test_warmup_is_flat(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=63, rebalance="M")
        # Before the vol window fills there is nothing to size on.
        assert w.iloc[:63].to_numpy().sum() == 0.0

    def test_no_look_ahead(self, panel: pd.DataFrame) -> None:
        cut = 250
        w_full = inverse_vol_weights(panel, vol_window=40, rebalance="M")
        perturbed = panel.copy()
        perturbed.iloc[cut + 1 :] *= 1.4
        w_pert = inverse_vol_weights(perturbed, vol_window=40, rebalance="M")
        pd.testing.assert_frame_equal(w_full.iloc[: cut + 1], w_pert.iloc[: cut + 1])


class TestEqualWeight:
    def test_equal_split(self, panel: pd.DataFrame) -> None:
        w = equal_weight_weights(panel, rebalance="M", gross_leverage=1.0)
        invested = w.sum(axis=1)
        assert np.allclose(invested[invested > 0], 1.0)
        # On a fully-present panel each of the 4 names gets 0.25.
        last = w.iloc[-1]
        assert np.allclose(last, 0.25)

    def test_validation(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="empty"):
            equal_weight_weights(pd.DataFrame())
        with pytest.raises(ValueError, match="gross_leverage"):
            equal_weight_weights(panel, gross_leverage=-1.0)


class TestStrategyFn:
    def test_adapter_matches_direct(self, panel: pd.DataFrame) -> None:
        direct = inverse_vol_weights(panel, vol_window=30, rebalance="W", gross_leverage=1.0)
        via = risk_parity_strategy_fn({"vol_window": 30, "rebalance": "W"}, panel)
        pd.testing.assert_frame_equal(direct, via)
