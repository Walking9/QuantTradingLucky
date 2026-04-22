"""Tests for the Trade / Side / CostBreakdown primitives."""

from __future__ import annotations

import dataclasses

import pytest

from quant_lucky.costs import CostBreakdown, Side, Trade


class TestSide:
    def test_values(self) -> None:
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"


class TestTrade:
    def test_notional(self) -> None:
        trade = Trade(symbol="AAPL", side=Side.BUY, quantity=100, price=150.0)
        assert trade.notional == pytest.approx(15_000.0)

    @pytest.mark.parametrize("quantity", [0, -1])
    def test_non_positive_quantity_rejected(self, quantity: float) -> None:
        with pytest.raises(ValueError, match="quantity"):
            Trade(symbol="X", side=Side.BUY, quantity=quantity, price=100.0)

    @pytest.mark.parametrize("price", [0, -5.0])
    def test_non_positive_price_rejected(self, price: float) -> None:
        with pytest.raises(ValueError, match="price"):
            Trade(symbol="X", side=Side.BUY, quantity=10, price=price)

    def test_trade_is_frozen(self) -> None:
        trade = Trade(symbol="X", side=Side.BUY, quantity=1, price=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            trade.quantity = 2  # type: ignore[misc]


class TestCostBreakdown:
    def test_total_sums_all_components(self) -> None:
        bd = CostBreakdown(
            commission=1.0,
            stamp_duty=2.0,
            transfer_fee=3.0,
            slippage=4.0,
            impact=5.0,
            funding=6.0,
            other=7.0,
        )
        assert bd.total == pytest.approx(28.0)

    def test_default_is_zero(self) -> None:
        bd = CostBreakdown()
        assert bd.total == 0.0

    def test_negative_components_rejected(self) -> None:
        with pytest.raises(ValueError, match="commission"):
            CostBreakdown(commission=-1.0)

    def test_bps_of_notional(self) -> None:
        bd = CostBreakdown(commission=10.0)
        assert bd.bps_of(10_000.0) == pytest.approx(10.0)

    def test_bps_of_requires_positive(self) -> None:
        with pytest.raises(ValueError, match="notional"):
            CostBreakdown(commission=1.0).bps_of(0.0)
