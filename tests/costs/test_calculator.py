"""Tests for the annualised-cost calculator."""

from __future__ import annotations

import pytest

from quant_lucky.costs import (
    AnnualCostCalculator,
    AShareCostModel,
    CryptoSpotCostModel,
    FixedBpsSlippage,
    USStockCostModel,
)


class TestRoundTripCost:
    def test_ashare_round_trip(self) -> None:
        # Commission 2.5 bps * 2 + transfer 0.2 bps * 2 + stamp 10 bps once
        # = 5 + 0.4 + 10 = 15.4 bps
        model = AShareCostModel()
        calc = AnnualCostCalculator(model)
        assert calc.round_trip_cost_bps() == pytest.approx(15.4, rel=1e-6)

    def test_crypto_spot_round_trip(self) -> None:
        # Taker 10 bps * 2 = 20 bps
        model = CryptoSpotCostModel(taker_bps=10.0)
        calc = AnnualCostCalculator(model)
        assert calc.round_trip_cost_bps() == pytest.approx(20.0)

    def test_zero_cost_model_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        model = USStockCostModel(
            commission_mode="bps",
            commission_bps=0.0,
            sec_fee_bps=0.0,
            finra_taf_per_share=0.0,
        )
        calc = AnnualCostCalculator(model)
        assert calc.round_trip_cost_bps() == 0.0
        # Just ensure the warning path does not raise — loguru routes
        # through stderr not caplog by default.


class TestAnnualCost:
    def test_negative_turnover_rejected(self) -> None:
        calc = AnnualCostCalculator(AShareCostModel())
        with pytest.raises(ValueError):
            calc.annual_cost(-0.1)

    def test_one_way_turnover_scales_linearly(self) -> None:
        # Monotonic + linear in turnover
        model = CryptoSpotCostModel(taker_bps=10.0)
        calc = AnnualCostCalculator(model)
        low = calc.annual_cost(2.0).annual_cost_bps
        high = calc.annual_cost(4.0).annual_cost_bps
        assert high == pytest.approx(2 * low)

    def test_round_trip_flag(self) -> None:
        # Round-trip turnover = 2 one-way units, so annual cost should match
        model = CryptoSpotCostModel(taker_bps=10.0)
        calc = AnnualCostCalculator(model)
        one_way = calc.annual_cost(2.0, round_trip=False).annual_cost_bps
        round_trip = calc.annual_cost(1.0, round_trip=True).annual_cost_bps
        assert one_way == pytest.approx(round_trip)


class TestAnnualCostByFrequency:
    def test_monthly_rebalance(self) -> None:
        model = CryptoSpotCostModel(taker_bps=10.0)
        calc = AnnualCostCalculator(model)
        est = calc.annual_cost_by_frequency(turnover_per_rebalance=0.20, frequency="monthly")
        assert est.rebalances_per_year == pytest.approx(12.0)
        assert est.annual_turnover == pytest.approx(2.4)
        # one-way 10 bps per unit, 2.4 units -> 24 bps
        assert est.annual_cost_bps == pytest.approx(24.0)
        assert est.annual_cost_fraction == pytest.approx(24.0 / 10_000.0)

    def test_daily_rebalance(self) -> None:
        model = CryptoSpotCostModel(taker_bps=10.0)
        calc = AnnualCostCalculator(model)
        est = calc.annual_cost_by_frequency(turnover_per_rebalance=1.0, frequency="daily")
        # 252 rebalances * 1.0 = 252 turnover * 10 bps = 2520 bps = 25.2%
        assert est.annual_cost_bps == pytest.approx(2520.0)

    def test_invalid_frequency_rejected(self) -> None:
        calc = AnnualCostCalculator(AShareCostModel())
        with pytest.raises(ValueError, match="frequency"):
            calc.annual_cost_by_frequency(0.1, frequency="fortnightly")  # type: ignore[arg-type]

    def test_negative_turnover_rejected(self) -> None:
        calc = AnnualCostCalculator(AShareCostModel())
        with pytest.raises(ValueError):
            calc.annual_cost_by_frequency(-0.1, frequency="monthly")


class TestCalculatorConstruction:
    def test_invalid_reference_price(self) -> None:
        with pytest.raises(ValueError):
            AnnualCostCalculator(AShareCostModel(), reference_price=0.0)

    def test_invalid_reference_quantity(self) -> None:
        with pytest.raises(ValueError):
            AnnualCostCalculator(AShareCostModel(), reference_quantity=-1.0)

    def test_works_with_slippage_integrated(self) -> None:
        model = AShareCostModel(slippage=FixedBpsSlippage(bps=5.0))
        calc = AnnualCostCalculator(model)
        # Base round-trip 15.4 + slippage 5*2 = 25.4 bps
        assert calc.round_trip_cost_bps() == pytest.approx(25.4, rel=1e-6)


class TestAnnualCostEstimateAsDict:
    def test_as_dict_roundtrip(self) -> None:
        calc = AnnualCostCalculator(AShareCostModel())
        est = calc.annual_cost_by_frequency(turnover_per_rebalance=0.10, frequency="weekly")
        d = est.as_dict()
        assert d["annual_turnover"] == pytest.approx(5.2)
        assert d["rebalances_per_year"] == pytest.approx(52.0)
