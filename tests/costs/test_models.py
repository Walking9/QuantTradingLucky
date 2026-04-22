"""Tests for concrete cost models + slippage models."""

from __future__ import annotations

import pytest

from quant_lucky.costs import (
    AShareCostModel,
    CryptoPerpCostModel,
    CryptoSpotCostModel,
    FixedBpsSlippage,
    FuturesCostModel,
    HalfSpreadSlippage,
    Side,
    SquareRootImpactSlippage,
    Trade,
    USStockCostModel,
)


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------
class TestFixedBpsSlippage:
    def test_estimate(self) -> None:
        s = FixedBpsSlippage(bps=5.0)
        trade = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        # 5 bps on 10,000 notional = 5
        assert s.estimate(trade) == pytest.approx(5.0)

    def test_zero_bps_costs_zero(self) -> None:
        s = FixedBpsSlippage(bps=0.0)
        trade = Trade(symbol="X", side=Side.BUY, quantity=10, price=100.0)
        assert s.estimate(trade) == 0.0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            FixedBpsSlippage(bps=-1.0)


class TestHalfSpreadSlippage:
    def test_estimate_halves_spread(self) -> None:
        s = HalfSpreadSlippage(spread_bps=20.0)
        trade = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        # half of 20 bps = 10 bps on 10,000 = 10
        assert s.estimate(trade) == pytest.approx(10.0)


class TestSquareRootImpactSlippage:
    def test_zero_without_adv(self) -> None:
        s = SquareRootImpactSlippage(coefficient_bps=10.0)
        trade = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        assert s.estimate(trade) == 0.0

    def test_impact_scales_with_sqrt_participation(self) -> None:
        s = SquareRootImpactSlippage(coefficient_bps=10.0, adv=10_000)
        trade_small = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        trade_large = Trade(symbol="X", side=Side.BUY, quantity=400, price=100.0)
        small = s.estimate(trade_small)
        large = s.estimate(trade_large)
        # quantity 4x -> participation 4x -> sqrt(4)==2 -> impact 2x
        # (notional also 4x so total cost is 8x)
        assert large / small == pytest.approx(8.0)

    def test_negative_coefficient_rejected(self) -> None:
        with pytest.raises(ValueError):
            SquareRootImpactSlippage(coefficient_bps=-1.0)

    def test_zero_adv_rejected(self) -> None:
        with pytest.raises(ValueError):
            SquareRootImpactSlippage(coefficient_bps=1.0, adv=0)


# ---------------------------------------------------------------------------
# A-share
# ---------------------------------------------------------------------------
class TestAShareCostModel:
    def test_buy_no_stamp_duty(self) -> None:
        model = AShareCostModel()
        trade = Trade(symbol="600519.SH", side=Side.BUY, quantity=100, price=1800.0)
        bd = model.estimate(trade)
        assert bd.stamp_duty == 0.0
        assert bd.commission > 0.0
        assert bd.transfer_fee > 0.0

    def test_sell_has_stamp_duty(self) -> None:
        model = AShareCostModel()
        trade = Trade(symbol="600519.SH", side=Side.SELL, quantity=100, price=1800.0)
        bd = model.estimate(trade)
        notional = trade.notional
        # default 10 bps on sell
        assert bd.stamp_duty == pytest.approx(notional * 10.0 / 10_000.0)

    def test_min_commission_applied(self) -> None:
        model = AShareCostModel(commission_bps=2.5, min_commission=5.0)
        # 2.5 bps on 1,000 notional = 0.25 -> floored to 5.0
        trade = Trade(symbol="X", side=Side.BUY, quantity=10, price=100.0)
        bd = model.estimate(trade)
        assert bd.commission == pytest.approx(5.0)

    def test_slippage_integrated(self) -> None:
        model = AShareCostModel(slippage=FixedBpsSlippage(bps=3.0))
        trade = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        bd = model.estimate(trade)
        # 3 bps of 10,000
        assert bd.slippage == pytest.approx(3.0)
        assert bd.total > bd.commission + bd.transfer_fee

    def test_no_slippage_flag(self) -> None:
        model = AShareCostModel()
        trade = Trade(symbol="X", side=Side.BUY, quantity=100, price=100.0)
        bd = model.estimate(trade)
        assert bd.slippage == 0.0
        assert any("slippage" in note for note in bd.notes)

    def test_negative_parameter_rejected(self) -> None:
        with pytest.raises(ValueError):
            AShareCostModel(commission_bps=-1.0)


# ---------------------------------------------------------------------------
# US equities
# ---------------------------------------------------------------------------
class TestUSStockCostModel:
    def test_bps_commission(self) -> None:
        model = USStockCostModel(commission_mode="bps", commission_bps=5.0)
        trade = Trade(symbol="AAPL", side=Side.BUY, quantity=100, price=200.0)
        bd = model.estimate(trade)
        # 5 bps on 20_000 = 10
        assert bd.commission == pytest.approx(10.0)
        # Buy -> no SEC fee
        assert bd.other == 0.0

    def test_per_share_commission_capped(self) -> None:
        # Per-share 0.005 on 10,000 shares = 50
        # Cap 1% of notional (0.01 * 10_000 * 0.01 == 1.0) -> commission floored to 1.0
        model = USStockCostModel(
            commission_mode="per_share",
            commission_per_share=0.005,
            commission_cap_pct=0.01,
        )
        trade = Trade(symbol="PENNY", side=Side.BUY, quantity=10_000, price=0.01)
        bd = model.estimate(trade)
        assert bd.commission == pytest.approx(1.0)

    def test_sell_regulatory_fees(self) -> None:
        model = USStockCostModel()
        trade = Trade(symbol="AAPL", side=Side.SELL, quantity=100, price=200.0)
        bd = model.estimate(trade)
        assert bd.other > 0.0  # SEC + FINRA TAF on sells only

    def test_buy_no_regulatory_fees(self) -> None:
        model = USStockCostModel()
        trade = Trade(symbol="AAPL", side=Side.BUY, quantity=100, price=200.0)
        bd = model.estimate(trade)
        assert bd.other == 0.0

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="commission_mode"):
            USStockCostModel(commission_mode="strange")

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            USStockCostModel(commission_bps=-1.0)


# ---------------------------------------------------------------------------
# Crypto spot
# ---------------------------------------------------------------------------
class TestCryptoSpotCostModel:
    def test_taker_default(self) -> None:
        model = CryptoSpotCostModel(taker_bps=10.0, maker_bps=5.0)
        trade = Trade(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, price=60_000.0)
        bd = model.estimate(trade)
        # 10 bps on 60_000
        assert bd.commission == pytest.approx(60.0)

    def test_maker_flag_reduces_fee(self) -> None:
        maker = CryptoSpotCostModel(taker_bps=10.0, maker_bps=5.0, is_maker=True)
        taker = CryptoSpotCostModel(taker_bps=10.0, maker_bps=5.0, is_maker=False)
        trade = Trade(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, price=60_000.0)
        assert maker.estimate(trade).commission < taker.estimate(trade).commission

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            CryptoSpotCostModel(taker_bps=-1.0)


# ---------------------------------------------------------------------------
# Crypto perpetual
# ---------------------------------------------------------------------------
class TestCryptoPerpCostModel:
    def test_no_funding_when_intervals_zero(self) -> None:
        model = CryptoPerpCostModel(funding_rate=0.001, hold_intervals=0)
        trade = Trade(symbol="BTC/USDT:USDT", side=Side.BUY, quantity=1.0, price=60_000.0)
        bd = model.estimate(trade)
        assert bd.funding == 0.0

    def test_funding_accrues_with_intervals(self) -> None:
        # 1 bp per 8-hour interval, 3 intervals, 60_000 notional
        # funding = 0.0001 * 3 * 60_000 = 18
        model = CryptoPerpCostModel(funding_rate=0.0001, hold_intervals=3)
        trade = Trade(symbol="BTC/USDT:USDT", side=Side.BUY, quantity=1.0, price=60_000.0)
        bd = model.estimate(trade)
        assert bd.funding == pytest.approx(18.0)

    def test_funding_absolute_value(self) -> None:
        # Negative funding is also a cost magnitude here.
        model = CryptoPerpCostModel(funding_rate=-0.0005, hold_intervals=2)
        trade = Trade(symbol="BTC/USDT:USDT", side=Side.BUY, quantity=1.0, price=60_000.0)
        bd = model.estimate(trade)
        assert bd.funding > 0

    def test_negative_intervals_rejected(self) -> None:
        with pytest.raises(ValueError):
            CryptoPerpCostModel(hold_intervals=-1)


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------
class TestFuturesCostModel:
    def test_per_contract_mode(self) -> None:
        model = FuturesCostModel(mode="per_contract", commission_per_contract=5.0)
        trade = Trade(symbol="IF2406", side=Side.BUY, quantity=3, price=3_800.0)
        bd = model.estimate(trade)
        assert bd.commission == pytest.approx(15.0)

    def test_bps_mode(self) -> None:
        model = FuturesCostModel(mode="bps", commission_bps=0.3)
        # 0.3 bps on notional of 10_000 => 0.3
        trade = Trade(symbol="IF2406", side=Side.BUY, quantity=1, price=10_000.0)
        bd = model.estimate(trade)
        assert bd.commission == pytest.approx(0.3)

    def test_close_today_multiplier(self) -> None:
        base = FuturesCostModel(mode="per_contract", commission_per_contract=5.0)
        punitive = FuturesCostModel(
            mode="per_contract", commission_per_contract=5.0, close_today_multiplier=6.0
        )
        trade = Trade(symbol="rb2406", side=Side.SELL, quantity=2, price=3_500.0)
        assert punitive.estimate(trade).commission == pytest.approx(
            base.estimate(trade).commission * 6.0
        )
        assert any("close-today" in note for note in punitive.estimate(trade).notes)

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            FuturesCostModel(mode="unknown")
