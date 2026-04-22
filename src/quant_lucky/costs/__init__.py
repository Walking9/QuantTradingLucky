"""Transaction cost models.

Covers:
- Commission (A-share, US, crypto — different structures).
- Stamp duty / transfer fees (A-share specifics).
- Slippage models (fixed bps, half-spread, square-root impact).
- Funding-rate cost for perpetual swaps.
- Annualised turnover-based cost drag estimation.

Realistic cost modelling is the difference between a strategy that works
and one that bleeds. Default cost assumptions err on the pessimistic
side; any "costless" backtest should be explicitly flagged.

Typical usage
-------------
::

    from quant_lucky.costs import (
        AShareCostModel,
        AnnualCostCalculator,
        FixedBpsSlippage,
        Side,
        Trade,
    )

    model = AShareCostModel(slippage=FixedBpsSlippage(bps=5))
    trade = Trade(symbol="600519.SH", side=Side.BUY, quantity=100, price=1800.0)
    breakdown = model.estimate(trade)
    print(breakdown.total, breakdown.bps_of(trade.notional))

    calc = AnnualCostCalculator(model)
    report = calc.annual_cost_by_frequency(turnover_per_rebalance=0.20, frequency="monthly")
    print(f"Annual drag: {report.annual_cost_bps:.1f} bps")
"""

from __future__ import annotations

from quant_lucky.costs.base import (
    BPS_PER_UNIT,
    CostBreakdown,
    CostModel,
    Side,
    SlippageModel,
    Trade,
)
from quant_lucky.costs.calculator import (
    AnnualCostCalculator,
    AnnualCostEstimate,
    RebalanceFrequency,
)
from quant_lucky.costs.models import (
    AShareCostModel,
    CryptoPerpCostModel,
    CryptoSpotCostModel,
    FixedBpsSlippage,
    FuturesCostModel,
    HalfSpreadSlippage,
    SquareRootImpactSlippage,
    USStockCostModel,
)

__all__ = [
    # Base
    "BPS_PER_UNIT",
    # Models
    "AShareCostModel",
    # Calculator
    "AnnualCostCalculator",
    "AnnualCostEstimate",
    "CostBreakdown",
    "CostModel",
    "CryptoPerpCostModel",
    "CryptoSpotCostModel",
    # Slippage
    "FixedBpsSlippage",
    "FuturesCostModel",
    "HalfSpreadSlippage",
    "RebalanceFrequency",
    "Side",
    "SlippageModel",
    "SquareRootImpactSlippage",
    "Trade",
    "USStockCostModel",
]
