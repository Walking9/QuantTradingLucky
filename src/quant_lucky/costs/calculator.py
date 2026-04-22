"""Annualized cost calculator.

Answers the practical question *"how much does trading cost me per
year?"* given a cost model plus two strategy-level parameters:

- **annual_turnover**: by default the two-way turnover — sum of |Δw|
  across BOTH buys and sells per year. A portfolio that fully rotates
  once per year has ``annual_turnover == 2.0`` (100% sold + 100%
  bought). A daily mean-reversion book that rotates every day has
  ``annual_turnover ≈ 504`` (2 × 252 trading days).
- **round_trip**: when True, ``annual_turnover`` is reinterpreted as
  the count of complete round-trips per year (1 rotation == 1.0). Use
  whichever convention matches the caller's data pipeline.

The output is expressed both in bps and as a fraction of portfolio NAV
(e.g. 0.015 means trading eats 1.5% of NAV per year).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_lucky.costs.base import BPS_PER_UNIT, CostModel, Side, Trade
from quant_lucky.utils.logging import logger

RebalanceFrequency = Literal["daily", "weekly", "biweekly", "monthly", "quarterly", "annual"]

_TRADING_DAYS_PER_YEAR = 252
_FREQUENCY_REBALANCES_PER_YEAR: dict[RebalanceFrequency, float] = {
    "daily": _TRADING_DAYS_PER_YEAR,
    "weekly": 52.0,
    "biweekly": 26.0,
    "monthly": 12.0,
    "quarterly": 4.0,
    "annual": 1.0,
}


@dataclass(frozen=True)
class AnnualCostEstimate:
    """Result of an annualised cost computation."""

    annual_turnover: float
    rebalances_per_year: float
    cost_per_rebalance_bps: float
    annual_cost_bps: float
    annual_cost_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "annual_turnover": self.annual_turnover,
            "rebalances_per_year": self.rebalances_per_year,
            "cost_per_rebalance_bps": self.cost_per_rebalance_bps,
            "annual_cost_bps": self.annual_cost_bps,
            "annual_cost_fraction": self.annual_cost_fraction,
        }


class AnnualCostCalculator:
    """Translate turnover + cost model into an annualised drag estimate.

    The calculator issues a *synthetic* round-trip trade through the
    cost model using ``reference_price`` and ``reference_quantity``.
    For cost models with no flat-fee floor (crypto, US bps) this is
    purely scale-invariant. For models with a ``min_commission`` floor
    (A-share, US per-share) the reference notional MUST exceed the
    strategy's typical trade size for the bps projection to be
    meaningful — the default 100_000 reference notional covers typical
    retail trades. Pick a larger value for institutional sizes.
    """

    def __init__(
        self,
        cost_model: CostModel,
        *,
        reference_price: float = 100.0,
        reference_quantity: float = 1_000.0,
    ) -> None:
        if reference_price <= 0:
            raise ValueError(f"reference_price must be > 0, got {reference_price}")
        if reference_quantity <= 0:
            raise ValueError(f"reference_quantity must be > 0, got {reference_quantity}")
        self.cost_model = cost_model
        self.reference_price = reference_price
        self.reference_quantity = reference_quantity

    def round_trip_cost_bps(self) -> float:
        """Cost of one buy + one sell, in bps of notional."""
        buy = Trade(
            symbol="__calc__",
            side=Side.BUY,
            quantity=self.reference_quantity,
            price=self.reference_price,
        )
        sell = Trade(
            symbol="__calc__",
            side=Side.SELL,
            quantity=self.reference_quantity,
            price=self.reference_price,
        )
        buy_bps = self.cost_model.estimate_bps(buy)
        sell_bps = self.cost_model.estimate_bps(sell)
        total = buy_bps + sell_bps
        if total == 0:
            logger.warning(
                "Cost model {name} returned 0 bps round-trip — check configuration",
                name=self.cost_model.name,
            )
        return total

    def annual_cost(
        self,
        annual_turnover: float,
        *,
        round_trip: bool = False,
    ) -> AnnualCostEstimate:
        """Annualised cost drag for a given annual turnover.

        Parameters
        ----------
        annual_turnover:
            Two-way turnover by default: sum of |Δw| across both buys
            AND sells per year. A portfolio that fully rotates once
            per year has ``annual_turnover == 2.0`` (1.0 sold + 1.0
            bought). The traditional academic "annual turnover" figure
            usually matches this definition.
        round_trip:
            When True, ``annual_turnover`` is interpreted as the
            number of **complete round-trips** per year — i.e. a full
            rotation counts as 1.0 (not 2.0). The equivalence is
            ``turnover_two_way == 2 * turnover_round_trip``.
        """
        if annual_turnover < 0:
            raise ValueError(f"annual_turnover must be >= 0, got {annual_turnover}")

        round_trip_bps = self.round_trip_cost_bps()
        one_way_bps = round_trip_bps / 2.0

        if round_trip:
            # Each unit is a full round-trip (one buy + one sell).
            annual_bps = annual_turnover * round_trip_bps
        else:
            # Each unit is one side of a trade (one buy OR one sell).
            annual_bps = annual_turnover * one_way_bps

        return AnnualCostEstimate(
            annual_turnover=annual_turnover,
            rebalances_per_year=float("nan"),  # unknown unless caller supplies
            cost_per_rebalance_bps=float("nan"),
            annual_cost_bps=annual_bps,
            annual_cost_fraction=annual_bps / BPS_PER_UNIT,
        )

    def annual_cost_by_frequency(
        self,
        turnover_per_rebalance: float,
        frequency: RebalanceFrequency,
        *,
        round_trip: bool = False,
    ) -> AnnualCostEstimate:
        """Annualised cost given turnover-per-rebalance + frequency.

        More intuitive when the caller thinks in terms of "I rotate 20%
        of the book each week" rather than total annual turnover.

        The same ``round_trip`` flag as :meth:`annual_cost` applies:
        ``turnover_per_rebalance`` is interpreted as two-way notional
        by default, or as round-trip count when ``round_trip=True``.
        """
        if turnover_per_rebalance < 0:
            raise ValueError(f"turnover_per_rebalance must be >= 0, got {turnover_per_rebalance}")
        if frequency not in _FREQUENCY_REBALANCES_PER_YEAR:
            raise ValueError(
                f"frequency must be one of {list(_FREQUENCY_REBALANCES_PER_YEAR)}, "
                f"got {frequency!r}"
            )

        rebalances = _FREQUENCY_REBALANCES_PER_YEAR[frequency]
        annual_turnover = turnover_per_rebalance * rebalances

        estimate = self.annual_cost(annual_turnover, round_trip=round_trip)

        # Fill in the per-rebalance fields now that we know them.
        round_trip_bps = self.round_trip_cost_bps()
        one_way_bps = round_trip_bps / 2.0
        cost_per_rebalance_bps = turnover_per_rebalance * (
            round_trip_bps if round_trip else one_way_bps
        )

        return AnnualCostEstimate(
            annual_turnover=annual_turnover,
            rebalances_per_year=rebalances,
            cost_per_rebalance_bps=cost_per_rebalance_bps,
            annual_cost_bps=estimate.annual_cost_bps,
            annual_cost_fraction=estimate.annual_cost_fraction,
        )
