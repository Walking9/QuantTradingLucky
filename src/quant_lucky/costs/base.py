"""Core abstractions for transaction cost modelling.

Every backtest or paper-trading session MUST pass all fills through a
cost model. The "costless" backtest is the single biggest source of
self-deception in quant research. Defaults here err on the pessimistic
side and any zero-cost configuration is logged as a warning by the
calculator.

Concepts
--------
``Trade``
    One atomic fill (symbol + side + quantity + price). The sign of
    ``quantity`` is encoded by :class:`Side`; ``quantity`` itself is
    always non-negative. Callers converting from signed positions must
    split into a side + magnitude before instantiating.

``CostBreakdown``
    A pure-data record of *every* cost component. Components are
    reported separately so reports can attribute drag to commission vs
    slippage vs funding etc. ``total`` is always the sum — never
    pre-net; net-vs-gross decisions belong to the caller.

``CostModel``
    ABC with a single method :meth:`estimate`. Implementations must be
    pure functions of the ``Trade`` (no hidden state, no I/O), which
    makes them trivially testable and composable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class Side(StrEnum):
    """Trade direction. ``quantity`` is always >= 0; sign lives here."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Trade:
    """Immutable description of one fill.

    Attributes
    ----------
    symbol:
        Any string the caller uses to identify the instrument. The
        cost model does not interpret it except for diagnostics.
    side:
        :class:`Side.BUY` or :class:`Side.SELL`.
    quantity:
        Number of shares / contracts / coins. Must be > 0.
    price:
        Execution price, in quote currency per unit. Must be > 0.
    """

    symbol: str
    side: Side
    quantity: float
    price: float

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Trade.quantity must be > 0, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Trade.price must be > 0, got {self.price}")

    @property
    def notional(self) -> float:
        """Gross notional value of the trade (price × quantity)."""
        return self.price * self.quantity


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised trading costs for a single trade.

    All amounts are expressed in the quote currency. Values are
    non-negative regardless of side; the cost model hides the
    "who pays which fee" logic internally.
    """

    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0
    funding: float = 0.0
    other: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in (
            "commission",
            "stamp_duty",
            "transfer_fee",
            "slippage",
            "impact",
            "funding",
            "other",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"CostBreakdown.{name} must be >= 0, got {value}")

    @property
    def total(self) -> float:
        """Sum of every cost component in quote currency."""
        return (
            self.commission
            + self.stamp_duty
            + self.transfer_fee
            + self.slippage
            + self.impact
            + self.funding
            + self.other
        )

    def bps_of(self, notional: float) -> float:
        """Express the total cost as basis points of ``notional``."""
        if notional <= 0:
            raise ValueError(f"notional must be > 0, got {notional}")
        return self.total / notional * 10_000.0


class SlippageModel(ABC):
    """Abstract base for slippage/impact models.

    Slippage is kept as a separate concern from commissions so the same
    slippage assumption can be combined with any commission model (e.g.
    fixed-bps slippage on top of either A-share or US commissions).
    """

    name: str = "base"

    @abstractmethod
    def estimate(self, trade: Trade) -> float:
        """Return the slippage cost of ``trade`` in quote currency."""
        raise NotImplementedError


class CostModel(ABC):
    """Abstract base for market-specific cost models.

    Implementations compose a commission/fee structure with a
    :class:`SlippageModel`. The resulting :meth:`estimate` is a pure
    function of the trade.
    """

    name: str = "base"

    @abstractmethod
    def estimate(self, trade: Trade) -> CostBreakdown:
        """Return the full cost breakdown for ``trade``."""
        raise NotImplementedError

    def estimate_bps(self, trade: Trade) -> float:
        """Convenience: total cost as bps of notional."""
        return self.estimate(trade).bps_of(trade.notional)


BPS_PER_UNIT: float = 10_000.0
"""Basis-points scaling factor. 100 bps == 1%."""
