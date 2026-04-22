"""Concrete cost models for each supported market.

Default parameters reflect retail-level fees as of early 2026. They are
intentionally conservative — in research always use realistic, or even
pessimistic, fees, because a strategy that only works at zero cost is
not a strategy.

Available models
----------------
- :class:`AShareCostModel`   — China A-shares (stamp duty, transfer fee)
- :class:`USStockCostModel`  — US equities (per-share OR bps commission, SEC fee)
- :class:`CryptoSpotCostModel` — Crypto spot (maker/taker bps)
- :class:`CryptoPerpCostModel` — Crypto perpetual swaps (+funding)
- :class:`FuturesCostModel`  — China commodity futures (per-contract fees)

Slippage models
---------------
- :class:`FixedBpsSlippage`         — constant bps
- :class:`HalfSpreadSlippage`       — half the quoted spread
- :class:`SquareRootImpactSlippage` — volume-proportional sqrt impact
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_lucky.costs.base import (
    BPS_PER_UNIT,
    CostBreakdown,
    CostModel,
    Side,
    SlippageModel,
    Trade,
)


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedBpsSlippage(SlippageModel):
    """Constant basis-points slippage regardless of trade size.

    Simple and conservative — good default for single-name equity
    research where no volume profile is available.
    """

    bps: float = 5.0
    name: str = "fixed_bps"

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError(f"FixedBpsSlippage.bps must be >= 0, got {self.bps}")

    def estimate(self, trade: Trade) -> float:
        return trade.notional * self.bps / BPS_PER_UNIT


@dataclass(frozen=True)
class HalfSpreadSlippage(SlippageModel):
    """Assume you cross the spread; cost == half the spread in bps."""

    spread_bps: float = 10.0
    name: str = "half_spread"

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError(f"HalfSpreadSlippage.spread_bps must be >= 0, got {self.spread_bps}")

    def estimate(self, trade: Trade) -> float:
        return trade.notional * (self.spread_bps / 2.0) / BPS_PER_UNIT


@dataclass(frozen=True)
class SquareRootImpactSlippage(SlippageModel):
    """Square-root market impact.

    Impact (bps) = ``coefficient * sqrt(participation_rate)`` where
    ``participation_rate = quantity / adv``. If ``adv`` is not supplied
    the model returns 0 — we never silently synthesise a volume number.

    References: Almgren et al. (2005), "Direct Estimation of Equity Market Impact".
    """

    coefficient_bps: float = 10.0
    adv: float | None = None  # average daily volume in same units as trade.quantity
    name: str = "sqrt_impact"

    def __post_init__(self) -> None:
        if self.coefficient_bps < 0:
            raise ValueError(
                f"SquareRootImpactSlippage.coefficient_bps must be >= 0, got {self.coefficient_bps}"
            )
        if self.adv is not None and self.adv <= 0:
            raise ValueError(f"adv must be > 0 when supplied, got {self.adv}")

    def estimate(self, trade: Trade) -> float:
        if self.adv is None:
            return 0.0
        participation = trade.quantity / self.adv
        impact_bps = self.coefficient_bps * math.sqrt(participation)
        return trade.notional * impact_bps / BPS_PER_UNIT


# ---------------------------------------------------------------------------
# A-share
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AShareCostModel(CostModel):
    """China A-share cost model (Shanghai + Shenzhen).

    Components (default values reflect typical retail 2026 levels):

    - **Commission**: broker fee, ``commission_bps`` of notional with a
      minimum of ``min_commission`` CNY per trade. Default 2.5 bps.
    - **Stamp duty**: 1 bp on sells only (Ministry of Finance).
    - **Transfer fee**: 0.2 bp on both sides (SSE/SZSE combined).
    - **Slippage**: delegated to a :class:`SlippageModel`.

    STAR board / BSE / ChiNext share this schedule; exceptions for very
    small caps can be handled by passing a different slippage model.
    """

    commission_bps: float = 2.5
    min_commission: float = 5.0
    stamp_duty_bps: float = 10.0  # sell side only
    transfer_fee_bps: float = 0.2  # both sides
    slippage: SlippageModel | None = None
    name: str = "a_share"

    def __post_init__(self) -> None:
        for attr in ("commission_bps", "min_commission", "stamp_duty_bps", "transfer_fee_bps"):
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"AShareCostModel.{attr} must be >= 0, got {value}")

    def estimate(self, trade: Trade) -> CostBreakdown:
        notional = trade.notional

        commission = max(notional * self.commission_bps / BPS_PER_UNIT, self.min_commission)
        stamp_duty = (
            notional * self.stamp_duty_bps / BPS_PER_UNIT if trade.side is Side.SELL else 0.0
        )
        transfer_fee = notional * self.transfer_fee_bps / BPS_PER_UNIT
        slippage = self.slippage.estimate(trade) if self.slippage is not None else 0.0

        notes: list[str] = []
        if self.slippage is None:
            notes.append("no slippage model (zero slippage assumed)")

        return CostBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            slippage=slippage,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# US equities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class USStockCostModel(CostModel):
    """US equity cost model with a pluggable commission structure.

    Commission can be expressed two ways, selected by
    :attr:`commission_mode`:

    - ``"per_share"``: ``commission_per_share`` USD per share, capped at
      a % of notional — matches IBKR Fixed, Schwab etc.
    - ``"bps"``: ``commission_bps`` of notional — matches zero-commission
      retail brokers when set to 0.

    Regulatory fees on **sells only** (SEC + FINRA TAF) are always
    applied — these are small but real.
    """

    commission_mode: str = "bps"
    commission_bps: float = 0.0  # retail zero-commission baseline
    commission_per_share: float = 0.005  # IBKR Fixed tier
    commission_cap_pct: float = 0.01  # cap per-share commission at 1% notional
    min_commission: float = 0.0
    sec_fee_bps: float = 0.278  # 27.8 bps / 10000 = $0.0000278 / $1 notional (sells)
    finra_taf_per_share: float = 0.000166  # current FINRA TAF (sells)
    slippage: SlippageModel | None = None
    name: str = "us_stock"

    def __post_init__(self) -> None:
        if self.commission_mode not in {"bps", "per_share"}:
            raise ValueError(
                f"commission_mode must be 'bps' or 'per_share', got {self.commission_mode!r}"
            )
        for attr in (
            "commission_bps",
            "commission_per_share",
            "commission_cap_pct",
            "min_commission",
            "sec_fee_bps",
            "finra_taf_per_share",
        ):
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"USStockCostModel.{attr} must be >= 0, got {value}")

    def _commission(self, trade: Trade) -> float:
        if self.commission_mode == "bps":
            base = trade.notional * self.commission_bps / BPS_PER_UNIT
        else:
            base = trade.quantity * self.commission_per_share
            cap = trade.notional * self.commission_cap_pct
            base = min(base, cap)
        return max(base, self.min_commission)

    def _regulatory_fees(self, trade: Trade) -> float:
        if trade.side is not Side.SELL:
            return 0.0
        sec = trade.notional * self.sec_fee_bps / BPS_PER_UNIT
        taf = trade.quantity * self.finra_taf_per_share
        return sec + taf

    def estimate(self, trade: Trade) -> CostBreakdown:
        commission = self._commission(trade)
        regulatory = self._regulatory_fees(trade)
        slippage = self.slippage.estimate(trade) if self.slippage is not None else 0.0

        notes: list[str] = []
        if self.slippage is None:
            notes.append("no slippage model (zero slippage assumed)")

        return CostBreakdown(
            commission=commission,
            other=regulatory,  # SEC + FINRA fees
            slippage=slippage,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Crypto spot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CryptoSpotCostModel(CostModel):
    """Crypto spot cost model.

    Exchanges almost universally use a maker/taker fee schedule
    expressed in bps. Defaults match Binance retail tier 0 (10 bps taker,
    10 bps maker, no BNB discount).
    """

    taker_bps: float = 10.0
    maker_bps: float = 10.0
    is_maker: bool = False  # default to taker — the pessimistic assumption
    slippage: SlippageModel | None = None
    name: str = "crypto_spot"

    def __post_init__(self) -> None:
        for attr in ("taker_bps", "maker_bps"):
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"CryptoSpotCostModel.{attr} must be >= 0, got {value}")

    def estimate(self, trade: Trade) -> CostBreakdown:
        bps = self.maker_bps if self.is_maker else self.taker_bps
        commission = trade.notional * bps / BPS_PER_UNIT
        slippage = self.slippage.estimate(trade) if self.slippage is not None else 0.0
        return CostBreakdown(
            commission=commission,
            slippage=slippage,
            notes=(f"{'maker' if self.is_maker else 'taker'} fill at {bps:.2f} bps",),
        )


# ---------------------------------------------------------------------------
# Crypto perpetual swaps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CryptoPerpCostModel(CostModel):
    """Crypto perpetual swap cost model with optional funding cost.

    ``funding_rate`` is the *realised* average rate per 8-hour interval,
    expressed as a fraction (0.0001 == 1 bp). Set to 0 when comparing
    strategies that hedge funding separately.

    ``hold_intervals`` lets the caller charge funding for a specific
    holding window — typical perps settle every 8 hours, so a 24-hour
    hold pays 3 intervals. For a single fill this is usually 0 (funding
    accrues over holding, not at execution).
    """

    taker_bps: float = 5.0
    maker_bps: float = 2.0
    is_maker: bool = False
    funding_rate: float = 0.0
    hold_intervals: int = 0
    slippage: SlippageModel | None = None
    name: str = "crypto_perp"

    def __post_init__(self) -> None:
        if self.taker_bps < 0:
            raise ValueError(f"taker_bps must be >= 0, got {self.taker_bps}")
        if self.maker_bps < 0:
            raise ValueError(f"maker_bps must be >= 0, got {self.maker_bps}")
        if self.hold_intervals < 0:
            raise ValueError(f"hold_intervals must be >= 0, got {self.hold_intervals}")

    def estimate(self, trade: Trade) -> CostBreakdown:
        bps = self.maker_bps if self.is_maker else self.taker_bps
        commission = trade.notional * bps / BPS_PER_UNIT

        # Funding is paid by longs to shorts when rate > 0 (and vice versa).
        # For cost estimation we charge the magnitude — strategy direction
        # determines the sign at portfolio level, not here.
        funding = abs(self.funding_rate) * self.hold_intervals * trade.notional
        slippage = self.slippage.estimate(trade) if self.slippage is not None else 0.0

        return CostBreakdown(
            commission=commission,
            funding=funding,
            slippage=slippage,
            notes=(
                f"{'maker' if self.is_maker else 'taker'} fill, "
                f"{self.hold_intervals} funding intervals",
            ),
        )


# ---------------------------------------------------------------------------
# China commodity futures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FuturesCostModel(CostModel):
    """China commodity futures cost model.

    Most CFFEX/SHFE/DCE/CZCE contracts charge either a fixed amount per
    contract OR a bps of notional — selected via :attr:`mode`. An extra
    ``close_today_multiplier`` captures the "closing today's position"
    surcharge which is 2x–6x normal on volatile contracts.
    """

    mode: str = "per_contract"
    commission_per_contract: float = 5.0
    commission_bps: float = 0.3
    close_today_multiplier: float = 1.0
    slippage: SlippageModel | None = None
    name: str = "futures"

    def __post_init__(self) -> None:
        if self.mode not in {"per_contract", "bps"}:
            raise ValueError(f"mode must be 'per_contract' or 'bps', got {self.mode!r}")
        for attr in ("commission_per_contract", "commission_bps", "close_today_multiplier"):
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"FuturesCostModel.{attr} must be >= 0, got {value}")

    def estimate(self, trade: Trade) -> CostBreakdown:
        base = (
            trade.quantity * self.commission_per_contract
            if self.mode == "per_contract"
            else trade.notional * self.commission_bps / BPS_PER_UNIT
        )
        commission = base * self.close_today_multiplier
        slippage = self.slippage.estimate(trade) if self.slippage is not None else 0.0
        notes: list[str] = []
        if self.close_today_multiplier != 1.0:
            notes.append(f"close-today multiplier {self.close_today_multiplier}x applied")
        return CostBreakdown(
            commission=commission,
            slippage=slippage,
            notes=tuple(notes),
        )
