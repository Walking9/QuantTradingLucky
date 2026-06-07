"""Risk-parity (inverse-volatility) multi-asset allocation.

The idea behind risk parity (Qian 2005; Asness, Frazzini & Pedersen
2012) is to let each asset contribute *equal risk* to the portfolio
rather than equal *capital*. A 60/40 stock/bond book is ~90 % equity
risk; risk parity fixes that by down-weighting the volatile leg.

For a small basket of weakly-correlated cross-market assets
(BTC / SPY / CSI-300) the **inverse-volatility** approximation —
``w_i ∝ 1 / σ_i`` — is within a hair of the full equal-risk-contribution
(ERC) solution while avoiding the ill-conditioned covariance inversion
that bites on short samples. We use it as the workhorse here; a proper
ERC optimiser (``cvxpy``) is a natural M7 extension once the multi-factor
covariance machinery lands.

Rebalancing
-----------
Target weights are recomputed only on rebalance dates (month- or
week-end) and **held flat in between** by forward-filling. This matters
for cost accounting: the :class:`~quant_lucky.backtest.VectorEngine`
reads an unchanged weight row as "no trade, no cost", so a monthly
rebalance pays cost ~12×/year, not daily. (Holding *weights* flat ignores
intra-period drift — a deliberate, documented approximation appropriate
for a low-frequency allocation backtest; a drift-aware NAV mode is left
to a later iteration.)

No-look-ahead contract
----------------------
A rebalance on date ``t`` sizes positions from the trailing volatility
ending at ``t`` (data ≤ t). The engine shifts forward to earn
``(t, t+1]``. Do not pre-shift.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "equal_weight_weights",
    "hold_between_rebalances",
    "inverse_vol_weights",
    "rebalance_dates",
    "risk_parity_strategy_fn",
]

_FREQ_ALIASES = {
    "D": "D",
    "DAILY": "D",
    "W": "W",
    "WEEKLY": "W",
    "M": "M",
    "MONTHLY": "M",
}


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Return the subset of ``index`` that are rebalance points.

    ``freq='D'`` rebalances every bar; ``'W'`` / ``'M'`` pick the **last
    available trading day** of each ISO-week / calendar-month present in
    the index (so a month that ends on a weekend rebalances on the last
    actual bar, not a non-existent calendar date).

    Args:
        index: The trading-date index of the price panel.
        freq: One of ``D/W/M`` (case-insensitive; ``DAILY/WEEKLY/MONTHLY``
            accepted).

    Returns:
        A sorted :class:`~pandas.DatetimeIndex` subset of ``index``.
    """
    key = _FREQ_ALIASES.get(freq.upper())
    if key is None:
        raise ValueError(f"freq must be one of D/W/M, got {freq!r}")
    if len(index) == 0:
        return index
    if key == "D":
        return index

    period = "W" if key == "W" else "M"
    grouped = pd.Series(index, index=index).groupby(index.to_period(period))
    # Last actual trading bar within each period.
    last_bars = grouped.last()
    return pd.DatetimeIndex(sorted(last_bars.to_numpy()))


def hold_between_rebalances(target: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Keep ``target`` weights only on rebalance dates, hold flat between.

    Rows on non-rebalance dates are blanked and forward-filled, so the
    portfolio is rebalanced on the last trading bar of each period
    (``freq`` ∈ ``D/W/M``) and drifts unchanged in weight-space in
    between. The engine reads an unchanged weight row as "no trade, no
    cost", so this is what makes a monthly strategy pay cost ~12×/year
    rather than daily. Forward-filling past weights never looks ahead.

    This is the seam shared by :func:`inverse_vol_weights`,
    :func:`equal_weight_weights`, and the monthly cross-sectional
    momentum book — any daily target can be down-sampled to a periodic
    rebalance with it.
    """
    rb = rebalance_dates(pd.DatetimeIndex(target.index), freq)
    held = target.copy()
    held.loc[~target.index.isin(rb)] = np.nan
    return held.ffill().fillna(0.0)


def inverse_vol_weights(
    prices: pd.DataFrame,
    *,
    vol_window: int = 63,
    rebalance: str = "M",
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Inverse-volatility (naive risk-parity) weights, long-only.

    Args:
        prices: Wide ``date × asset`` close prices.
        vol_window: Trailing window (bars) for each asset's realised
            volatility. ``63`` ≈ one quarter.
        rebalance: Rebalance frequency (``D/W/M``). Weights are held flat
            between rebalance dates.
        gross_leverage: Total long exposure ``Σ w`` on each rebalance
            (``1.0`` = fully invested, no leverage).

    Returns:
        Wide ``date × asset`` weights (non-negative, sum to
        ``gross_leverage`` on populated dates, ``0`` before the vol
        window fills).

    Raises:
        ValueError: on invalid parameters or an empty panel.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if vol_window < 2:
        raise ValueError(f"vol_window must be >= 2, got {vol_window}")
    if gross_leverage <= 0:
        raise ValueError(f"gross_leverage must be > 0, got {gross_leverage}")

    px = prices.sort_index()
    rets = px.pct_change(fill_method=None)
    vol = rets.rolling(vol_window, min_periods=vol_window).std()

    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    row_sum = inv_vol.sum(axis=1, min_count=1).replace(0.0, np.nan)
    target = inv_vol.div(row_sum, axis=0) * gross_leverage

    weights = hold_between_rebalances(target, rebalance)
    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights


def equal_weight_weights(
    prices: pd.DataFrame,
    *,
    rebalance: str = "M",
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Equal-weight benchmark (the risk-parity null hypothesis).

    Each asset that has a valid price on a rebalance date receives the
    same weight; total exposure is ``gross_leverage``. Held flat between
    rebalances, mirroring :func:`inverse_vol_weights` so the two are a
    fair comparison.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if gross_leverage <= 0:
        raise ValueError(f"gross_leverage must be > 0, got {gross_leverage}")

    px = prices.sort_index()
    present = px.notna().astype(float)
    n_present = present.sum(axis=1).replace(0.0, np.nan)
    target = present.div(n_present, axis=0) * gross_leverage

    weights = hold_between_rebalances(target, rebalance)
    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights


def risk_parity_strategy_fn(param: Mapping[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Adapter matching the :data:`~quant_lucky.backtest.validation.StrategyFn`
    contract for :func:`~quant_lucky.backtest.walk_forward`.

    Recognised keys: ``vol_window``, ``rebalance``, ``gross_leverage``.
    """
    return inverse_vol_weights(
        prices,
        vol_window=int(param.get("vol_window", 63)),
        rebalance=str(param.get("rebalance", "M")),
        gross_leverage=float(param.get("gross_leverage", 1.0)),
    )
