"""Dual moving-average crossover with a volatility filter (trend-following).

A textbook trend system (Kaufman, *Trading Systems and Methods*): hold an
asset while its **fast** moving average sits above its **slow** moving
average, step aside otherwise. On its own a dual-MA crossover whipsaws
badly in choppy, high-volatility regimes, so we add a **volatility
filter** — a name only earns a position while its realised volatility is
below a ceiling. The combination is the M6 A-share trend strategy:
trend-following names that have *also* calmed down.

Weighting is **budget-per-name**: each of the ``N`` universe members is
allocated ``gross_leverage / N`` when both gates (trend up *and* vol
below ceiling) are open, and sits in cash otherwise. Gross exposure
therefore *falls* when fewer names qualify — the defensive behaviour we
want from a filter, and the thing an "always fully invested" normaliser
would hide. Set ``normalise=True`` to instead spread a constant
``gross_leverage`` across whatever names are active.

Markets without easy shorting (A-shares for retail) motivate the
``long_only`` default: the down-trend leg is *cash*, not a short.

No-look-ahead contract
----------------------
``weights.loc[t]`` uses moving averages and a realised-vol estimate that
end at ``t`` (data ≤ t). The :class:`~quant_lucky.backtest.VectorEngine`
shifts forward one bar to earn ``(t, t+1]``. Do not pre-shift.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "dual_ma_strategy_fn",
    "dual_ma_vol_filter_weights",
]


def dual_ma_vol_filter_weights(
    prices: pd.DataFrame,
    *,
    fast: int = 20,
    slow: int = 60,
    vol_window: int = 20,
    vol_max: float | None = 0.40,
    long_only: bool = True,
    gross_leverage: float = 1.0,
    normalise: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build a dual-MA + volatility-filter trend weight book.

    Args:
        prices: Wide ``date × asset`` close prices.
        fast: Fast SMA window (bars). Must be ``>= 1``.
        slow: Slow SMA window (bars). Must be ``> fast``.
        vol_window: Window for the realised-volatility estimate (bars).
        vol_max: Annualised realised-vol ceiling. A name is gated **off**
            on any date its trailing annualised vol exceeds this. ``None``
            disables the filter (pure dual-MA crossover).
        long_only: If True the down-trend state is flat (cash). If False,
            a fast-below-slow crossover takes a short of equal size.
        gross_leverage: Target gross exposure when fully deployed.
        normalise: If False (default) each name gets a fixed
            ``gross_leverage / N`` budget and the book holds cash when
            names are gated off. If True, ``gross_leverage`` is spread
            across the active names so the book stays fully invested.
        periods_per_year: Annualisation factor for the vol estimate
            (252 equities, 365 crypto). Only affects the ``vol_max``
            comparison.

    Returns:
        Wide ``date × asset`` weights, zero where gated off or where the
        slow MA / vol estimate has not filled yet.

    Raises:
        ValueError: on invalid windows or parameters.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if fast < 1:
        raise ValueError(f"fast must be >= 1, got {fast}")
    if slow <= fast:
        raise ValueError(f"slow must be > fast, got slow={slow}, fast={fast}")
    if vol_window < 2:
        raise ValueError(f"vol_window must be >= 2, got {vol_window}")
    if vol_max is not None and vol_max <= 0:
        raise ValueError(f"vol_max must be > 0 or None, got {vol_max}")
    if gross_leverage <= 0:
        raise ValueError(f"gross_leverage must be > 0, got {gross_leverage}")

    px = prices.sort_index()

    fast_ma = px.rolling(fast, min_periods=fast).mean()
    slow_ma = px.rolling(slow, min_periods=slow).mean()

    # Trend state: +1 up-trend, 0 flat, -1 down-trend (only if shortable).
    signal = (fast_ma > slow_ma).astype(float)
    if not long_only:
        signal = signal - (fast_ma < slow_ma).astype(float)
    # Undefined until the slow MA fills.
    signal = signal.where(slow_ma.notna(), 0.0)

    # Volatility gate: realised annualised vol of simple returns.
    if vol_max is not None:
        rets = px.pct_change(fill_method=None)
        realised_vol = rets.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(
            periods_per_year
        )
        calm = (realised_vol <= vol_max) & realised_vol.notna()
        signal = signal.where(calm, 0.0)

    if normalise:
        # Spread gross_leverage across active names → always fully invested.
        gross = signal.abs().sum(axis=1).replace(0, np.nan)
        weights = signal.div(gross, axis=0).fillna(0.0) * gross_leverage
    else:
        # Fixed per-name budget → cash when names are gated off.
        n_assets = px.shape[1]
        weights = signal / n_assets * gross_leverage

    weights = weights.fillna(0.0)
    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights


def dual_ma_strategy_fn(param: Mapping[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Adapter matching the :data:`~quant_lucky.backtest.validation.StrategyFn`
    contract for :func:`~quant_lucky.backtest.walk_forward`.

    Recognised keys: ``fast``, ``slow``, ``vol_window``, ``vol_max``,
    ``long_only``, ``gross_leverage``, ``normalise``, ``periods_per_year``.
    ``vol_max`` may be ``None`` to disable the filter.
    """
    vol_max = param.get("vol_max", 0.40)
    return dual_ma_vol_filter_weights(
        prices,
        fast=int(param.get("fast", 20)),
        slow=int(param.get("slow", 60)),
        vol_window=int(param.get("vol_window", 20)),
        vol_max=None if vol_max is None else float(vol_max),
        long_only=bool(param.get("long_only", True)),
        gross_leverage=float(param.get("gross_leverage", 1.0)),
        normalise=bool(param.get("normalise", False)),
        periods_per_year=int(param.get("periods_per_year", 252)),
    )
