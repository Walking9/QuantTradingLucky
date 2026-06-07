"""Cross-sectional momentum (Jegadeesh & Titman 1993).

The canonical "buy winners, sell losers" anomaly: rank assets by their
trailing total return, go long the top quantile and short the bottom,
rebalance periodically. The classic equity configuration is **12-1
momentum** — the past 12 months of return *excluding the most recent
month* (``lookback≈252``, ``skip≈21`` trading days). Skipping the last
month sidesteps the well-documented short-term reversal that would
otherwise contaminate the signal.

This module produces a ``(date × asset)`` weight book directly from a
wide close-price panel. It is intentionally standalone (no dependency on
the factor-research stack) so it is cheap to unit-test and to drop into
:func:`quant_lucky.backtest.walk_forward`. The momentum *factor* in
:mod:`quant_lucky.factors.base` plus
:func:`quant_lucky.backtest.factor_bridge.long_short_weights` is an
equivalent path through the factor framework; the momentum strategy
report cross-checks the two agree.

No-look-ahead contract
----------------------
``weights.loc[t]`` is the portfolio formed **at the close of day t** and
uses only prices up to and including ``t`` (the trailing-return window
ends ``skip`` bars before ``t``). The :class:`~quant_lucky.backtest.VectorEngine`
internally shifts weights forward one bar so the position earns the
return over ``(t, t+1]``. **Do not pre-shift** the output of this module
before handing it to the engine — that double-counts the delay.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "cross_sectional_momentum_weights",
    "momentum_strategy_fn",
]


def cross_sectional_momentum_weights(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    top_quantile: float = 0.2,
    long_only: bool = False,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Build a cross-sectional momentum weight book.

    On every date the trailing ``lookback``-bar return ending ``skip``
    bars ago is computed per asset, the cross-section is ranked, and the
    top ``top_quantile`` fraction is bought equal-weight (and, unless
    ``long_only``, the bottom fraction is sold equal-weight).

    Args:
        prices: Wide ``date × asset`` close prices. NaN entries (asset
            not yet listed / halted) are excluded from that date's
            ranking.
        lookback: Length of the trailing return window in bars. ``252``
            ≈ one trading year.
        skip: Number of most-recent bars to exclude from the window.
            ``21`` ≈ one month → the classic "12-1" momentum. ``0`` uses
            the most recent bar (more reversal-prone).
        top_quantile: Fraction of the cross-section in each leg. ``0.2``
            → long the top 20 %, short the bottom 20 %. Must be in
            ``(0, 0.5]``.
        long_only: If True, hold only the long (top) leg with total
            weight ``gross_leverage``; markets without easy shorting
            (e.g. A-shares) use this.
        gross_leverage: Target gross exposure ``Σ|w|``. For a long-short
            book each side gets ``gross_leverage / 2``; dollar-neutral by
            construction when both legs are populated.

    Returns:
        Wide ``date × asset`` weights, zero where an asset is out of the
        selected legs or the cross-section is too thin to rank. Suitable
        for ``VectorEngine.run(weights, prices)``.

    Raises:
        ValueError: on out-of-range parameters or an empty panel.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if lookback <= 0:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if not 0.0 < top_quantile <= 0.5:
        raise ValueError(f"top_quantile must be in (0, 0.5], got {top_quantile}")
    if gross_leverage <= 0:
        raise ValueError(f"gross_leverage must be > 0, got {gross_leverage}")

    px = prices.sort_index()

    # Trailing return ending ``skip`` bars ago: uses close[t-skip] and
    # close[t-skip-lookback] — strictly information available at t.
    momentum = px.shift(skip).pct_change(lookback, fill_method=None)

    # Per-date count of rankable names and per-date leg size.
    n_valid = momentum.notna().sum(axis=1)
    n_side = np.floor(top_quantile * n_valid).astype(int).clip(lower=1)

    if long_only:
        # 1 = highest momentum (best). Long the top ``n_side`` names.
        asc_rank = momentum.rank(axis=1, ascending=False, method="first")
        long_mask = asc_rank.le(n_side, axis=0) & momentum.notna()
        n_long = long_mask.sum(axis=1).replace(0, np.nan)
        weights = long_mask.div(n_long, axis=0).fillna(0.0) * gross_leverage
        return _finalise(weights)

    # Long-short: cap the leg size so the two legs never overlap on a
    # thin cross-section (need >= 2 names; n_side <= n_valid // 2).
    n_side = np.minimum(n_side, n_valid // 2)
    best_rank = momentum.rank(axis=1, ascending=False, method="first")  # 1 = winner
    worst_rank = momentum.rank(axis=1, ascending=True, method="first")  # 1 = loser

    long_mask = best_rank.le(n_side, axis=0) & momentum.notna()
    short_mask = worst_rank.le(n_side, axis=0) & momentum.notna()

    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    side = gross_leverage / 2.0
    long_w = long_mask.div(n_long, axis=0).fillna(0.0) * side
    short_w = short_mask.div(n_short, axis=0).fillna(0.0) * side

    # Only trade dates with both legs populated; otherwise flat (a naked
    # one-sided book would contradict the long-short contract).
    both = (n_long.notna() & n_short.notna()).astype(float)
    weights = (long_w - short_w).mul(both, axis=0)
    return _finalise(weights)


def _finalise(weights: pd.DataFrame) -> pd.DataFrame:
    """Name the axes and guarantee a clean float frame."""
    weights = weights.fillna(0.0)
    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights


def momentum_strategy_fn(param: Mapping[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Adapter matching the :data:`~quant_lucky.backtest.validation.StrategyFn`
    contract so momentum can be swept by
    :func:`~quant_lucky.backtest.walk_forward`.

    Recognised keys: ``lookback``, ``skip``, ``top_quantile``,
    ``long_only``, ``gross_leverage`` (all optional; defaults match
    :func:`cross_sectional_momentum_weights`).
    """
    return cross_sectional_momentum_weights(
        prices,
        lookback=int(param.get("lookback", 252)),
        skip=int(param.get("skip", 21)),
        top_quantile=float(param.get("top_quantile", 0.2)),
        long_only=bool(param.get("long_only", False)),
        gross_leverage=float(param.get("gross_leverage", 1.0)),
    )
