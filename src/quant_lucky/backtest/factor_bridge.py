"""Bridge factor-tester output to backtest-engine input.

The factor module (``quant_lucky.factors``) outputs
:class:`~quant_lucky.factors.tester.CleanFactorData`, which carries a
``quantile`` Series in ``(date, asset)`` long form. The backtest engine
(``quant_lucky.backtest.VectorEngine``) consumes a wide
``(date × asset)`` weight DataFrame. This module is the only seam
between them.

Keeping the bridge in ``backtest/`` rather than ``factors/`` enforces
the right dependency direction: backtest depends on factors, not the
other way around. A future iteration can grow alternative weight
schemes (rank-weighted, IC-weighted, ...) here without touching the
factor module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lucky.factors.tester import CleanFactorData


def long_short_weights(
    data: CleanFactorData,
    *,
    higher_is_better: bool = True,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Build a ``(date × asset)`` long-short weight book from a factor.

    The top quantile bucket is bought equal-weight; the bottom is sold
    equal-weight; everything in between is zero. Total gross exposure
    (``Σ|w|``) equals ``gross_leverage`` on every date that has at least
    one name in **both** the top and bottom buckets; on degenerate dates
    (missing top or bottom) the row is all zeros — the engine sees no
    position and pays no cost.

    Args:
        data: Output of
            :func:`~quant_lucky.factors.tester.get_clean_factor_and_forward_returns`.
            Only ``data.quantile`` and ``data.quantiles`` are read.
        higher_is_better: Reading direction of the factor.
            * True  → long top quantile, short bottom (the natural reading
              for momentum, quality, ...).
            * False → long bottom, short top (low-vol, low-turnover,
              cheap-PE anomalies). Equivalent to negating the returned
              weights.
        gross_leverage: Σ|w| target. ``1.0`` means each side gets 0.5
            and the book is dollar-neutral with unit gross. Use ``2.0``
            for "100% long, 100% short" books.

    Returns:
        Wide ``DataFrame`` indexed by date with one column per asset.
        Suitable for ``VectorEngine.run(weights, prices)``.
    """
    if gross_leverage <= 0:
        raise ValueError(f"gross_leverage must be > 0, got {gross_leverage}")
    if data.quantiles < 2:
        raise ValueError(
            f"need at least 2 quantile buckets to form long-short; "
            f"data.quantiles = {data.quantiles}"
        )

    q = data.quantile
    top_q = data.quantiles
    bottom_q = 1

    # Bucket masks in long form, then unstack to wide. `q` may have NaN
    # entries for dates where the cross-section was too thin to bucket;
    # `q == top_q` evaluates NaN to False, so those rows naturally land
    # at zero weight.
    top_mask = (q == top_q).astype(float).unstack("asset").fillna(0.0)
    bot_mask = (q == bottom_q).astype(float).unstack("asset").fillna(0.0)

    # Per-date bucket sizes. Replace 0 with NaN so a division-by-zero
    # propagates to NaN (then fillna(0)), keeping degenerate rows flat.
    n_top = top_mask.sum(axis=1).replace(0, np.nan)
    n_bot = bot_mask.sum(axis=1).replace(0, np.nan)

    side_leverage = gross_leverage / 2.0
    long_w = top_mask.div(n_top, axis=0).fillna(0.0) * side_leverage
    short_w = bot_mask.div(n_bot, axis=0).fillna(0.0) * side_leverage

    # Force degenerate dates (only one side populated) to flat; otherwise
    # we'd hold a naked one-sided book that contradicts the "long-short"
    # contract.
    valid = (n_top.notna() & n_bot.notna()).astype(float).values
    weights = (long_w - short_w).mul(valid, axis=0)

    if not higher_is_better:
        weights = -weights

    weights.index.name = "date"
    weights.columns.name = "asset"
    return weights


__all__ = ["long_short_weights"]
