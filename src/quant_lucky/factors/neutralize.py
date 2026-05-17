"""Cross-sectional pre-processing for factor values.

Factor IC is dominated by outliers and scale-mismatch unless we
*winsorise* and *standardise* the cross-section before evaluation.
Industry / size *neutralisation* removes systematic style exposure so
the residual factor is what we actually want to test (e.g. value-after-
controlling-for-size).

All functions here operate on a Series indexed by ``MultiIndex(date,
asset)`` and apply the transform **per date**. Pandas' ``groupby
+ transform`` keeps the index aligned with the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(factor: pd.Series, k: float = 3.0) -> pd.Series:
    """Clip cross-sectional outliers to ``median ± k * MAD``.

    The MAD (median absolute deviation) is multiplied by 1.4826 so that
    for a Gaussian sample it estimates one standard deviation. This is
    the standard non-parametric winsorisation in factor research — much
    more robust than the +/- 3σ rule when the input has fat tails.

    Args:
        factor: ``Series`` indexed by ``(date, asset)``.
        k: How many MADs from the median count as "not an outlier".
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    def _clip(s: pd.Series) -> pd.Series:
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return s
        sigma = 1.4826 * mad
        lo = median - k * sigma
        hi = median + k * sigma
        return s.clip(lower=lo, upper=hi)

    return factor.groupby(level="date", group_keys=False).transform(_clip)


def standardize_zscore(factor: pd.Series) -> pd.Series:
    """Cross-sectional z-score: ``(x - mean) / std`` per date.

    Apply after :func:`winsorize_mad` so the mean and std are not pulled
    by a handful of outliers.
    """

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s - s.mean()
        return (s - s.mean()) / std

    return factor.groupby(level="date", group_keys=False).transform(_z)


def standardize_rank(factor: pd.Series) -> pd.Series:
    """Cross-sectional uniform rank transform on [-0.5, +0.5].

    More robust than z-score under non-stationary distributions, at the
    cost of throwing away magnitude information. Use this when you only
    care about the ranking and want to neutralise heteroskedastic noise.
    """

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True) - 0.5

    return factor.groupby(level="date", group_keys=False).transform(_rank)


def neutralize_by_group(
    factor: pd.Series,
    group: pd.Series,
    *,
    standardize: bool = True,
) -> pd.Series:
    """Subtract each (date, group) mean from the factor.

    The classic poor-man's industry neutralisation: rather than running
    a full OLS against industry dummies, just demean per-industry per-
    date. Mathematically equivalent for the categorical-only case and
    avoids requiring a regression library.

    Args:
        factor: ``(date, asset)``-indexed factor Series.
        group: ``(date, asset)``-indexed Series of group labels
            (industry codes, market-cap deciles, ...). NaN groups are
            preserved but their values become NaN in the output.
        standardize: If True, z-score the residual per-date so the
            output is comparable to a standardised raw factor.

    Returns:
        Residual factor with the same index as ``factor``.
    """
    if not factor.index.equals(group.index):
        # Align by intersection. We require a non-trivial overlap so a
        # totally-mismatched group input doesn't silently produce NaNs.
        common = factor.index.intersection(group.index)
        if len(common) == 0:
            raise ValueError("factor and group share no (date, asset) rows after align")
        if len(common) < 0.5 * len(factor):
            raise ValueError(
                f"factor/group alignment kept {len(common)}/{len(factor)} rows; "
                "check that both share the same (date, asset) MultiIndex"
            )
        factor = factor.loc[common]
        group = group.loc[common]

    # Per-date, per-group demean. groupby on a tuple of levels + the
    # external Series gives us exactly what we want.
    by = [factor.index.get_level_values("date"), group.values]
    demeaned = factor - factor.groupby(by).transform("mean")

    if standardize:
        return standardize_zscore(demeaned)
    return demeaned


def neutralize_by_size(
    factor: pd.Series, market_cap: pd.Series, *, standardize: bool = True
) -> pd.Series:
    """Regress factor on log(market cap) per date and return the residual.

    This is the standard size-neutralisation. We use log(cap) because
    market caps span 3-4 orders of magnitude and a linear regression on
    the raw cap is dominated by the largest names.
    """
    if (market_cap <= 0).any():
        raise ValueError("market_cap must be strictly positive")

    log_cap = np.log(market_cap)

    def _residualise(group: pd.DataFrame) -> pd.Series:
        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return pd.Series(np.nan, index=group.index)
        # OLS: y = a + b*x + e ; closed form to avoid statsmodels dependency.
        xm, ym = x[mask], y[mask]
        b, a = np.polyfit(xm, ym, 1)
        resid = y - (a + b * x)
        return pd.Series(resid, index=group.index)

    df = pd.concat([factor.rename("y"), log_cap.rename("x")], axis=1)
    residual = df.groupby(level="date", group_keys=False).apply(_residualise)
    # apply returns a DataFrame when the inner function returns Series-of-floats;
    # squeeze to a Series with the original index for downstream use.
    if isinstance(residual, pd.DataFrame):
        residual = residual.squeeze("columns")
    residual = residual.reindex(factor.index)

    if standardize:
        return standardize_zscore(residual)
    return residual


__all__ = [
    "neutralize_by_group",
    "neutralize_by_size",
    "standardize_rank",
    "standardize_zscore",
    "winsorize_mad",
]
