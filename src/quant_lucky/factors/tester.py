"""Single-factor test framework.

Given a factor Series (output of :class:`quant_lucky.factors.base.Factor`)
and a panel of prices, this module produces the standard ICs / quantile
returns / long-short / turnover report that any factor researcher will
recognise.

The API mirrors ``alphalens`` for ease of comparison, but the
implementation is intentionally compact and dependency-free: just numpy
+ pandas + scipy. Roughly:

* :func:`get_clean_factor_and_forward_returns` aligns a factor with the
  forward returns for each requested horizon and bucketises the factor
  into ``quantiles`` groups cross-sectionally per date.
* :func:`compute_ic` returns daily IC / rank-IC time series per horizon.
* :func:`compute_mean_returns_by_quantile` averages forward returns
  inside each quantile (per date or pooled).
* :func:`compute_long_short` returns the long-top minus short-bottom
  cumulative return series and a small Sharpe / max-drawdown summary.
* :func:`compute_turnover` measures average rebalance turnover for each
  quantile portfolio — a sanity check for cost analysis later.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

ICMethod = Literal["pearson", "spearman"]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _validate_factor(factor: pd.Series) -> pd.Series:
    """Defensive: ensure factor is a Series with (date, asset) MultiIndex."""
    if not isinstance(factor, pd.Series):
        raise TypeError(f"factor must be a pd.Series, got {type(factor).__name__}")
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise ValueError(
            "factor must be indexed by MultiIndex(date, asset); "
            f"got {factor.index.nlevels} level(s)"
        )
    if factor.index.names != ["date", "asset"]:
        factor = factor.copy()
        factor.index = factor.index.set_names(["date", "asset"])
    return factor


def _forward_returns(prices: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Compute future N-day simple returns for each horizon.

    ``prices`` is a wide DataFrame (date x asset). For period ``p`` we
    return ``prices.shift(-p) / prices - 1``, so each row at date ``t``
    contains the return realised over ``(t, t+p]``. Last ``p`` rows are
    NaN by construction.
    """
    if prices.isna().all(axis=None):
        raise ValueError("prices is entirely NaN")

    out = {}
    for p in periods:
        if p <= 0:
            raise ValueError(f"forward return period must be >= 1, got {p}")
        out[f"period_{p}"] = (prices.shift(-p) / prices - 1.0).stack(future_stack=True)
    df = pd.concat(out, axis=1)
    df.index = df.index.set_names(["date", "asset"])
    return df


def _quantile_bucketise(
    factor: pd.Series, quantiles: int, *, min_assets_per_date: int = 5
) -> pd.Series:
    """Bucket each date's cross-section into ``quantiles`` groups (1..Q).

    Uses ``pd.qcut`` per date. Dates with fewer non-NaN factor values
    than ``min_assets_per_date`` are dropped — quantile binning on a
    handful of names is more noise than signal.
    """
    if quantiles < 2:
        raise ValueError(f"quantiles must be >= 2, got {quantiles}")

    def _bucket(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) < max(min_assets_per_date, quantiles):
            return pd.Series(np.nan, index=s.index)
        try:
            bins = pd.qcut(s_clean, q=quantiles, labels=False, duplicates="drop")
        except ValueError:
            # All ties (e.g. all zeros) — qcut fails. Treat as no signal.
            return pd.Series(np.nan, index=s.index)
        # If duplicates="drop" reduced bin count, signal is too degenerate.
        if bins.nunique() < 2:
            return pd.Series(np.nan, index=s.index)
        # qcut labels from 0..Q-1; shift to 1..Q for human readability.
        return bins.reindex(s.index).add(1)

    return factor.groupby(level="date", group_keys=False).apply(_bucket)


@dataclass
class CleanFactorData:
    """Bundle returned by :func:`get_clean_factor_and_forward_returns`.

    Attributes
    ----------
    factor: aligned factor values (Series, MultiIndex)
    forward_returns: forward return columns ``period_N`` (DataFrame)
    quantile: quantile bucket 1..Q (Series, aligned)
    quantiles: how many buckets were requested
    periods: list of forward-return horizons
    """

    factor: pd.Series
    forward_returns: pd.DataFrame
    quantile: pd.Series
    quantiles: int
    periods: list[int]

    def joined(self) -> pd.DataFrame:
        """Return everything as a single DataFrame (handy for groupby work)."""
        df = self.forward_returns.copy()
        df["factor"] = self.factor
        df["quantile"] = self.quantile
        return df


def get_clean_factor_and_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    *,
    periods: list[int] | tuple[int, ...] = (1, 5, 20),
    quantiles: int = 5,
    drop_na: bool = True,
) -> CleanFactorData:
    """Align factor with forward returns and bucket the cross-section.

    Args:
        factor: Series indexed by ``(date, asset)``.
        prices: Wide price DataFrame (date x asset). Typically
            ``df['close'].unstack('asset')``.
        periods: Forward return horizons in bars (e.g. 1, 5, 20 days).
        quantiles: How many cross-sectional quantile buckets.
        drop_na: Drop rows missing the factor value or *all* forward
            returns. Keeps memory bounded; downstream code is more
            robust if you leave the NaNs in.

    Returns:
        A :class:`CleanFactorData` bundle. The factor and quantile
        Series share the same MultiIndex; ``forward_returns`` is a
        DataFrame with the same index and one column per period.
    """
    factor = _validate_factor(factor)
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"prices must be a wide DataFrame (date x asset); got {type(prices).__name__}"
        )

    periods_list = list(periods)
    fwd = _forward_returns(prices, periods_list)

    # Inner-join: only keep (date, asset) rows where factor is defined AND
    # at least one forward return column is.
    combined = pd.concat([factor.rename("factor"), fwd], axis=1)
    if drop_na:
        combined = combined.dropna(subset=["factor"], how="any")
        # Drop rows where every forward return is NaN — keeps tail rows
        # with only the long-horizon column populated.
        period_cols = [f"period_{p}" for p in periods_list]
        combined = combined.dropna(subset=period_cols, how="all")

    factor_aligned = combined["factor"]
    fwd_aligned = combined[[f"period_{p}" for p in periods_list]]
    quantile = _quantile_bucketise(factor_aligned, quantiles)

    return CleanFactorData(
        factor=factor_aligned,
        forward_returns=fwd_aligned,
        quantile=quantile,
        quantiles=quantiles,
        periods=periods_list,
    )


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------
def compute_ic(
    data: CleanFactorData,
    *,
    method: ICMethod = "spearman",
) -> pd.DataFrame:
    """IC time series: correlation(factor, forward return) per date.

    Args:
        data: Output of :func:`get_clean_factor_and_forward_returns`.
        method: ``"spearman"`` for rank IC (recommended — robust to
            outliers), ``"pearson"`` for linear IC.

    Returns:
        DataFrame indexed by date with one column per forward horizon
        (``period_1``, ``period_5``, ...).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    df = data.joined()

    def _ic(group: pd.DataFrame) -> pd.Series:
        f = group["factor"].to_numpy()
        out = {}
        for p in data.periods:
            r = group[f"period_{p}"].to_numpy()
            mask = ~(np.isnan(f) | np.isnan(r))
            if mask.sum() < 5:  # too few non-NaN pairs for a meaningful corr
                out[f"period_{p}"] = np.nan
                continue
            if method == "pearson":
                rho, _ = stats.pearsonr(f[mask], r[mask])
            else:
                rho, _ = stats.spearmanr(f[mask], r[mask])
            out[f"period_{p}"] = rho
        return pd.Series(out)

    ic = df.groupby(level="date").apply(_ic)
    return ic


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-horizon summary table: mean IC, IC IR, t-stat, hit rate.

    * IC mean: average daily IC.
    * IC IR: mean / std (information ratio of the IC time series — the
      classic measure of factor consistency).
    * t-stat: ``mean * sqrt(N) / std`` (one-sample t-stat for IC > 0).
    * Hit rate: fraction of dates where IC > 0.
    """
    rows = []
    for col in ic.columns:
        s = ic[col].dropna()
        n = len(s)
        if n == 0:
            rows.append(
                {
                    "horizon": col,
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "ic_ir": np.nan,
                    "t_stat": np.nan,
                    "hit_rate": np.nan,
                    "n": 0,
                }
            )
            continue
        mean = s.mean()
        std = s.std(ddof=1)
        ir = mean / std if std > 0 else np.nan
        t = mean * np.sqrt(n) / std if std > 0 else np.nan
        rows.append(
            {
                "horizon": col,
                "ic_mean": mean,
                "ic_std": std,
                "ic_ir": ir,
                "t_stat": t,
                "hit_rate": (s > 0).mean(),
                "n": n,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Quantile returns
# ---------------------------------------------------------------------------
def compute_mean_returns_by_quantile(
    data: CleanFactorData,
    *,
    by_date: bool = False,
) -> pd.DataFrame:
    """Average forward return inside each quantile bucket.

    Args:
        data: Clean factor data.
        by_date: If True, return a (date, quantile) MultiIndex DataFrame
            (useful for plotting quantile-return time series). If False,
            pool across dates.

    Returns:
        DataFrame with columns for each forward horizon.
    """
    df = data.joined().dropna(subset=["quantile"])
    period_cols = [f"period_{p}" for p in data.periods]

    if by_date:
        return df.groupby(["date", "quantile"], observed=True)[period_cols].mean()
    return df.groupby("quantile", observed=True)[period_cols].mean()


def _empty_long_short_summary() -> dict[str, float | pd.Series]:
    """Return a placeholder long-short summary when no data is available."""
    return {
        "series": pd.Series(dtype=float),
        "cumulative": pd.Series(dtype=float),
        "annualised_return": float("nan"),
        "annualised_vol": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "n_obs": 0,
    }


def compute_long_short(
    data: CleanFactorData,
    *,
    period: int | None = None,
    higher_is_better: bool = True,
) -> dict[str, float | pd.Series]:
    """Long-top-quantile minus short-bottom-quantile (daily, then summary).

    Returns a dict with:
      - ``series``: the daily long-short return as a Series indexed by date
      - ``cumulative``: cumulative compounded return
      - ``annualised_return``: 252-day-annualised mean
      - ``annualised_vol``: 252-day-annualised std
      - ``sharpe``: annualised Sharpe (rf=0)
      - ``max_drawdown``: peak-to-trough on the cumulative curve
    """
    if period is None:
        period = data.periods[0]
    col = f"period_{period}"
    if col not in data.forward_returns.columns:
        raise ValueError(f"period={period} not in computed periods {data.periods}")

    df = data.joined().dropna(subset=["quantile", col])
    top_q = data.quantiles
    bottom_q = 1

    if df.empty:
        return _empty_long_short_summary()

    by_q = df.groupby(["date", "quantile"], observed=True)[col].mean().unstack("quantile")

    if top_q not in by_q.columns or bottom_q not in by_q.columns:
        return _empty_long_short_summary()

    long_short = by_q[top_q] - by_q[bottom_q]
    if not higher_is_better:
        long_short = -long_short

    # Forward return of period ``p`` is realised over the next p bars, so to
    # avoid overlapping-return double counting on Sharpe / vol we sample
    # every ``period`` rows. For the summary we are deliberately loose: the
    # framework is for learning, not paper-publishable error bars.
    sampled = long_short.iloc[::period].dropna()
    n = len(sampled)
    if n < 2:
        out = _empty_long_short_summary()
        out["series"] = long_short
        out["n_obs"] = n
        return out

    # Bars-per-year for an annualisation factor. 252 is the standard equity
    # convention; with ``period`` bars between samples we annualise the
    # *sample-level* statistics.
    bars_per_year = 252.0 / period
    mean_per_sample = sampled.mean()
    std_per_sample = sampled.std(ddof=1)
    annualised_return = mean_per_sample * bars_per_year
    annualised_vol = std_per_sample * np.sqrt(bars_per_year)
    sharpe = annualised_return / annualised_vol if annualised_vol > 0 else float("nan")

    cumulative = (1 + long_short.fillna(0)).cumprod() - 1
    running_max = (1 + cumulative).cummax()
    drawdown = (1 + cumulative) / running_max - 1
    max_dd = float(drawdown.min())

    return {
        "series": long_short,
        "cumulative": cumulative,
        "annualised_return": float(annualised_return),
        "annualised_vol": float(annualised_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------
def compute_turnover(data: CleanFactorData) -> pd.Series:
    """Average per-rebalance turnover for each quantile bucket.

    Turnover at date ``t`` for quantile ``q`` is::

        1 - |members_t ∩ members_{t-1}| / |members_t|

    i.e. the fraction of names that left the bucket since the previous
    date. Returns the time-average per quantile.
    """
    quantile = data.quantile.dropna()
    # Build a (date, quantile) -> frozenset(assets) map.
    grp = quantile.groupby(level="date")
    by_date = {d: g.groupby(g.values, observed=True).groups for d, g in grp}

    dates = sorted(by_date.keys())
    turnovers: dict[int, list[float]] = {q: [] for q in range(1, data.quantiles + 1)}

    for prev, curr in pairwise(dates):
        prev_groups = by_date[prev]
        curr_groups = by_date[curr]
        for q, bucket in turnovers.items():
            prev_members = {idx[1] for idx in prev_groups.get(q, [])}
            curr_members = {idx[1] for idx in curr_groups.get(q, [])}
            if not curr_members:
                continue
            overlap = len(prev_members & curr_members) / len(curr_members)
            bucket.append(1.0 - overlap)

    return pd.Series(
        {q: float(np.mean(v)) if v else float("nan") for q, v in turnovers.items()},
        name="turnover",
    ).sort_index()


__all__ = [
    "CleanFactorData",
    "compute_ic",
    "compute_long_short",
    "compute_mean_returns_by_quantile",
    "compute_turnover",
    "get_clean_factor_and_forward_returns",
    "ic_summary",
]
