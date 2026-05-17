"""Factor abstractions and a small library of canonical factors.

A *factor* in this codebase is a function from the panel of OHLCV bars
to a cross-sectional signal: at each timestamp ``t`` it assigns every
asset a real number. Sort assets by that number and you have a ranking;
the tester (in :mod:`quant_lucky.factors.tester`) turns rankings into
information coefficient (IC), quantile-portfolio returns and turnover.

The classes here are deliberately thin — the goal is a uniform input/
output contract so factors compose with the tester and with each other,
not a clever DSL.

Conventions
-----------
* **Input panel**: a ``pd.DataFrame`` indexed by ``MultiIndex(date, asset)``
  with at least the canonical OHLCV columns ``open, high, low, close,
  volume``. Extra columns (``amount``, ``turnover``, ``free_float``...)
  may be required by specific factors and are noted in their docstring.
* **Output**: a ``pd.Series`` with the same ``MultiIndex`` as the input.
  Values may be NaN — typically at the start of an asset's history where
  the look-back window has not filled yet.
* **No look-ahead**: every computation uses only information available
  on or before each date. Rolling windows must be applied per-asset
  (groupby + rolling) so a high-volatility asset's history does not
  contaminate its neighbour's signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

_REQUIRED_OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Input contract helpers
# ---------------------------------------------------------------------------
def validate_panel(df: pd.DataFrame, *, required: tuple[str, ...] = _REQUIRED_OHLCV) -> None:
    """Assert ``df`` matches the factor input contract.

    Raises ``ValueError`` (not assertion) so the message is informative
    when this is hit from a notebook.
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            "Factor input must have a 2-level MultiIndex (date, asset); "
            f"got {type(df.index).__name__} with {getattr(df.index, 'nlevels', 1)} level(s)"
        )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Factor input missing required columns: {missing}")


def _ensure_date_asset_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Rename MultiIndex levels to ``date, asset`` if they look right but are unnamed.

    Cheap defensive helper so notebooks that build their own panels
    do not need to remember the exact level names.
    """
    if df.index.names != ["date", "asset"] and df.index.nlevels == 2:
        # Only rename if both levels are unnamed or have alternative names;
        # never overwrite levels that are clearly something else.
        df = df.copy()
        df.index = df.index.set_names(["date", "asset"])
    return df


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class Factor(ABC):
    """Abstract base for every factor.

    Subclasses implement :meth:`_compute` and may declare additional
    required columns via the ``required_columns`` attribute. The public
    :meth:`compute` wrapper handles input validation and post-processing
    so subclasses can focus on the math.
    """

    name: str = "factor"
    #: Columns the subclass needs in addition to the canonical OHLCV set.
    required_columns: tuple[str, ...] = ()
    #: If True, higher factor value is expected to predict higher future
    #: return. The tester uses this only for reporting (sign-flipping the
    #: long-short leg) — IC and rank IC are scale/sign agnostic anyway.
    higher_is_better: bool = True

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the factor on a panel and return a Series of values.

        Args:
            df: Panel with ``MultiIndex(date, asset)`` and the columns
                listed in :attr:`required_columns` plus the canonical
                OHLCV set.

        Returns:
            A ``pd.Series`` with the same MultiIndex as the input. Name
            is set to ``self.name``.
        """
        df = _ensure_date_asset_levels(df)
        validate_panel(df, required=_REQUIRED_OHLCV + self.required_columns)
        out = self._compute(df).rename(self.name)
        # Always sort: downstream code (winsorize, quantile bucketing) assumes
        # the same lexicographic (date, asset) order as the input panel.
        return out.sort_index()

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.Series:
        """Subclass hook. Receives a validated panel; returns a Series."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete factors
# ---------------------------------------------------------------------------
@dataclass
class MomentumFactor(Factor):
    """Classic price momentum: past N-day total return.

    Computed as ``close.pct_change(window)`` per asset. We deliberately
    use simple returns rather than log returns because the cross-section
    is later transformed by ``winsorize_mad`` + ``standardize_zscore``,
    which makes the choice irrelevant — but simple returns are easier to
    sanity-check against a chart.

    A common refinement is to exclude the most recent period (``skip``)
    to avoid contamination from short-term reversal. The ``skip``
    parameter shifts the lookback window back by ``skip`` bars.
    """

    window: int = 20
    skip: int = 0
    name: str = "momentum"
    higher_is_better: bool = True

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        if self.window <= 0:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.skip < 0:
            raise ValueError(f"skip must be >= 0, got {self.skip}")

        close = df["close"].unstack("asset")  # date x asset wide form
        if self.skip == 0:
            mom = close.pct_change(self.window, fill_method=None)
        else:
            # Past `window` return ending `skip` bars ago.
            mom = close.shift(self.skip).pct_change(self.window, fill_method=None)
        return mom.stack(future_stack=True)


@dataclass
class ReversalFactor(Factor):
    """Short-term reversal: negative of past N-day return.

    The classic short-term reversal anomaly (Jegadeesh 1990): assets
    that under-performed over the last 1-5 days tend to outperform over
    the following days. We return ``-close.pct_change(window)`` so that
    a *higher* factor value still means "buy" — consistent with
    :attr:`higher_is_better = True`.
    """

    window: int = 5
    name: str = "reversal"
    higher_is_better: bool = True

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        if self.window <= 0:
            raise ValueError(f"window must be >= 1, got {self.window}")
        close = df["close"].unstack("asset")
        rev = -close.pct_change(self.window, fill_method=None)
        return rev.stack(future_stack=True)


@dataclass
class VolatilityFactor(Factor):
    """Realised volatility of daily log returns over a rolling window.

    The low-volatility anomaly says high-vol assets *underperform* on a
    risk-adjusted basis, so high vol predicts *low* future returns.
    We expose the raw vol and set ``higher_is_better=False`` so the
    tester reports the long-short leg with the correct sign.

    Uses log returns to keep the distribution closer to symmetric, which
    matters once the cross-section gets z-scored.
    """

    window: int = 20
    name: str = "volatility"
    higher_is_better: bool = False

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        if self.window <= 1:
            raise ValueError(f"window must be >= 2, got {self.window}")
        close = df["close"].unstack("asset")
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(self.window, min_periods=self.window).std()
        return vol.stack(future_stack=True)


@dataclass
class TurnoverFactor(Factor):
    """Average turnover (volume / something proxying float) over a window.

    True turnover requires free-float share count, which we usually do
    not have at this stage of the project. We use a robust proxy:

        turnover_proxy = volume / rolling_mean(volume, long_window)

    i.e. how heavily an asset traded recently relative to its own
    history. This avoids cross-sectional size effects (a small-cap can
    have lower absolute volume than a large-cap but higher relative
    activity). High turnover historically predicts lower returns
    (liquidity premium reverses sign for retail-dominated names), so
    ``higher_is_better=False``.

    If the panel carries a ``turnover`` column from the provider
    (akshare does), pass ``use_provider_column=True`` to use it directly.
    """

    window: int = 20
    baseline_window: int = 60
    use_provider_column: bool = False
    name: str = "turnover"
    higher_is_better: bool = False

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        if self.use_provider_column:
            if "turnover" not in df.columns:
                raise ValueError("use_provider_column=True but panel has no 'turnover' column")
            turnover = df["turnover"].unstack("asset")
            avg = turnover.rolling(self.window, min_periods=self.window).mean()
            return avg.stack(future_stack=True)

        vol = df["volume"].unstack("asset")
        recent = vol.rolling(self.window, min_periods=self.window).mean()
        baseline = vol.rolling(self.baseline_window, min_periods=self.baseline_window).mean()
        # Replace zero baseline with NaN so we don't propagate inf.
        proxy = recent / baseline.replace(0, np.nan)
        return proxy.stack(future_stack=True)


@dataclass
class PriceVolumeFactor(Factor):
    """A simple value-proxy: price normalised by trading activity.

    Without fundamental data we cannot build a real PE / PB / EPYield
    factor here. Instead we expose this convenience factor:

        signal = close / rolling_mean(amount, window)

    which captures "expensive relative to recent dollar volume". It is
    NOT a substitute for a real value factor and is meant only to give
    the framework a fifth factor to demonstrate generality. Real value
    factors live in a future iteration once fundamentals are wired in.
    """

    window: int = 20
    required_columns: tuple[str, ...] = ("amount",)
    name: str = "price_volume"
    higher_is_better: bool = False

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].unstack("asset")
        amount = df["amount"].unstack("asset")
        avg_amount = amount.rolling(self.window, min_periods=self.window).mean()
        signal = close / avg_amount.replace(0, np.nan)
        return signal.stack(future_stack=True)


__all__ = [
    "Factor",
    "MomentumFactor",
    "PriceVolumeFactor",
    "ReversalFactor",
    "TurnoverFactor",
    "VolatilityFactor",
    "validate_panel",
]
