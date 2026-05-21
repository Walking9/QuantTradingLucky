"""Performance report data structures and metric computations.

A ``PerformanceReport`` is the uniform output of every backtest in this
codebase (vectorised today, event-driven tomorrow). Producing it from a
return series is one helper; rendering it as a one-row summary is another.

Conventions
-----------
* Inputs are *simple* returns (e.g. ``0.01`` for +1%) indexed by date.
* Annualisation factor defaults to 252 (equity convention). Override
  via ``periods_per_year`` for crypto (365), weekly (52), monthly (12).
* "Sharpe" is risk-free-rate=0 by default; set ``rf_annual`` to subtract
  a constant annual risk-free rate before annualising.
* Drawdowns are reported as **negative** numbers (``min`` of the
  drawdown series). A flat curve has zero drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceReport:
    """Standard performance summary for a return series.

    All numeric scalars are floats so the report serialises cleanly to
    JSON / Parquet. The accompanying time series (``returns``,
    ``cumulative``, ``drawdown_series``) keep their original date index
    so charts can be drawn directly off the report.
    """

    returns: pd.Series
    cumulative: pd.Series
    drawdown_series: pd.Series
    annual_return: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hit_rate: float
    n_periods: int
    periods_per_year: int
    turnover_annual: float | None = None

    def to_summary(self) -> pd.Series:
        """One-row Series for tabular display in notebooks / reports."""
        data = {
            "annual_return": self.annual_return,
            "annual_vol": self.annual_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "n_periods": self.n_periods,
        }
        if self.turnover_annual is not None:
            data["turnover_annual"] = self.turnover_annual
        return pd.Series(data)

    def __repr__(self) -> str:
        parts = [
            f"PerformanceReport(n={self.n_periods}, ",
            f"annual_return={self.annual_return:.2%}, ",
            f"annual_vol={self.annual_vol:.2%}, ",
            f"sharpe={self.sharpe:.2f}, ",
            f"sortino={self.sortino:.2f}, ",
            f"calmar={self.calmar:.2f}, ",
            f"mdd={self.max_drawdown:.2%}, ",
            f"hit_rate={self.hit_rate:.2%}",
        ]
        if self.turnover_annual is not None:
            parts.append(f", turnover_annual={self.turnover_annual:.2f}")
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _max_drawdown(cumulative: pd.Series) -> tuple[float, pd.Series]:
    """Return (min_drawdown, drawdown_series).

    ``cumulative`` is the *cumulative simple return* (i.e. starts near 0,
    not 1). We add 1 internally so the drawdown is computed on the value
    curve.
    """
    value = 1.0 + cumulative.fillna(0.0)
    running_max = value.cummax()
    drawdown = value / running_max - 1.0
    if drawdown.empty:
        return 0.0, drawdown
    return float(drawdown.min()), drawdown


def compute_performance(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    turnover_annual: float | None = None,
) -> PerformanceReport:
    """Compute the standard report from a simple-return series.

    Args:
        returns: simple returns (per-period, e.g. daily). NaNs are
            dropped before statistics; the full series (with NaNs) is
            still stored on the report for plotting alignment.
        periods_per_year: annualisation factor. 252 for equities, 365
            for crypto, 52 for weekly, 12 for monthly.
        rf_annual: annual risk-free rate. Converted to per-period via
            ``rf_annual / periods_per_year`` and subtracted from returns
            before computing Sharpe / Sortino.
        turnover_annual: optional pre-computed annualised one-way
            turnover (e.g. from the engine). Carried on the report for
            display; not used in any metric.

    Returns:
        :class:`PerformanceReport`.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")

    clean = returns.dropna()
    n = len(clean)

    if n == 0:
        empty = pd.Series(dtype=float)
        return PerformanceReport(
            returns=returns,
            cumulative=empty,
            drawdown_series=empty,
            annual_return=float("nan"),
            annual_vol=float("nan"),
            sharpe=float("nan"),
            sortino=float("nan"),
            calmar=float("nan"),
            max_drawdown=0.0,
            hit_rate=float("nan"),
            n_periods=0,
            periods_per_year=periods_per_year,
            turnover_annual=turnover_annual,
        )

    rf_per_period = rf_annual / periods_per_year
    excess = clean - rf_per_period

    mean = float(excess.mean())
    # ddof=1 unless n==1 (std is undefined for a single sample → NaN). We
    # don't special-case here: pandas returns NaN, and the downstream
    # Sharpe / Sortino become NaN, which is the right behaviour.
    std = float(excess.std(ddof=1)) if n > 1 else float("nan")

    annual_return = mean * periods_per_year
    annual_vol = std * np.sqrt(periods_per_year) if not np.isnan(std) else float("nan")

    # ``annual_vol > 0`` is too loose — a "constant" series in float64 has
    # std on the order of machine epsilon, producing astronomical Sharpe.
    # Guard against this by requiring vol to be material relative to the
    # mean (or the period-mean if mean is zero).
    vol_floor = 1e-12 * max(abs(mean), 1e-6)
    sharpe = (
        annual_return / annual_vol
        if not np.isnan(annual_vol) and annual_vol > vol_floor
        else float("nan")
    )

    # Sortino: downside deviation = std of min(r - target, 0). Target = 0
    # (i.e. losses only), per Sortino's original 1991 definition. Using
    # excess returns above the risk-free rate makes "downside" mean
    # "worse than the RF benchmark". Use the population std (Sortino's
    # original convention, not Bessel-corrected).
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt((downside**2).sum() / (n - 1))) if n > 1 else float("nan")
    downside_annual = (
        downside_dev * np.sqrt(periods_per_year) if not np.isnan(downside_dev) else float("nan")
    )
    sortino = (
        annual_return / downside_annual
        if not np.isnan(downside_annual) and downside_annual > vol_floor
        else float("nan")
    )

    # Cumulative compounded return. Operate on the original ``returns`` so
    # the index alignment for plotting is preserved.
    cumulative = (1.0 + returns.fillna(0.0)).cumprod() - 1.0
    max_dd, drawdown_series = _max_drawdown(cumulative)

    calmar = annual_return / abs(max_dd) if max_dd < 0 else float("nan")

    hit_rate = float((clean > 0).mean())

    return PerformanceReport(
        returns=returns,
        cumulative=cumulative,
        drawdown_series=drawdown_series,
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        n_periods=n,
        periods_per_year=periods_per_year,
        turnover_annual=turnover_annual,
    )


__all__ = [
    "PerformanceReport",
    "compute_performance",
]
