"""Vectorised pandas-based backtest engine.

The engine takes:

* ``weights``: a wide DataFrame ``(date × asset)`` of target portfolio
  weights. Sum of *absolute* weights may exceed 1 (long-short books).
  Weights at row ``t`` are decisions made using information **up to and
  including** ``t`` and earn the return realised over ``(t, t+1]``.
* ``prices``: a wide DataFrame ``(date × asset)`` of prices used to
  derive simple per-period returns.

It produces a :class:`BacktestResult` bundling the gross / net return
series, turnover, cost time series, portfolio value curve, and a
standard :class:`~quant_lucky.backtest.report.PerformanceReport`.

Design rules
------------
1. **No look-ahead.** Weights are internally shifted forward by one bar
   so the user's ``weights[t]`` semantics ("decision at t, earned
   t→t+1") match the implementation. Try to use the engine without
   reading this docstring once and the test ``test_look_ahead_*`` will
   bite you.
2. **Costs are explicit.** Either a scalar ``cost_bps`` (constant
   per-side basis points) or a :class:`~quant_lucky.costs.base.CostModel`
   instance. Zero-cost runs are allowed but the engine refuses to
   silently default to them — pass ``cost_bps=0.0`` explicitly to
   acknowledge.
3. **Pure-pandas, single-pass.** No event loop, no rebalance scheduler.
   If you need event-driven semantics (limit orders, stops, partial
   fills) use the upcoming event engine; the vectorised path will give
   wrong answers for those.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_lucky.backtest.report import PerformanceReport, compute_performance
from quant_lucky.costs.base import CostModel, Side, Trade


@dataclass(frozen=True)
class BacktestResult:
    """Bundle returned by :meth:`VectorEngine.run`.

    Attributes
    ----------
    report:
        Performance summary (Sharpe / MDD / etc.) computed on
        ``net_returns``.
    gross_returns:
        Per-period portfolio return **before** costs.
    net_returns:
        Per-period portfolio return after subtracting ``cost_series``.
    portfolio_value:
        Cumulative net-return curve, normalised to start at 1.0.
    turnover_series:
        Per-period one-way turnover ``Σ |Δw_{t,i}|``. The first
        rebalance counts the full position as a buy (turnover = Σ|w_0|).
    cost_series:
        Per-period transaction cost, expressed as a *fraction of
        portfolio value*. Same index as the returns.
    """

    report: PerformanceReport
    gross_returns: pd.Series
    net_returns: pd.Series
    portfolio_value: pd.Series
    turnover_series: pd.Series
    cost_series: pd.Series


class VectorEngine:
    """Single-frequency vectorised backtest engine.

    Parameters
    ----------
    cost_bps:
        Per-side cost in basis points applied to the absolute change in
        weights. ``cost_bps=10`` means 10 bps of notional traded each
        time we trade. Conservative default for retail equities.
        Mutually exclusive with ``cost_model``.
    cost_model:
        A :class:`~quant_lucky.costs.base.CostModel` instance for
        realistic per-fill cost simulation. The engine synthesises one
        :class:`~quant_lucky.costs.base.Trade` per (date, asset) with
        non-zero weight change.
    periods_per_year:
        Annualisation factor forwarded to
        :func:`~quant_lucky.backtest.report.compute_performance`.
        Defaults to 252 (equity convention).
    initial_capital:
        Notional starting capital used only when ``cost_model`` is set
        (to convert weight changes into trade quantities). Reported
        metrics are scale-invariant.
    """

    def __init__(
        self,
        *,
        cost_bps: float | None = None,
        cost_model: CostModel | None = None,
        periods_per_year: int = 252,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        if cost_bps is not None and cost_model is not None:
            raise ValueError("Pass either cost_bps or cost_model, not both.")
        if cost_bps is None and cost_model is None:
            raise ValueError(
                "Either cost_bps (e.g. 0.0 for cost-free) or cost_model must be "
                "specified. Zero-cost runs must be explicit to discourage "
                "accidental costless backtests."
            )
        if cost_bps is not None and cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
        if periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")

        self.cost_bps = cost_bps
        self.cost_model = cost_model
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, weights: pd.DataFrame, prices: pd.DataFrame) -> BacktestResult:
        """Run the backtest and return the result bundle.

        Args:
            weights: ``date × asset`` target weights. NaN → treated as 0.
            prices: ``date × asset`` close prices.

        Returns:
            :class:`BacktestResult` with returns, turnover, costs, report.
        """
        w, px = self._align_and_validate(weights, prices)

        # Asset simple returns: return at row t is realised from t-1 → t.
        asset_returns = px.pct_change(fill_method=None)

        # Convention: weight decided at end of day t earns the return over
        # (t, t+1]. Implemented as ``weights.shift(1) * asset_returns``.
        # This is THE place look-ahead bias hides; treat with care.
        held = w.shift(1).fillna(0.0)
        gross_per_asset = held * asset_returns
        gross_returns = gross_per_asset.sum(axis=1)

        # Turnover & costs ------------------------------------------------
        # |Δw_t| = |w_t - w_{t-1}|. Treat the first observation as buying
        # from a zero portfolio (turnover = Σ|w_0|). Empty/NaN weights at
        # the head are zero by construction.
        prev_w = w.shift(1).fillna(0.0)
        delta_w = w.fillna(0.0) - prev_w
        turnover_series = delta_w.abs().sum(axis=1)
        turnover_series.name = "turnover"

        cost_series = self._compute_costs(delta_w, px)
        cost_series.name = "cost"

        net_returns = (gross_returns - cost_series).rename("net_return")
        gross_returns = gross_returns.rename("gross_return")

        portfolio_value = (1.0 + net_returns.fillna(0.0)).cumprod()
        portfolio_value.name = "portfolio_value"

        # Annualised one-way turnover (handy for cost-drag intuition).
        n = len(net_returns)
        turnover_annual: float | None
        if n > 1:
            turnover_annual = float(turnover_series.sum() / n * self.periods_per_year)
        else:
            turnover_annual = None

        report = compute_performance(
            net_returns,
            periods_per_year=self.periods_per_year,
            turnover_annual=turnover_annual,
        )

        return BacktestResult(
            report=report,
            gross_returns=gross_returns,
            net_returns=net_returns,
            portfolio_value=portfolio_value,
            turnover_series=turnover_series,
            cost_series=cost_series,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _align_and_validate(
        weights: pd.DataFrame, prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reindex both frames to a common (date, asset) grid.

        We intersect both axes:
        * dates: rows present in *both* (so prices and weights advance in
          lockstep — the alternative, taking the union and forward-filling
          weights, hides bugs where the strategy "skips" a date).
        * assets: columns present in *both*. Weights for assets without
          prices would never realise a return; prices without weights are
          irrelevant.
        """
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(f"weights must be a DataFrame, got {type(weights).__name__}")
        if not isinstance(prices, pd.DataFrame):
            raise TypeError(f"prices must be a DataFrame, got {type(prices).__name__}")
        if weights.empty:
            raise ValueError("weights is empty")
        if prices.empty:
            raise ValueError("prices is empty")

        common_dates = weights.index.intersection(prices.index)
        if len(common_dates) == 0:
            raise ValueError("weights and prices share no common dates")
        common_assets = weights.columns.intersection(prices.columns)
        if len(common_assets) == 0:
            raise ValueError("weights and prices share no common assets")

        w = weights.loc[common_dates, common_assets].sort_index()
        px = prices.loc[common_dates, common_assets].sort_index()
        return w, px

    def _compute_costs(self, delta_w: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Translate per-asset weight changes into per-period cost fractions."""
        if self.cost_bps is not None:
            # ``cost_bps`` is per-side, applied to the absolute weight
            # change. Σ|Δw| is two-sided already, so a 1.0 turnover with
            # cost_bps=10 → 10 bps drag.
            return (delta_w.abs().sum(axis=1) * self.cost_bps / 10_000.0).rename("cost")

        # cost_model path: synthesise a Trade for each non-zero (date, asset).
        # Costs come out in *quote currency* — divide by portfolio value at
        # that date to get a fraction of NAV. We use ``initial_capital`` as
        # the running portfolio value reference; this is a fixed-NAV approx
        # appropriate for cost-drag analysis and matches what
        # ``AnnualCostCalculator`` does. A growing-NAV mode is left to a
        # future iteration.
        assert self.cost_model is not None  # for mypy

        cost_per_period: dict[pd.Timestamp, float] = {}
        nav = self.initial_capital
        for date in delta_w.index:
            row = delta_w.loc[date]
            nonzero = row[row != 0.0]
            if nonzero.empty:
                cost_per_period[date] = 0.0
                continue
            total = 0.0
            for asset, dw in nonzero.items():
                price = prices.at[date, asset]
                if pd.isna(price) or price <= 0:
                    # No tradable price → cannot construct a Trade. Skip
                    # silently rather than throw; this matches a real-life
                    # halt where no fill happens.
                    continue
                notional = abs(dw) * nav
                quantity = notional / price
                if quantity <= 0:
                    continue
                side = Side.BUY if dw > 0 else Side.SELL
                trade = Trade(symbol=str(asset), side=side, quantity=quantity, price=price)
                total += self.cost_model.estimate(trade).total
            cost_per_period[date] = total / nav

        return pd.Series(cost_per_period, name="cost").reindex(delta_w.index).fillna(0.0)


__all__ = [
    "BacktestResult",
    "VectorEngine",
]
