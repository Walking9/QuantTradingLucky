"""Backtesting engines.

Two complementary engines:

1. **Vectorised engine** (``backtest.vector``)
   - Fast, pandas-based.
   - Best for factor research and portfolio-level backtests at daily
     or lower frequency.
   - Not suitable for order-book or complex order-type simulation.

2. **Event-driven engine** (``backtest.event``) *(later phase)*
   - Simulates order submission, fills, rejections, partial fills.
   - Required for realistic intraday / HFT-style testing and for
     matching the live-trading code path.

Core responsibilities shared by both engines:

- Deterministic random seeding for reproducibility.
- Bias guards: no look-ahead, no survivorship, proper point-in-time.
- Uniform ``PerformanceReport`` output (Sharpe, Sortino, Calmar, MDD,
  turnover, hit rate, factor exposures).

Typical usage
-------------
::

    from quant_lucky.backtest import VectorEngine

    engine = VectorEngine(cost_bps=10.0)
    result = engine.run(weights, prices)
    print(result.report)
    result.portfolio_value.plot()
"""

from __future__ import annotations

from quant_lucky.backtest.event import Event, EventEngine
from quant_lucky.backtest.factor_bridge import long_short_weights
from quant_lucky.backtest.report import PerformanceReport, compute_performance
from quant_lucky.backtest.validation import (
    Split,
    WalkForwardResult,
    anchored_walk_forward,
    deflated_sharpe_ratio,
    fixed_split,
    rolling_walk_forward,
    walk_forward,
)
from quant_lucky.backtest.vector import BacktestResult, VectorEngine

__all__ = [
    "BacktestResult",
    "Event",
    "EventEngine",
    "PerformanceReport",
    "Split",
    "VectorEngine",
    "WalkForwardResult",
    "anchored_walk_forward",
    "compute_performance",
    "deflated_sharpe_ratio",
    "fixed_split",
    "long_short_weights",
    "rolling_walk_forward",
    "walk_forward",
]
