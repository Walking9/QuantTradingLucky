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
"""
