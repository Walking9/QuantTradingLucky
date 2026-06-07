"""Classic trading strategies (M6: strategy reproduction & evaluation).

Each module here is a **reusable, no-look-ahead signal generator** that
maps a wide ``date × asset`` close-price panel to a ``date × asset``
weight book, ready for :class:`quant_lucky.backtest.VectorEngine`. They
are intentionally small and dependency-light so they unit-test cheaply
and plug directly into :func:`quant_lucky.backtest.walk_forward` via the
``*_strategy_fn`` adapters.

Strategies
----------
* :mod:`~quant_lucky.strategies.momentum` — cross-sectional momentum
  (Jegadeesh & Titman 1993; the 12-1 configuration).
* :mod:`~quant_lucky.strategies.dual_ma` — dual moving-average crossover
  with a volatility filter (trend following).
* :mod:`~quant_lucky.strategies.risk_parity` — inverse-volatility
  (naive risk-parity) multi-asset allocation, plus an equal-weight
  benchmark.

Data plumbing for the runnable ``strategies/<name>/`` research packages
lives in :mod:`~quant_lucky.strategies.data`.

Contract shared by every weight generator
------------------------------------------
``weights.loc[t]`` is the portfolio formed at the **close of day t**
using only information available at ``t``. The engine shifts it forward
one bar to earn ``(t, t+1]`` — callers must **not** pre-shift.
"""

from __future__ import annotations

from quant_lucky.strategies.data import (
    PriceSpec,
    load_close_panel,
    synthetic_price_panel,
)
from quant_lucky.strategies.dual_ma import (
    dual_ma_strategy_fn,
    dual_ma_vol_filter_weights,
)
from quant_lucky.strategies.evaluation import (
    FactorAttribution,
    ResearchSpec,
    SplitEvaluation,
    StrategyEvaluation,
    attribution,
    evaluate_strategy,
    is_oos_split,
    parameter_sensitivity,
    run_research,
    write_artifacts,
)
from quant_lucky.strategies.momentum import (
    cross_sectional_momentum_weights,
    momentum_strategy_fn,
)
from quant_lucky.strategies.risk_parity import (
    equal_weight_weights,
    hold_between_rebalances,
    inverse_vol_weights,
    rebalance_dates,
    risk_parity_strategy_fn,
)

__all__ = [
    "FactorAttribution",
    "PriceSpec",
    "ResearchSpec",
    "SplitEvaluation",
    "StrategyEvaluation",
    "attribution",
    "cross_sectional_momentum_weights",
    "dual_ma_strategy_fn",
    "dual_ma_vol_filter_weights",
    "equal_weight_weights",
    "evaluate_strategy",
    "hold_between_rebalances",
    "inverse_vol_weights",
    "is_oos_split",
    "load_close_panel",
    "momentum_strategy_fn",
    "parameter_sensitivity",
    "rebalance_dates",
    "risk_parity_strategy_fn",
    "run_research",
    "synthetic_price_panel",
    "write_artifacts",
]
