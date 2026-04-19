"""QuantTradingLucky — a personal quantitative trading learning project.

This package is organised by functional domain rather than by strategy.
Each subpackage encapsulates one concept from the 12-month learning plan
(see `docs/LEARNING_PLAN.md`).

Subpackages
-----------
- ``data``        : Data acquisition, cleaning, storage.
- ``universe``    : Tradeable asset universe construction.
- ``costs``       : Transaction cost, slippage and impact models.
- ``indicators``  : Technical indicators (MA, MACD, RSI, ATR, ...).
- ``factors``     : Factor library + single-factor testing framework.
- ``alpha``       : Multi-factor alpha models and signal combination.
- ``portfolio``   : Portfolio optimization and weight construction.
- ``backtest``    : Backtesting engines (vectorised + event-driven).
- ``risk``        : Risk metrics, position sizing, drawdown control.
- ``derivatives`` : Options pricing, Greeks, futures utilities.
- ``crypto``      : Crypto-specific data & strategies (CEX / on-chain).
- ``ml``          : Machine-learning pipelines for quant (with proper CV).
- ``utils``       : Shared helpers, config, logging.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
