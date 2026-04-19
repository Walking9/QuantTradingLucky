"""Technical indicators implemented from scratch.

Each indicator is:
- Implemented in vectorised NumPy / Pandas (no Python loops in hot paths).
- Unit-tested against TA-Lib or well-known reference implementations.
- Documented with its mathematical definition and edge cases
  (warm-up period, NaN handling, division-by-zero).

Indicators here are **features**, not factors. A factor must additionally
pass IC / turnover / stability checks in ``quant_lucky.factors``.
"""
