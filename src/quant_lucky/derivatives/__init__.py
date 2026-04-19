"""Derivatives: options and futures utilities.

- Black-Scholes / Black-76 pricers with all five Greeks.
- Monte Carlo pricers for path-dependent / early-exercise products.
- Implied volatility solver (Brent / Newton with fallback).
- Volatility surface builders (SVI, SABR) — progressive depth.
- Futures utilities: continuous-contract construction (back-adjusted,
  ratio-adjusted), roll-yield calculation, contango/backwardation flags.

For production use we rely on QuantLib; this module focuses on *educational*
clarity — every formula is derivable from the docstrings.
"""
