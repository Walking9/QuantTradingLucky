"""Risk management primitives.

Pre-trade, intra-trade, and post-trade:

- **Measures**: volatility, VaR (historical / parametric / MC), CVaR (ES),
  max drawdown, downside deviation, tail ratio.
- **Sizing**: fixed fraction, volatility targeting, Kelly (fractional),
  risk parity allocation.
- **Limits**: per-position, per-industry, per-factor exposure, gross and
  net leverage, concentration caps.
- **Stops**: ATR stop, time stop, trailing stop, equity-curve stop.
- **Stress tests**: historical scenarios (2008, 2015, 2020-03, 2022 Luna)
  and Monte Carlo perturbations.

Every strategy routed to paper or live trading MUST pass through this
module — risk limits are enforced, not advisory.
"""
