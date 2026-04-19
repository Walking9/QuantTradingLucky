"""Portfolio construction and optimisation.

Converts alpha scores + risk model into actual target weights subject to
realistic constraints:

- Mean-variance (Markowitz) with covariance shrinkage (Ledoit-Wolf).
- Risk parity / equal risk contribution.
- Max-diversification, Black-Litterman.
- Constraints: industry/style neutrality, turnover penalty, position
  limits, short-sale restrictions (A-share context).

Implementations use ``cvxpy`` for clarity; performance-critical inner
loops can fall back to ``scipy.optimize`` if needed.
"""
