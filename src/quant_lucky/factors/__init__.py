"""Factor library and single-factor testing framework.

A factor is a cross-sectional signal :math:`f_{i,t}` that ranks assets at
time ``t``. This subpackage provides:

- A catalogue of factor definitions (value, momentum, quality, volatility, ...).
- A standardised tester that produces, for any factor:

  * Information Coefficient (IC) time series + IC-IR
  * Quantile-portfolio cumulative returns (typically 5 groups)
  * Long-short return + max drawdown
  * Turnover and decay analysis
  * Industry / size exposure

Neutralisation helpers (industry-neutral, size-neutral, Barra-neutral)
live here too, because factor evaluation is meaningless without them.
"""
