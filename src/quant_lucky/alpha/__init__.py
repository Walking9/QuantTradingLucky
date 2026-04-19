"""Alpha signal combination.

Starting from individual factors (``quant_lucky.factors``), produce a
unified alpha score per asset per day using:

- Equal weighting, IC-weighted, ICIR-maximising.
- Shrinkage / regularised regression (Ridge, Lasso).
- Non-linear models (GBDT, NN) — see ``quant_lucky.ml``.

Outputs a *single cross-sectional score* that downstream
``portfolio`` optimisation converts into weights.
"""
