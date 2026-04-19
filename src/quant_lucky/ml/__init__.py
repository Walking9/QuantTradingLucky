"""Machine-learning pipelines for quant finance.

Financial ML has traps that generic ML does not. This module enforces:

- **Purged K-Fold + Embargo** cross-validation (López de Prado).
- **Triple-Barrier labelling** for event-driven classification targets.
- **Meta-labelling** to layer sizing on top of a primary model.
- **Fractional differentiation** to obtain stationary series that retain
  memory.
- Explainability by default (SHAP on every trained model).

Model zoo (filled in month 10):
- Linear: Lasso, Ridge, ElasticNet.
- Tree-based: LightGBM, XGBoost, Random Forest.
- Sequence (experimental): LSTM / Transformer with strict guardrails.

Reinforcement learning is deliberately **out of scope** for now —
sample inefficiency and non-stationarity make it unsuitable for a
solo learner in year one.
"""
