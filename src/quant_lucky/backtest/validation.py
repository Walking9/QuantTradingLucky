"""In-sample / out-of-sample splitting and Walk-Forward evaluation.

Why this module exists
----------------------
M5's biggest pedagogical mistake-machine is parameter-tuning on a fixed
history and reporting the resulting Sharpe as if it were out-of-sample.
``notebooks/M05_backtest_traps.ipynb`` showed this trap with a window
sweep on pure noise — peak in-sample Sharpe was a sampling artefact.

This module gives the engine three honest answers to "did my parameters
actually generalise?":

1. **Fixed IS/OOS split** (:func:`fixed_split`)
   Pick a cut date (default: last 30 % of history). Tune on the left
   half, *score once* on the right half. Cheap; the result is a single
   honest number.

2. **Rolling Walk-Forward** (:func:`rolling_walk_forward`)
   Walk a sliding window across history: train on ``[t, t+W)``, test on
   ``[t+W, t+W+T)``, advance by ``step``. Captures non-stationarity (your
   2018 alpha might die in 2022). Gives a concatenated OOS return series.

3. **Anchored Walk-Forward** (:func:`anchored_walk_forward`)
   Grow the training window monotonically (``[t0, t0+W+k*step)``) while
   stepping the test slice forward. Mirrors how a live researcher would
   refit periodically using *all* past data. Smoother OOS curves than
   rolling but slower to react to regime change.

On top of the splits, :func:`walk_forward` runs a user-supplied
parameter grid for every split, picks the best param on the in-sample
slice, and reports the corresponding out-of-sample metric. The selected
parameter may differ from split to split — that is the *point*. A
parameter that stays constant across windows is more robust than one
that flips every step.

Anti-overfit accounting
-----------------------
:func:`deflated_sharpe_ratio` implements Lopez de Prado's (2014) DSR. It
takes the observed Sharpe and applies a haircut that grows with:

* the **number of trials** ``N`` (the more we tried, the higher the
  best-of-N noise floor);
* the **skew / kurtosis** of returns (heavier tails ⇒ higher effective
  variance of the Sharpe estimator).

DSR ≥ 0 is "real beyond multiple-testing noise"; DSR < 0 means the
result is consistent with luck across the trial budget. Always pair a
walk-forward report with its DSR — it's the cheapest possible defence
against the parameter-sweep trap.

References
~~~~~~~~~~
* Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* (2014).
* Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd
  ed., Ch. 5–6 (rolling vs anchored walk-forward conventions).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from quant_lucky.backtest.vector import BacktestResult, VectorEngine


# ---------------------------------------------------------------------------
# Split primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """A single train / test slice over a date index.

    Half-open conventions
    ---------------------
    Both ``train`` and ``test`` slices use *closed-closed* bounds (both
    endpoints inclusive). This matches the way pandas slices a
    DatetimeIndex with ``loc[start:end]`` and is what every test in this
    module asserts on.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.train_end < self.train_start:
            raise ValueError(f"train_end {self.train_end} precedes train_start {self.train_start}")
        if self.test_end < self.test_start:
            raise ValueError(f"test_end {self.test_end} precedes test_start {self.test_start}")
        if self.test_start <= self.train_end:
            # Critical guard: if the test slice overlaps with training, we
            # are fitting on what we then score — that's the classic
            # data-snooping trap this module exists to prevent.
            raise ValueError(
                f"test_start {self.test_start} must be strictly after "
                f"train_end {self.train_end}"
            )

    def slice_train(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.train_start : self.train_end]

    def slice_test(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.test_start : self.test_end]


def fixed_split(
    index: pd.DatetimeIndex,
    *,
    oos_fraction: float = 0.3,
) -> Split:
    """Single-shot IS/OOS split at ``1 - oos_fraction`` of the index.

    The boundary is placed strictly between two adjacent bars so the
    invariant ``test_start > train_end`` holds without needing
    business-day arithmetic.

    Args:
        index: DatetimeIndex of the panel you'll evaluate on.
        oos_fraction: portion reserved for OOS. ``0.3`` keeps the last
            30 % of the index for testing. Must be in ``(0, 1)``.

    Returns:
        A single :class:`Split` covering the entire ``index``.
    """
    if not 0.0 < oos_fraction < 1.0:
        raise ValueError(f"oos_fraction must be in (0, 1), got {oos_fraction}")
    if len(index) < 2:
        raise ValueError(f"index too short to split: n={len(index)}")

    cut = int(len(index) * (1.0 - oos_fraction))
    cut = max(1, min(len(index) - 1, cut))
    return Split(
        train_start=index[0],
        train_end=index[cut - 1],
        test_start=index[cut],
        test_end=index[-1],
    )


def rolling_walk_forward(
    index: pd.DatetimeIndex,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[Split]:
    """Rolling fixed-width Walk-Forward splits.

    The training window is **constant width** and slides forward by
    ``step`` bars each iteration. This is the "Pardo Type 1" Walk-Forward:
    older history rolls off as new history rolls in.

    Args:
        index: DatetimeIndex to split.
        train_size: number of bars used to fit each window.
        test_size: number of bars in each OOS slice.
        step: bars to advance between iterations. Defaults to
            ``test_size`` (contiguous, non-overlapping test slices —
            the convention that lets us concatenate test returns into
            one continuous OOS curve).

    Returns:
        List of :class:`Split` with no test-slice overlap. Empty list if
        the index is too short to produce even one full window.
    """
    if train_size <= 0:
        raise ValueError(f"train_size must be > 0, got {train_size}")
    if test_size <= 0:
        raise ValueError(f"test_size must be > 0, got {test_size}")
    if step is None:
        step = test_size
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    splits: list[Split] = []
    n = len(index)
    train_start = 0
    while True:
        train_end = train_start + train_size - 1
        test_start = train_end + 1
        test_end = min(test_start + test_size - 1, n - 1)
        if test_start >= n or test_end < test_start:
            break
        splits.append(
            Split(
                train_start=index[train_start],
                train_end=index[train_end],
                test_start=index[test_start],
                test_end=index[test_end],
            )
        )
        # Advance the WHOLE window by `step` (both train and test).
        train_start += step
        if train_start + train_size > n:
            break
    return splits


def anchored_walk_forward(
    index: pd.DatetimeIndex,
    *,
    initial_train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[Split]:
    """Anchored Walk-Forward — training window starts at ``index[0]``
    and grows monotonically.

    Mirrors a researcher who keeps refitting on "all data so far" and
    rolling the OOS forward. Use this when you believe the data-
    generating process is roughly stationary and more history is always
    better. If you suspect regime change, prefer
    :func:`rolling_walk_forward`.
    """
    if initial_train_size <= 0:
        raise ValueError(f"initial_train_size must be > 0, got {initial_train_size}")
    if test_size <= 0:
        raise ValueError(f"test_size must be > 0, got {test_size}")
    if step is None:
        step = test_size
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    splits: list[Split] = []
    n = len(index)
    train_size = initial_train_size
    while True:
        train_end = train_size - 1
        test_start = train_end + 1
        test_end = min(test_start + test_size - 1, n - 1)
        if test_start >= n or test_end < test_start:
            break
        splits.append(
            Split(
                train_start=index[0],
                train_end=index[train_end],
                test_start=index[test_start],
                test_end=index[test_end],
            )
        )
        train_size += step
        if train_size > n:
            break
    return splits


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    annualisation: float = 252.0,
) -> float:
    """Lopez de Prado's Deflated Sharpe Ratio (DSR).

    The DSR penalises an observed Sharpe by the expected best-of-N
    Sharpe under the null and inflates the denominator by the Sharpe-
    estimator variance under heavier-tailed returns. The output is a
    probability ``∈ [0, 1]`` that the strategy's *true* Sharpe is > 0.

    DSR ≥ 0.95 is the usual "publishable" threshold. DSR < 0.5 means
    the result is consistent with luck.

    Implementation
    --------------
    Following Bailey & Lopez de Prado (2014), eq. (5):

        DSR = Φ((SR_obs - E[max SR | null]) * sqrt(T - 1) /
                 sqrt(1 - γ_3 SR_obs + (γ_4 - 1)/4 SR_obs²))

    where:

    * ``SR_obs`` is the *non-annualised* (per-period) Sharpe;
    * ``E[max SR | null]`` is the expected maximum of N iid standard
      normals, approximated via the Sasahara expression;
    * ``γ_3`` is sample skewness, ``γ_4`` is sample EXCESS kurtosis
      (i.e., Gaussian → 0).

    Args:
        observed_sharpe: ANNUALISED Sharpe (this is what most people
            quote and what :class:`PerformanceReport` reports). We
            de-annualise internally using ``annualisation``.
        n_trials: number of distinct parameterisations evaluated.
        n_observations: number of in-sample return observations behind
            ``observed_sharpe``. Must be > 1.
        skewness: sample skewness of the OOS return series.
        excess_kurtosis: sample EXCESS kurtosis (subtract 3 from the raw
            kurtosis). 0 for Gaussian.
        annualisation: periods/year used to annualise the input Sharpe.
            252 for daily equities (default).

    Returns:
        Probability in ``[0, 1]``.
    """
    if n_observations <= 1:
        raise ValueError(f"n_observations must be > 1, got {n_observations}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    sr_obs_pp = observed_sharpe / np.sqrt(annualisation)

    # Expected best-of-N Sharpe under the null (returns iid normal):
    # Bailey-Prado closed-form approximation (eq. 4 in the paper):
    # E[max] ≈ (1 - γ) Z^{-1}(1 - 1/N) + γ Z^{-1}(1 - 1/(N e))
    # where γ ≈ 0.5772... is Euler-Mascheroni.
    if n_trials == 1:
        expected_max = 0.0
    else:
        euler_mascheroni = 0.5772156649015329
        # Inverse normal CDF — guard the extreme tails to keep things finite.
        z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
        expected_max = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Sharpe estimator variance (in per-period units), Mertens (2002):
    # var(SR) ≈ (1 - γ_3 SR + (γ_4)/4 SR²) / (T - 1).
    # NB: γ_4 here is excess kurtosis (Gaussian = 0). The original paper
    # writes (γ_4 - 1)/4 because they use raw kurtosis (Gaussian = 3).
    var_factor = 1.0 - skewness * sr_obs_pp + (excess_kurtosis / 4.0) * sr_obs_pp**2
    if var_factor <= 0:
        # Indicates an absurdly fat-tailed sample; treat as no signal.
        return 0.0

    z = (sr_obs_pp - expected_max / np.sqrt(annualisation)) * np.sqrt(
        (n_observations - 1) / var_factor
    )
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Walk-Forward parameter selection
# ---------------------------------------------------------------------------
# A strategy function takes a parameter dict + the FULL price panel and
# returns the weights it would have used on every date in that panel.
# Key contract: the strategy MUST be no-look-ahead — weight at ``t`` may
# only depend on prices ``≤ t-1``. The Walk-Forward driver does NOT
# enforce this; it trusts the strategy. (Why? Because a strategy needs
# lookback for moving averages and the like, so we can't just pass it a
# train slice — we'd amputate the lookback. The honest contract is "use
# full data, but use it honestly.")
#
# Performance note: weights are computed ONCE per ``param`` over the
# full panel, then sliced into train / test windows for evaluation.
# This is both faster (no redundant recompute per split) and clearer
# (the strategy's lookback story stays in one place).
StrategyFn = Callable[[Mapping[str, Any], pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate output of :func:`walk_forward`.

    Attributes
    ----------
    splits:
        The :class:`Split`\\s used (handy if you computed them inline).
    selected_params:
        Best in-sample parameter set per split (same order as ``splits``).
    is_metrics, oos_metrics:
        Metric value (typically Sharpe) per split. ``is_metrics[i]`` is
        the best in-sample metric; ``oos_metrics[i]`` is the OOS metric
        of THE SAME parameter set on the OOS slice. The gap between IS
        and OOS is the headline diagnostic for overfitting.
    oos_returns:
        Concatenated OOS net-return series. Use this for the aggregate
        Sharpe / MDD / equity curve.
    n_trials:
        Total parameter evaluations across all splits. Plugs into
        :func:`deflated_sharpe_ratio` as ``n_trials``.
    deflated_sharpe:
        DSR of ``oos_returns`` given ``n_trials``. 0–1 probability of a
        true positive Sharpe.
    """

    splits: list[Split]
    selected_params: list[dict[str, Any]]
    is_metrics: list[float]
    oos_metrics: list[float]
    oos_returns: pd.Series
    n_trials: int
    deflated_sharpe: float
    extra: dict[str, Any] = field(default_factory=dict)


def _aggregate_oos_metrics(
    oos_returns: pd.Series, periods_per_year: int
) -> tuple[float, float, float, int]:
    """Compute aggregate Sharpe, skew, excess-kurt, n_obs from the
    concatenated OOS series. Helper to keep ``walk_forward`` focused.

    Returns ``(sharpe, skewness, excess_kurtosis, n_observations)``.
    Sharpe is NaN if the series is too short or vol-degenerate.
    """
    clean = oos_returns.dropna()
    if len(clean) <= 1:
        return float("nan"), 0.0, 0.0, len(clean)

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    ann_return = mean * periods_per_year
    ann_vol = std * np.sqrt(periods_per_year) if std > 0 else float("nan")
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    skew = float(clean.skew()) if len(clean) > 2 else 0.0
    # pandas .kurt() returns EXCESS kurtosis (Gaussian = 0) — exactly
    # what DSR expects.
    excess_kurt = float(clean.kurt()) if len(clean) > 3 else 0.0
    return sharpe, skew, excess_kurt, len(clean)


def walk_forward(  # noqa: PLR0912 — the loop body is intentionally explicit
    *,
    prices: pd.DataFrame,
    splits: list[Split],
    param_grid: Iterable[Mapping[str, Any]],
    strategy_fn: StrategyFn,
    cost_bps: float,
    metric_fn: Callable[[BacktestResult], float] | None = None,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """Run a Walk-Forward parameter sweep on ``prices``.

    Algorithm
    ---------
    1. For each ``param`` in ``param_grid``, call ``strategy_fn(param,
       prices)`` ONCE to get a full-history weights DataFrame. The
       strategy is trusted to be no-look-ahead.
    2. For each :class:`Split`:
       a. For every param, evaluate the engine on the train slice (weights
          and prices both restricted to the train dates) and read the
          ``metric_fn`` score.
       b. Pick the param with the highest score → ``selected_params[i]``.
       c. Evaluate that same param's weights on the test slice → record
          OOS metric and stash net returns.
    3. Concatenate all test slices' net returns into ``oos_returns``.
    4. Compute aggregate OOS Sharpe and pass it to
       :func:`deflated_sharpe_ratio` with
       ``n_trials = len(param_grid) * len(splits)`` (every IS evaluation
       counted; this is the conservative trial count).

    Args:
        prices: ``date × asset`` close prices. The full history; do NOT
            pre-slice to a window.
        splits: list of :class:`Split` (e.g., from
            :func:`rolling_walk_forward`).
        param_grid: iterable of parameter dicts to evaluate. Materialised
            internally so we can count trials.
        strategy_fn: ``(param_dict, full_prices) -> full_weights_df``.
            Must be no-look-ahead.
        cost_bps: per-side bps cost used in every VectorEngine run.
        metric_fn: ``BacktestResult -> float`` (higher is better). Defaults
            to ``result.report.sharpe``.
        periods_per_year: forwarded to the engine and to the DSR.

    Returns:
        :class:`WalkForwardResult`.
    """
    if not splits:
        raise ValueError("splits must contain at least one Split")
    param_list = [dict(p) for p in param_grid]
    if not param_list:
        raise ValueError("param_grid is empty")

    if metric_fn is None:

        def metric_fn(r: BacktestResult) -> float:
            return r.report.sharpe

    engine = VectorEngine(cost_bps=cost_bps, periods_per_year=periods_per_year)

    # Pre-compute weights per param over the FULL history. This is the
    # main perf win: a strategy that needs a long lookback (e.g., 252-day
    # moving average) doesn't have to recompute it for each window. It
    # also avoids the bug where a per-split call to strategy_fn would
    # see an amputated history and silently degrade to all-zeros.
    weights_by_param: list[pd.DataFrame] = []
    for param in param_list:
        w_full = strategy_fn(param, prices)
        if not isinstance(w_full, pd.DataFrame):
            raise TypeError(f"strategy_fn must return a DataFrame, got {type(w_full).__name__}")
        # Align to the prices grid so engine.run won't trim later.
        weights_by_param.append(
            w_full.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        )

    selected_params: list[dict[str, Any]] = []
    is_metrics: list[float] = []
    oos_metrics: list[float] = []
    oos_chunks: list[pd.Series] = []
    n_trials = 0

    for split in splits:
        train_prices = split.slice_train(prices)
        test_prices = split.slice_test(prices)
        if train_prices.empty or test_prices.empty:
            # Skipping rather than raising — caller may have an off-by-one
            # split. Surface this in ``extra`` instead of crashing the run.
            continue

        # In-sample sweep.
        best_score = -np.inf
        best_idx: int | None = None
        for i, w_full in enumerate(weights_by_param):
            w_train = split.slice_train(w_full)
            res_is = engine.run(w_train, train_prices)
            score = metric_fn(res_is)
            n_trials += 1
            # NaN scores (e.g., constant returns → Sharpe NaN) lose to a
            # finite score automatically because `np.nan > -inf` is False.
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            # No usable score on this window. Skip.
            continue

        # Out-of-sample evaluation with the chosen parameter.
        w_test = split.slice_test(weights_by_param[best_idx])
        res_oos = engine.run(w_test, test_prices)
        oos_score = metric_fn(res_oos)

        selected_params.append(param_list[best_idx])
        is_metrics.append(best_score)
        oos_metrics.append(oos_score)
        oos_chunks.append(res_oos.net_returns)

    if not oos_chunks:
        raise RuntimeError("No split produced an OOS evaluation; check splits vs price index.")

    oos_returns = pd.concat(oos_chunks).sort_index()
    # Aggregate Sharpe (over the concatenated OOS series) — this is the
    # number that DSR will deflate.
    agg_sharpe, skew, excess_kurt, n_obs = _aggregate_oos_metrics(oos_returns, periods_per_year)

    if np.isfinite(agg_sharpe):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=agg_sharpe,
            n_trials=n_trials,
            n_observations=n_obs,
            skewness=skew,
            excess_kurtosis=excess_kurt,
            annualisation=float(periods_per_year),
        )
    else:
        dsr = float("nan")

    return WalkForwardResult(
        splits=splits,
        selected_params=selected_params,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        oos_returns=oos_returns,
        n_trials=n_trials,
        deflated_sharpe=dsr,
        extra={
            "aggregate_oos_sharpe": agg_sharpe,
            "aggregate_oos_skewness": skew,
            "aggregate_oos_excess_kurtosis": excess_kurt,
        },
    )


__all__ = [
    "Split",
    "WalkForwardResult",
    "anchored_walk_forward",
    "deflated_sharpe_ratio",
    "fixed_split",
    "rolling_walk_forward",
    "walk_forward",
]
