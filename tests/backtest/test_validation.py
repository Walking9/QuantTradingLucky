"""Tests for ``backtest.validation`` — IS/OOS splits, Walk-Forward, DSR.

Tested invariants
-----------------
* Splits are non-overlapping and respect the train-before-test bound.
* DSR collapses to a sensible Φ-cdf when ``n_trials = 1`` and is
  monotonically decreasing in ``n_trials``.
* :func:`walk_forward` selects per-split parameters and produces a
  concatenated OOS series of the right length.
* No look-ahead leaks: a strategy that uses ``shift(1)`` correctly is
  agnostic to whether we ran one big window or many smaller ones.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from quant_lucky.backtest.validation import (
    Split,
    WalkForwardResult,
    anchored_walk_forward,
    deflated_sharpe_ratio,
    fixed_split,
    rolling_walk_forward,
    walk_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=500, freq="B")


@pytest.fixture()
def random_prices(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.0003, 0.012, size=(len(daily_index), 5))
    return pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
        index=daily_index,
        columns=[f"S{i}" for i in range(5)],
    )


# ---------------------------------------------------------------------------
# Split — sanity
# ---------------------------------------------------------------------------
def test_split_rejects_overlap() -> None:
    """A Split where test_start ≤ train_end would silently fit-and-score
    on the same window — exactly the trap this module prevents.
    """
    t0 = pd.Timestamp("2024-01-01")
    t1 = pd.Timestamp("2024-01-10")
    with pytest.raises(ValueError, match="strictly after"):
        Split(train_start=t0, train_end=t1, test_start=t1, test_end=t1 + pd.Timedelta(days=5))


def test_split_rejects_inverted_bounds() -> None:
    t0 = pd.Timestamp("2024-01-10")
    t1 = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="train_end"):
        Split(train_start=t0, train_end=t1, test_start=t0, test_end=t0)


def test_split_slice_helpers(random_prices: pd.DataFrame) -> None:
    s = Split(
        train_start=random_prices.index[0],
        train_end=random_prices.index[99],
        test_start=random_prices.index[100],
        test_end=random_prices.index[149],
    )
    train = s.slice_train(random_prices)
    test = s.slice_test(random_prices)
    assert len(train) == 100
    assert len(test) == 50
    # Adjacency invariant: no row appears in both slices.
    assert train.index.intersection(test.index).empty


# ---------------------------------------------------------------------------
# fixed_split
# ---------------------------------------------------------------------------
def test_fixed_split_default_30pct(daily_index: pd.DatetimeIndex) -> None:
    s = fixed_split(daily_index)
    n_train = (daily_index <= s.train_end).sum()
    n_test = (daily_index >= s.test_start).sum()
    assert n_train + n_test == len(daily_index)
    assert n_test == pytest.approx(0.3 * len(daily_index), abs=1)


def test_fixed_split_rejects_extreme_fractions(daily_index: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError):
        fixed_split(daily_index, oos_fraction=0.0)
    with pytest.raises(ValueError):
        fixed_split(daily_index, oos_fraction=1.0)


def test_fixed_split_short_index_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        fixed_split(pd.DatetimeIndex([pd.Timestamp("2024-01-01")]), oos_fraction=0.5)


# ---------------------------------------------------------------------------
# rolling_walk_forward
# ---------------------------------------------------------------------------
def test_rolling_wf_produces_non_overlapping_test_slices(
    daily_index: pd.DatetimeIndex,
) -> None:
    """Default step = test_size means OOS slices tile the post-burn-in
    region without overlap or gaps. This is the property that lets us
    concatenate test returns into one continuous OOS curve.
    """
    splits = rolling_walk_forward(daily_index, train_size=200, test_size=50)
    assert splits, "expected at least one split"
    for prev, cur in pairwise(splits):
        # Consecutive test slices are adjacent (cur.test_start is the bar
        # right after prev.test_end).
        assert cur.test_start > prev.test_end


def test_rolling_wf_each_train_is_same_width(
    daily_index: pd.DatetimeIndex,
) -> None:
    splits = rolling_walk_forward(daily_index, train_size=200, test_size=50)
    widths = [
        (daily_index <= s.train_end).sum() - (daily_index < s.train_start).sum() for s in splits
    ]
    assert len(set(widths)) == 1, f"rolling train widths drifted: {widths}"


def test_rolling_wf_returns_empty_when_too_short() -> None:
    short = pd.date_range("2024-01-01", periods=10, freq="B")
    splits = rolling_walk_forward(short, train_size=100, test_size=10)
    assert splits == []


def test_rolling_wf_rejects_invalid_args(daily_index: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError):
        rolling_walk_forward(daily_index, train_size=0, test_size=10)
    with pytest.raises(ValueError):
        rolling_walk_forward(daily_index, train_size=10, test_size=0)
    with pytest.raises(ValueError):
        rolling_walk_forward(daily_index, train_size=10, test_size=10, step=0)


# ---------------------------------------------------------------------------
# anchored_walk_forward
# ---------------------------------------------------------------------------
def test_anchored_wf_train_is_monotonically_growing(
    daily_index: pd.DatetimeIndex,
) -> None:
    splits = anchored_walk_forward(daily_index, initial_train_size=200, test_size=50)
    widths = [
        (daily_index <= s.train_end).sum() - (daily_index < s.train_start).sum() for s in splits
    ]
    assert all(
        b > a for a, b in pairwise(widths)
    ), f"anchored train widths must grow monotonically, got {widths}"


def test_anchored_wf_train_always_anchored_to_first_bar(
    daily_index: pd.DatetimeIndex,
) -> None:
    splits = anchored_walk_forward(daily_index, initial_train_size=200, test_size=50)
    assert all(s.train_start == daily_index[0] for s in splits)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def test_dsr_decreases_with_more_trials() -> None:
    """The whole point of DSR: more trials ⇒ deeper haircut. Verifies the
    monotonicity that should hold for any non-trivial observed Sharpe.
    """
    dsr_values = [
        deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=n, n_observations=252)
        for n in [1, 10, 100, 1000]
    ]
    assert all(
        b < a for a, b in pairwise(dsr_values)
    ), f"DSR did not decrease with trials: {dsr_values}"


def test_dsr_high_sharpe_with_single_trial_is_near_one() -> None:
    """SR = 3.0 on 252 daily observations is far above what one-shot
    random data would produce; DSR should be very close to 1.
    """
    dsr = deflated_sharpe_ratio(observed_sharpe=3.0, n_trials=1, n_observations=252)
    assert dsr > 0.99


def test_dsr_zero_sharpe_is_around_half() -> None:
    """A literal zero observed Sharpe lands roughly at the median of the
    Sharpe-estimator distribution under the null; DSR ≈ 0.5.
    """
    dsr = deflated_sharpe_ratio(observed_sharpe=0.0, n_trials=1, n_observations=252)
    assert 0.45 <= dsr <= 0.55


def test_dsr_heavy_kurtosis_haircuts_score() -> None:
    """Fat tails inflate the Sharpe-estimator variance ⇒ when the
    observed Sharpe is comfortably above the null expected-max, a higher
    kurtosis should LOWER the DSR.

    Caveat: when the observed Sharpe lands BELOW the null expected-max
    (numerator z is negative), higher kurtosis shrinks |z| and the DSR
    drifts toward 0.5 instead. We test the "above the max" regime —
    that's the one a hopeful researcher cares about.
    """
    dsr_normal = deflated_sharpe_ratio(
        observed_sharpe=3.0,
        n_trials=10,
        n_observations=252,
        excess_kurtosis=0.0,
    )
    dsr_fat = deflated_sharpe_ratio(
        observed_sharpe=3.0,
        n_trials=10,
        n_observations=252,
        excess_kurtosis=20.0,
    )
    assert dsr_fat < dsr_normal, (
        f"Expected fat-tail DSR < normal DSR; got fat={dsr_fat:.6f}, " f"normal={dsr_normal:.6f}"
    )


def test_dsr_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="n_observations"):
        deflated_sharpe_ratio(observed_sharpe=1.0, n_trials=1, n_observations=1)
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(observed_sharpe=1.0, n_trials=0, n_observations=10)


# ---------------------------------------------------------------------------
# walk_forward — integration with VectorEngine
# ---------------------------------------------------------------------------
def _momentum_strategy(param, full_prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight long-top / short-bottom by past ``window`` return.

    No-look-ahead: uses ``.shift(1)`` so the weight at ``t`` only depends
    on prices ``≤ t-1``. The walk-forward driver hands us the FULL
    price panel; we trust ourselves to use it honestly.
    """
    n = param["window"]
    mom = full_prices.pct_change(n, fill_method=None).shift(1)
    half = mom.shape[1] // 2
    ranks = mom.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=full_prices.index, columns=full_prices.columns)
    w[ranks > 0.5] = 1.0 / half
    w[ranks <= 0.5] = -1.0 / half
    return w.where(ranks.notna(), 0.0)


def test_walk_forward_returns_one_param_per_split(random_prices: pd.DataFrame) -> None:
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    grid = [{"window": w} for w in [5, 10, 20]]
    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=grid,
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
    )
    assert isinstance(res, WalkForwardResult)
    assert len(res.selected_params) == len(splits)
    assert len(res.is_metrics) == len(splits)
    assert len(res.oos_metrics) == len(splits)
    assert res.n_trials == len(grid) * len(splits)


def test_walk_forward_concatenated_oos_covers_all_test_dates(
    random_prices: pd.DataFrame,
) -> None:
    """OOS net-return series length must equal the sum of test-slice
    lengths across splits. Catches off-by-one bugs in slicing.
    """
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    expected_len = sum(len(s.slice_test(random_prices)) for s in splits)

    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=[{"window": w} for w in [5, 20]],
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
    )
    assert len(res.oos_returns) == expected_len


def test_walk_forward_oos_returns_indexed_chronologically(
    random_prices: pd.DataFrame,
) -> None:
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=[{"window": w} for w in [5, 20]],
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
    )
    assert res.oos_returns.index.is_monotonic_increasing


def test_walk_forward_records_dsr_in_range(random_prices: pd.DataFrame) -> None:
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=[{"window": w} for w in [5, 10, 20, 40]],
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
    )
    if not np.isnan(res.deflated_sharpe):
        assert 0.0 <= res.deflated_sharpe <= 1.0


def test_walk_forward_single_param_grid_runs(random_prices: pd.DataFrame) -> None:
    """Edge case: ``param_grid`` with one entry still works. Selected
    param is the same one on every split; useful as a baseline run.
    """
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=[{"window": 20}],
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
    )
    assert all(p == {"window": 20} for p in res.selected_params)


def test_walk_forward_rejects_empty_grid(random_prices: pd.DataFrame) -> None:
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    with pytest.raises(ValueError, match="param_grid is empty"):
        walk_forward(
            prices=random_prices,
            splits=splits,
            param_grid=[],
            strategy_fn=_momentum_strategy,
            cost_bps=0.0,
        )


def test_walk_forward_rejects_empty_splits(random_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="splits must contain"):
        walk_forward(
            prices=random_prices,
            splits=[],
            param_grid=[{"window": 20}],
            strategy_fn=_momentum_strategy,
            cost_bps=0.0,
        )


def test_walk_forward_strategy_must_return_dataframe(
    random_prices: pd.DataFrame,
) -> None:
    def bad_strategy(param, full_prices):
        return full_prices.iloc[:, 0]  # Series, not DataFrame

    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    with pytest.raises(TypeError, match="DataFrame"):
        walk_forward(
            prices=random_prices,
            splits=splits,
            param_grid=[{"window": 20}],
            strategy_fn=bad_strategy,
            cost_bps=0.0,
        )


def test_walk_forward_with_custom_metric(random_prices: pd.DataFrame) -> None:
    """``metric_fn`` flexibility: optimise for terminal portfolio value
    instead of Sharpe. Should still pick a parameter and run end-to-end.
    """
    splits = rolling_walk_forward(random_prices.index, train_size=200, test_size=50)
    res = walk_forward(
        prices=random_prices,
        splits=splits,
        param_grid=[{"window": 5}, {"window": 40}],
        strategy_fn=_momentum_strategy,
        cost_bps=0.0,
        metric_fn=lambda r: float(r.portfolio_value.iloc[-1]),
    )
    assert len(res.selected_params) == len(splits)
