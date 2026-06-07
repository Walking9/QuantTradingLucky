"""Tests for the shared strategy-evaluation harness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_lucky.backtest.validation import rolling_walk_forward, walk_forward
from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.evaluation import (
    ResearchSpec,
    attribution,
    evaluate_strategy,
    is_oos_split,
    parameter_sensitivity,
    run_research,
    write_artifacts,
)
from quant_lucky.strategies.momentum import (
    cross_sectional_momentum_weights,
    momentum_strategy_fn,
)
from quant_lucky.strategies.risk_parity import equal_weight_weights, inverse_vol_weights


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_price_panel(n_assets=10, n_days=500, seed=17)


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
class TestAttribution:
    def test_recovers_known_beta_and_alpha(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2021-01-01", periods=600)
        bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
        true_beta, true_alpha_pp = 0.8, 0.0002
        # Tight idiosyncratic noise so the OLS estimates are precise enough
        # to assert on (alpha's annualised standard error stays << tol).
        noise = pd.Series(rng.normal(0.0, 0.0005, len(idx)), index=idx)
        strat = true_alpha_pp + true_beta * bench + noise

        a = attribution(strat, bench, periods_per_year=252)
        assert a.beta == pytest.approx(true_beta, abs=0.02)
        assert a.alpha_annual == pytest.approx(true_alpha_pp * 252, abs=0.02)
        assert 0.0 <= a.r_squared <= 1.0
        assert a.n_obs == len(idx)

    def test_market_neutral_has_zero_beta(self) -> None:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2021-01-01", periods=400)
        bench = pd.Series(rng.normal(0.0005, 0.012, len(idx)), index=idx)
        # Strategy return independent of the benchmark → beta ≈ 0.
        strat = pd.Series(rng.normal(0.0001, 0.008, len(idx)), index=idx)
        a = attribution(strat, bench, periods_per_year=252)
        assert abs(a.beta) < 0.15

    def test_too_few_observations(self) -> None:
        s = pd.Series([0.01, 0.02], index=pd.bdate_range("2021-01-01", periods=2))
        a = attribution(s, s)
        assert a.n_obs == 2
        assert np.isnan(a.beta)

    def test_degenerate_benchmark(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=50)
        strat = pd.Series(np.linspace(0, 0.05, 50), index=idx)
        flat = pd.Series(np.zeros(50), index=idx)  # zero-variance benchmark
        a = attribution(strat, flat)
        assert np.isnan(a.beta)

    def test_misaligned_index_inner_joins(self) -> None:
        idx = pd.bdate_range("2021-01-01", periods=100)
        a_ser = pd.Series(np.random.default_rng(2).normal(0, 0.01, 100), index=idx)
        b_ser = a_ser.iloc[20:80]  # overlap window only
        res = attribution(a_ser, b_ser)
        assert res.n_obs == 60


# ---------------------------------------------------------------------------
# is_oos_split
# ---------------------------------------------------------------------------
class TestIsOosSplit:
    def test_partitions_timeline(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(panel, lookback=120, skip=21)
        ev = is_oos_split(w, panel, cost_bps=10.0, oos_fraction=0.3)
        s = ev.split
        # Test era starts strictly after the train era ends (no overlap).
        assert s.test_start > s.train_end
        # Both eras have observations.
        assert ev.is_report.n_periods > 0
        assert ev.oos_report.n_periods > 0

    def test_oos_fraction_controls_split(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(panel, lookback=120, skip=21)
        small = is_oos_split(w, panel, cost_bps=10.0, oos_fraction=0.2)
        large = is_oos_split(w, panel, cost_bps=10.0, oos_fraction=0.5)
        assert large.oos_report.n_periods > small.oos_report.n_periods

    def test_reports_are_finite_or_nan_not_crash(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=63, rebalance="M")
        ev = is_oos_split(w, panel, cost_bps=15.0)
        # Sharpe may be any sign on a no-edge panel, but must be a float.
        assert isinstance(ev.oos_report.sharpe, float)


# ---------------------------------------------------------------------------
# parameter_sensitivity
# ---------------------------------------------------------------------------
class TestParameterSensitivity:
    def test_shape_and_columns(self, panel: pd.DataFrame) -> None:
        df = parameter_sensitivity(
            momentum_strategy_fn,
            panel,
            base_param={"lookback": 120, "skip": 21, "top_quantile": 0.2},
            sweep={"lookback": [60, 120, 180], "top_quantile": [0.1, 0.2, 0.3]},
            cost_bps=10.0,
        )
        assert len(df) == 6
        for col in ["param", "value", "is_base", "full_sharpe", "oos_sharpe"]:
            assert col in df.columns

    def test_base_flag_marks_base_value(self, panel: pd.DataFrame) -> None:
        df = parameter_sensitivity(
            momentum_strategy_fn,
            panel,
            base_param={"lookback": 120, "skip": 21},
            sweep={"lookback": [60, 120, 180]},
            cost_bps=10.0,
        )
        base_rows = df[df["is_base"]]
        assert len(base_rows) == 1
        assert base_rows.iloc[0]["value"] == 120

    def test_base_value_reproduces_full_backtest(self, panel: pd.DataFrame) -> None:
        from quant_lucky.backtest import VectorEngine

        base = {"lookback": 100, "skip": 10, "top_quantile": 0.2}
        df = parameter_sensitivity(
            momentum_strategy_fn,
            panel,
            base_param=base,
            sweep={"lookback": [100]},
            cost_bps=12.0,
        )
        w = momentum_strategy_fn(base, panel)
        direct = VectorEngine(cost_bps=12.0).run(w, panel).report.sharpe
        swept = df.iloc[0]["full_sharpe"]
        assert swept == pytest.approx(direct, nan_ok=True)


# ---------------------------------------------------------------------------
# evaluate_strategy bundle
# ---------------------------------------------------------------------------
class TestEvaluateStrategy:
    def test_bundle_populated(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=63, rebalance="M")
        bench = equal_weight_weights(panel, rebalance="M")
        ev = evaluate_strategy(
            name="rp_test",
            prices=panel,
            weights=w,
            benchmark_weights=bench,
            cost_bps=15.0,
            data_source="synthetic",
        )
        assert ev.name == "rp_test"
        assert ev.data_source == "synthetic"
        assert ev.n_assets == panel.shape[1]
        assert ev.benchmark is not None
        assert ev.attribution is not None
        assert ev.span[0] < ev.span[1]

    def test_no_benchmark_no_attribution(self, panel: pd.DataFrame) -> None:
        w = cross_sectional_momentum_weights(panel, lookback=120, skip=21)
        ev = evaluate_strategy(name="m", prices=panel, weights=w, cost_bps=10.0)
        assert ev.benchmark is None
        assert ev.attribution is None

    def test_metrics_dict_is_json_serialisable(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=63)
        bench = equal_weight_weights(panel)
        ev = evaluate_strategy(
            name="rp",
            prices=panel,
            weights=w,
            benchmark_weights=bench,
            cost_bps=15.0,
        )
        blob = json.dumps(ev.metrics_dict())  # must not raise
        loaded = json.loads(blob)
        assert loaded["name"] == "rp"
        assert "full" in loaded and "out_of_sample" in loaded
        assert "attribution" in loaded

    def test_to_markdown_has_sections(self, panel: pd.DataFrame) -> None:
        w = inverse_vol_weights(panel, vol_window=63)
        bench = equal_weight_weights(panel)
        sens = parameter_sensitivity(
            lambda p, px: inverse_vol_weights(px, vol_window=int(p["vol_window"])),
            panel,
            base_param={"vol_window": 63},
            sweep={"vol_window": [42, 63, 84]},
            cost_bps=15.0,
        )
        ev = evaluate_strategy(
            name="rp",
            prices=panel,
            weights=w,
            benchmark_weights=bench,
            cost_bps=15.0,
            sensitivity=sens,
        )
        md = ev.to_markdown()
        assert "Headline metrics" in md
        assert "Attribution vs benchmark" in md
        assert "Parameter sensitivity" in md

    def test_carries_walk_forward(self, panel: pd.DataFrame) -> None:
        splits = rolling_walk_forward(pd.DatetimeIndex(panel.index), train_size=200, test_size=60)
        wf = walk_forward(
            prices=panel,
            splits=splits,
            param_grid=[{"lookback": 90}, {"lookback": 150}],
            strategy_fn=momentum_strategy_fn,
            cost_bps=10.0,
        )
        w = momentum_strategy_fn({"lookback": 120}, panel)
        ev = evaluate_strategy(
            name="m",
            prices=panel,
            weights=w,
            cost_bps=10.0,
            walk_forward_result=wf,
        )
        assert ev.walk_forward is not None
        md = ev.to_markdown()
        assert "Walk-Forward" in md
        assert "Deflated Sharpe" in md
        # walk_forward metrics surface in the JSON too.
        assert "walk_forward" in ev.metrics_dict()


# ---------------------------------------------------------------------------
# run_research driver + write_artifacts
# ---------------------------------------------------------------------------
class TestRunResearch:
    def _spec(self, panel: pd.DataFrame) -> ResearchSpec:
        base = {"vol_window": 63, "rebalance": "M"}
        return ResearchSpec(
            name="rp_research",
            prices=panel,
            weights=inverse_vol_weights(panel, vol_window=63, rebalance="M"),
            cost_bps=15.0,
            data_source="synthetic",
            benchmark_weights=equal_weight_weights(panel, rebalance="M"),
            strategy_fn=lambda p, px: inverse_vol_weights(
                px, vol_window=int(p["vol_window"]), rebalance=str(p.get("rebalance", "M"))
            ),
            param_grid=[{"vol_window": 42}, {"vol_window": 63}, {"vol_window": 84}],
            wf_train_size=200,
            wf_test_size=60,
            base_param=base,
            sensitivity_sweep={"vol_window": [42, 63, 84]},
        )

    def test_full_run_populates_everything(self, panel: pd.DataFrame) -> None:
        ev = run_research(self._spec(panel))
        assert ev.benchmark is not None
        assert ev.attribution is not None
        assert ev.walk_forward is not None
        assert ev.sensitivity is not None
        assert not ev.sensitivity.empty

    def test_skips_walk_forward_when_panel_too_short(self) -> None:
        short = synthetic_price_panel(n_assets=4, n_days=120, seed=3)
        spec = ResearchSpec(
            name="short",
            prices=short,
            weights=inverse_vol_weights(short, vol_window=20),
            cost_bps=15.0,
            strategy_fn=lambda p, px: inverse_vol_weights(px, vol_window=int(p["vol_window"])),
            param_grid=[{"vol_window": 20}],
            wf_train_size=10_000,  # larger than the panel → no splits
            wf_test_size=5_000,
        )
        ev = run_research(spec)
        assert ev.walk_forward is None  # degrades gracefully

    def test_write_artifacts_creates_files(self, panel: pd.DataFrame, tmp_path) -> None:
        ev = run_research(self._spec(panel))
        written = write_artifacts(ev, tmp_path / "reports")
        names = {p.name for p in written}
        assert {"metrics.json", "RESULTS.md", "equity_curve.csv", "sensitivity.csv"} <= names
        # metrics.json round-trips and equity curve has both columns.
        loaded = json.loads((tmp_path / "reports" / "metrics.json").read_text())
        assert loaded["name"] == "rp_research"
        curve = pd.read_csv(tmp_path / "reports" / "equity_curve.csv", index_col=0)
        assert "strategy_net" in curve.columns
        assert "benchmark_net" in curve.columns
