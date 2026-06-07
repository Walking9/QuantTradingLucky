"""Tests for the strategy-package CLI helpers (config + summary)."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lucky.strategies.cli import format_summary, load_config
from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.evaluation import run_research
from quant_lucky.strategies.risk_parity import (
    equal_weight_weights,
    inverse_vol_weights,
    risk_parity_strategy_fn,
)

DEFAULTS = {
    "name": "demo",
    "cost_bps": 10.0,
    "signal": {"vol_window": 63, "rebalance": "M"},
    "nested": {"a": {"b": 1}},
}


class TestLoadConfig:
    def test_defaults_only(self) -> None:
        cfg = load_config(DEFAULTS)
        assert cfg == DEFAULTS
        assert cfg is not DEFAULTS  # deep-copied, caller's dict untouched

    def test_yaml_deep_merges(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("signal:\n  vol_window: 21\ncost_bps: 25.0\n", encoding="utf-8")
        cfg = load_config(DEFAULTS, config_path=path)
        assert cfg["cost_bps"] == 25.0
        assert cfg["signal"]["vol_window"] == 21
        assert cfg["signal"]["rebalance"] == "M"  # untouched key survives merge

    def test_missing_yaml_is_ignored(self, tmp_path) -> None:
        cfg = load_config(DEFAULTS, config_path=tmp_path / "nope.yaml")
        assert cfg["cost_bps"] == 10.0

    def test_overrides_parse_scalar_types(self) -> None:
        cfg = load_config(
            DEFAULTS,
            overrides=["cost_bps=30", "signal.rebalance=W", "signal.vol_window=42"],
        )
        assert cfg["cost_bps"] == 30  # parsed as int/number, not "30"
        assert cfg["signal"]["rebalance"] == "W"
        assert cfg["signal"]["vol_window"] == 42

    def test_override_none_and_bool(self) -> None:
        cfg = load_config(DEFAULTS, overrides=["signal.vol_max=null", "signal.flag=true"])
        assert cfg["signal"]["vol_max"] is None
        assert cfg["signal"]["flag"] is True

    def test_override_creates_nested_key(self) -> None:
        cfg = load_config(DEFAULTS, overrides=["walk_forward.train_size=200"])
        assert cfg["walk_forward"]["train_size"] == 200

    def test_bad_override_raises(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            load_config(DEFAULTS, overrides=["no_equals_sign"])

    def test_non_mapping_yaml_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(DEFAULTS, config_path=path)


class TestFormatSummary:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        return synthetic_price_panel(n_assets=4, n_days=500, seed=8)

    def test_contains_headline_fields(self, panel: pd.DataFrame) -> None:
        spec = _spec(panel)
        text = format_summary(run_research(spec))
        for token in ("demo", "data: synthetic", "full", "in-sample", "out-sample"):
            assert token in text

    def test_includes_benchmark_attribution_and_wf(self, panel: pd.DataFrame) -> None:
        text = format_summary(run_research(_spec(panel)))
        assert "benchmark" in text
        assert "attribution vs benchmark" in text
        assert "walk-forward" in text
        assert "DSR" in text

    def test_minimal_evaluation_without_extras(self, panel: pd.DataFrame) -> None:
        # No benchmark / walk-forward / attribution → summary still renders.
        from quant_lucky.strategies.evaluation import evaluate_strategy

        w = inverse_vol_weights(panel, vol_window=63)
        ev = evaluate_strategy(name="bare", prices=panel, weights=w, cost_bps=10.0)
        text = format_summary(ev)
        assert "bare" in text
        assert "attribution" not in text  # nothing to attribute against


def _spec(panel: pd.DataFrame):
    from quant_lucky.strategies.evaluation import ResearchSpec

    base = {"vol_window": 63, "rebalance": "M"}
    return ResearchSpec(
        name="demo",
        prices=panel,
        weights=inverse_vol_weights(panel, vol_window=63),
        cost_bps=15.0,
        data_source="synthetic",
        benchmark_weights=equal_weight_weights(panel),
        strategy_fn=risk_parity_strategy_fn,
        param_grid=[{"vol_window": 42}, {"vol_window": 63}],
        wf_train_size=200,
        wf_test_size=60,
        base_param=base,
        sensitivity_sweep={"vol_window": [42, 63]},
    )
