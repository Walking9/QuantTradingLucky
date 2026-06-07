"""Regression tests for the runnable ``strategies/<name>/`` packages.

The packages live outside the importable ``quant_lucky`` tree (they are
runnable script dirs, not library code), so each ``strategy.py`` is loaded
here by file path under a *unique* module name — all three are literally
named ``strategy`` and would otherwise collide in ``sys.modules``.

Two tiers:

* **Contract / wiring tests** (always run): exercise each package's
  ``strategy.py`` on a *synthetic* panel so they need no cached data and
  run in CI. They check the uniform interface, weight shape/no-look-ahead,
  the YAML config round-trip, and a full :func:`run_research` pass.
* **Real-data smoke tests** (``@pytest.mark.integration``, skipped in CI):
  the two real-data packages run ``run_backtest.main`` against the cached
  parquet, proving the end-to-end entry point works where data exists.

A single offline subprocess smoke test proves
``python strategies/momentum_us_monthly_top20/run_backtest.py`` actually
executes as a script (the "clone & run" deliverable guarantee).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from quant_lucky.strategies.cli import load_config
from quant_lucky.strategies.data import synthetic_price_panel
from quant_lucky.strategies.evaluation import run_research

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "strategies"

# (package dir, synthetic panel kwargs sized to yield Walk-Forward splits)
PACKAGES = [
    ("momentum_us_monthly_top20", {"n_assets": 20, "n_days": 800, "seed": 1}),
    ("dual_ma_a_daily_vol_filter", {"n_assets": 8, "n_days": 600, "seed": 2}),
    ("risk_parity_multi_asset", {"n_assets": 3, "n_days": 600, "seed": 3}),
]
PACKAGE_IDS = [name for name, _ in PACKAGES]


def _load_strategy(pkg_name: str) -> ModuleType:
    """Load ``strategies/<pkg>/strategy.py`` under a collision-free name."""
    path = PKG_ROOT / pkg_name / "strategy.py"
    spec = importlib.util.spec_from_file_location(f"_strat_{pkg_name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panel(kwargs: dict) -> pd.DataFrame:
    return synthetic_price_panel(**kwargs)


@pytest.mark.parametrize(("pkg_name", "panel_kw"), PACKAGES, ids=PACKAGE_IDS)
class TestPackageContract:
    def test_uniform_interface(self, pkg_name: str, panel_kw: dict) -> None:
        s = _load_strategy(pkg_name)
        for attr in (
            "NAME",
            "CONFIG_PATH",
            "DEFAULT_CONFIG",
            "load_prices",
            "build_weights",
            "build_benchmark",
            "build_spec",
            "strategy_fn",
        ):
            assert hasattr(s, attr), f"{pkg_name}.strategy missing {attr}"
        assert pkg_name == s.NAME
        assert s.DEFAULT_CONFIG["name"] == pkg_name

    def test_config_yaml_round_trips(self, pkg_name: str, panel_kw: dict) -> None:
        s = _load_strategy(pkg_name)
        cfg = load_config(s.DEFAULT_CONFIG, config_path=s.CONFIG_PATH)
        # The shipped YAML must agree with the in-code default name and carry
        # the blocks run_backtest depends on.
        assert cfg["name"] == pkg_name
        for block in ("signal", "universe", "walk_forward", "sensitivity"):
            assert block in cfg

    def test_build_weights_shape(self, pkg_name: str, panel_kw: dict) -> None:
        s = _load_strategy(pkg_name)
        panel = _panel(panel_kw)
        w = s.build_weights(s.DEFAULT_CONFIG, panel)
        assert w.shape == panel.shape
        assert list(w.columns) == list(panel.columns)
        assert w.index.equals(panel.index)

    def test_no_look_ahead(self, pkg_name: str, panel_kw: dict) -> None:
        s = _load_strategy(pkg_name)
        panel = _panel(panel_kw)
        cut = len(panel) // 2
        w_full = s.build_weights(s.DEFAULT_CONFIG, panel)
        perturbed = panel.copy()
        perturbed.iloc[cut + 1 :] *= 1.5  # mangle the future
        w_pert = s.build_weights(s.DEFAULT_CONFIG, perturbed)
        pd.testing.assert_frame_equal(w_full.iloc[: cut + 1], w_pert.iloc[: cut + 1])

    def test_run_research_populates_bundle(self, pkg_name: str, panel_kw: dict) -> None:
        s = _load_strategy(pkg_name)
        panel = _panel(panel_kw)
        spec = s.build_spec(s.DEFAULT_CONFIG, panel, "synthetic")
        ev = run_research(spec)
        assert ev.name == pkg_name
        assert ev.data_source == "synthetic"
        assert ev.benchmark is not None  # all three define a benchmark
        assert ev.attribution is not None
        assert ev.walk_forward is not None  # panels sized to yield splits
        assert ev.sensitivity is not None and not ev.sensitivity.empty
        # metrics_dict must be JSON-clean (no numpy scalars leaking).
        import json

        json.dumps(ev.metrics_dict())


# ---------------------------------------------------------------------------
# Offline end-to-end: the momentum script must run as a script
# ---------------------------------------------------------------------------
def test_momentum_run_backtest_executes_offline() -> None:
    """`python run_backtest.py` runs with the synthetic fallback (no data)."""
    script = PKG_ROOT / "momentum_us_monthly_top20" / "run_backtest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--no-write",
            "--set",
            "synthetic.n_days=500",
            "--set",
            "synthetic.n_assets=20",
            "--set",
            "walk_forward.train_size=300",
            "--set",
            "walk_forward.test_size=80",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "momentum_us_monthly_top20" in result.stdout
    assert "data: synthetic" in result.stdout


# ---------------------------------------------------------------------------
# Real-data smoke tests (skipped in CI: parquet cache is gitignored)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize("pkg_name", ["dual_ma_a_daily_vol_filter", "risk_parity_multi_asset"])
def test_real_data_run_backtest(pkg_name: str) -> None:
    """End-to-end run on the cached parquet (requires local data)."""
    sys.path.insert(0, str(PKG_ROOT / pkg_name))
    try:
        s = _load_strategy(pkg_name)
        cfg = load_config(s.DEFAULT_CONFIG, config_path=s.CONFIG_PATH)
        prices, source = s.load_prices(cfg)
        assert source == "real"
        ev = run_research(s.build_spec(cfg, prices, source))
        assert ev.full.report.n_periods > 0
    finally:
        sys.path.remove(str(PKG_ROOT / pkg_name))
