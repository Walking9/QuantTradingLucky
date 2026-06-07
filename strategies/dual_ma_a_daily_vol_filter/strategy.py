"""Dual moving-average crossover + volatility filter (A-shares) — body.

A textbook trend system (Kaufman): hold a name while its fast SMA sits
above its slow SMA, *and* its realised volatility is below a ceiling;
step to cash otherwise. Long-only, because A-share retail cannot easily
short. This module wires
:func:`quant_lucky.strategies.dual_ma.dual_ma_vol_filter_weights` onto a
small basket of **real** cached A-share names and declares the benchmark,
Walk-Forward grid and sensitivity sweep for ``run_backtest.py``.

Unlike the momentum package this one runs on genuine market data
(akshare daily, ~2022–2023). It is a *small* basket (8 names, ~2 years),
so it demonstrates the strategy and the evaluation framework honestly but
does not support a population-level claim about A-share trend following —
that needs the full CSI-300 history (an open M6 data task).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_lucky.strategies.data import load_close_panel
from quant_lucky.strategies.dual_ma import dual_ma_strategy_fn, dual_ma_vol_filter_weights
from quant_lucky.strategies.evaluation import ResearchSpec
from quant_lucky.strategies.risk_parity import equal_weight_weights

NAME = "dual_ma_a_daily_vol_filter"
CONFIG_PATH = Path(__file__).with_name("config.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "name": NAME,
    "cost_bps": 30.0,
    "periods_per_year": 252,
    "oos_fraction": 0.3,
    "signal": {
        "fast": 20,
        "slow": 60,
        "vol_window": 20,
        "vol_max": 0.40,
        "long_only": True,
        "gross_leverage": 1.0,
        "normalise": False,
    },
    "universe": {
        "specs": [
            ["akshare", "600519.SH"],
            ["akshare", "600036.SH"],
            ["akshare", "601318.SH"],
            ["akshare", "600276.SH"],
            ["akshare", "000333.SZ"],
            ["akshare", "600887.SH"],
            ["akshare", "000001.SZ"],
            ["akshare", "600000.SH"],
        ],
        "min_rows": 200,
    },
    "walk_forward": {"train_size": 252, "test_size": 63},
    "sensitivity": {
        "fast": [10, 20, 30],
        "slow": [40, 60, 90],
        "vol_max": [0.30, 0.40, 0.50, None],
    },
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Load the real A-share basket (inner-joined on common trading days)."""
    uni = config["universe"]
    specs = list(uni["specs"])
    panel = load_close_panel(specs, min_overlap=int(uni["min_rows"]))
    return panel, "real"


# ---------------------------------------------------------------------------
# Signal → weights
# ---------------------------------------------------------------------------
def base_param(config: dict[str, Any]) -> dict[str, Any]:
    base = dict(config["signal"])
    base["periods_per_year"] = int(config["periods_per_year"])
    return base


def strategy_fn(param: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """No-look-ahead dual-MA + vol-filter book (the primitive's adapter)."""
    return dual_ma_strategy_fn(param, prices)


def build_weights(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    sig = config["signal"]
    return dual_ma_vol_filter_weights(
        prices,
        fast=int(sig["fast"]),
        slow=int(sig["slow"]),
        vol_window=int(sig["vol_window"]),
        vol_max=None if sig["vol_max"] is None else float(sig["vol_max"]),
        long_only=bool(sig["long_only"]),
        gross_leverage=float(sig["gross_leverage"]),
        normalise=bool(sig["normalise"]),
        periods_per_year=int(config["periods_per_year"]),
    )


def build_benchmark(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight, monthly-rebalanced buy & hold of the basket.

    Attribution against this answers "how much of the return is just being
    long these names?" — the trend filter's value is the *alpha* (timing)
    plus the drawdown it avoids by sitting in cash.
    """
    return equal_weight_weights(prices, rebalance="M")


# ---------------------------------------------------------------------------
# Research spec
# ---------------------------------------------------------------------------
def build_spec(config: dict[str, Any], prices: pd.DataFrame, data_source: str) -> ResearchSpec:
    base = base_param(config)
    grid = [{**base, "slow": slow} for slow in config["sensitivity"]["slow"]]
    return ResearchSpec(
        name=config["name"],
        prices=prices,
        weights=build_weights(config, prices),
        cost_bps=float(config["cost_bps"]),
        periods_per_year=int(config["periods_per_year"]),
        oos_fraction=float(config["oos_fraction"]),
        data_source=data_source,
        benchmark_weights=build_benchmark(config, prices),
        strategy_fn=strategy_fn,
        param_grid=grid,
        wf_train_size=int(config["walk_forward"]["train_size"]),
        wf_test_size=int(config["walk_forward"]["test_size"]),
        base_param=base,
        sensitivity_sweep=config["sensitivity"],
    )
