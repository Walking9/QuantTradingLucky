"""Inverse-volatility risk parity (cross-market) — strategy body.

Risk parity (Qian 2005; Asness, Frazzini & Pedersen 2012) sizes each
asset to contribute *equal risk* rather than equal capital. For a small,
weakly-correlated basket the inverse-volatility approximation
``w_i ∝ 1/σ_i`` is within a hair of the full equal-risk-contribution
solution while avoiding an ill-conditioned covariance inversion on short
samples. This module wires
:func:`quant_lucky.strategies.risk_parity.inverse_vol_weights` onto a
**real** cross-market basket (BTC / SPY / CSI-300) and declares the
equal-weight benchmark, Walk-Forward grid and sensitivity sweep.

The headline question is risk parity's reason to exist: does letting each
asset contribute equal *risk* beat naive equal *capital*? With a wildly
volatile BTC leg in the basket, the answer should show up as a much lower
drawdown and a higher Sharpe than the equal-weight benchmark — the
inverse-vol sizing down-weights BTC to roughly its risk share.

Annualisation
-------------
The panel is inner-joined on days **all three venues trade**, which drops
crypto's weekend bars. So the effective calendar is ~252 bars/year (the
equity convention), not 365 — ``periods_per_year`` is 252 here on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_lucky.strategies.data import load_close_panel
from quant_lucky.strategies.evaluation import ResearchSpec
from quant_lucky.strategies.risk_parity import (
    equal_weight_weights,
    inverse_vol_weights,
    risk_parity_strategy_fn,
)

NAME = "risk_parity_multi_asset"
CONFIG_PATH = Path(__file__).with_name("config.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "name": NAME,
    "cost_bps": 15.0,
    "periods_per_year": 252,
    "oos_fraction": 0.3,
    "signal": {"vol_window": 63, "rebalance": "M", "gross_leverage": 1.0},
    "universe": {
        "specs": [
            ["ccxt.binance", "BTC-USDT"],
            ["yfinance", "SPY"],
            ["yfinance", "000300.SS"],
        ],
        "min_rows": 300,
    },
    "walk_forward": {"train_size": 252, "test_size": 63},
    "sensitivity": {
        "vol_window": [21, 42, 63, 126],
        "rebalance": ["W", "M"],
    },
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Load the real cross-market basket (inner join on common trading days)."""
    uni = config["universe"]
    specs = list(uni["specs"])
    panel = load_close_panel(specs, min_overlap=int(uni["min_rows"]))
    return panel, "real"


# ---------------------------------------------------------------------------
# Signal → weights
# ---------------------------------------------------------------------------
def base_param(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["signal"])


def strategy_fn(param: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """No-look-ahead inverse-vol book (the primitive's adapter)."""
    return risk_parity_strategy_fn(param, prices)


def build_weights(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    sig = config["signal"]
    return inverse_vol_weights(
        prices,
        vol_window=int(sig["vol_window"]),
        rebalance=str(sig["rebalance"]),
        gross_leverage=float(sig["gross_leverage"]),
    )


def build_benchmark(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight (equal *capital*) — risk parity's null hypothesis.

    Same rebalance cadence so the comparison is apples-to-apples; the only
    difference is capital-weight vs risk-weight.
    """
    return equal_weight_weights(
        prices,
        rebalance=str(config["signal"]["rebalance"]),
        gross_leverage=float(config["signal"]["gross_leverage"]),
    )


# ---------------------------------------------------------------------------
# Research spec
# ---------------------------------------------------------------------------
def build_spec(config: dict[str, Any], prices: pd.DataFrame, data_source: str) -> ResearchSpec:
    base = base_param(config)
    grid = [{**base, "vol_window": vw} for vw in config["sensitivity"]["vol_window"]]
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
