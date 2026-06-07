"""Cross-sectional momentum (Jegadeesh & Titman 1993) — strategy body.

The "12-1" winners-minus-losers anomaly: each month, rank the universe
by trailing 12-month return *excluding the most recent month*, go long
the top quantile and (canonically) short the bottom, hold one month.
This module wires the reusable
:func:`quant_lucky.strategies.momentum.cross_sectional_momentum_weights`
primitive into a monthly-rebalanced book and declares the universe,
benchmark, Walk-Forward grid and sensitivity sweep for ``run_backtest.py``.

Data honesty
------------
A real S&P 500 cross-section is not available offline in this repo (a
documented yfinance 429 history), and the few cached US names cannot
form a 500-name ranking. :func:`load_prices` therefore falls back to a
**synthetic, edgeless** panel (``synthetic.autocorr = 0``). On that panel
momentum *should* earn ≈0 gross and lose money net of cost — that is the
honest result, and it validates the pipeline rather than a discovered
alpha. Set ``synthetic.autocorr > 0`` (CLI: ``--set synthetic.autocorr=0.12``)
to inject trailing-return persistence and watch the same machinery
capture an edge when one exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_lucky.strategies.data import load_close_panel, synthetic_price_panel
from quant_lucky.strategies.evaluation import ResearchSpec
from quant_lucky.strategies.momentum import cross_sectional_momentum_weights
from quant_lucky.strategies.risk_parity import equal_weight_weights, hold_between_rebalances

NAME = "momentum_us_monthly_top20"
CONFIG_PATH = Path(__file__).with_name("config.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "name": NAME,
    "cost_bps": 10.0,
    "periods_per_year": 252,
    "oos_fraction": 0.3,
    "signal": {
        "lookback": 252,
        "skip": 21,
        "top_quantile": 0.2,
        "long_only": False,
        "gross_leverage": 1.0,
        "rebalance": "M",
    },
    "universe": {
        "real_specs": [
            ["yfinance", "AAPL"],
            ["yfinance", "KO"],
            ["yfinance", "PEP"],
            ["yfinance", "SPY"],
        ],
        "min_real_assets": 30,
        "min_real_rows": 504,
    },
    "synthetic": {
        "n_assets": 60,
        "n_days": 1512,
        "seed": 20260601,
        "autocorr": 0.0,
        "drift_dispersion": 0.0,
        "beta_low": 1.0,
        "beta_high": 1.0,
    },
    "walk_forward": {"train_size": 504, "test_size": 126},
    "sensitivity": {
        "lookback": [126, 189, 252, 315],
        "skip": [0, 21, 42],
        "top_quantile": [0.1, 0.2, 0.3],
    },
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Return ``(panel, data_source)``: a real cross-section if one is
    cached and wide enough, else the synthetic edgeless fallback.

    The real basket is intentionally rejected unless it has at least
    ``min_real_assets`` names and ``min_real_rows`` rows — a 4-name
    "cross-section" cannot rank into quintiles, so we would rather be
    honest and synthesise.
    """
    uni = config["universe"]
    specs = list(uni["real_specs"])
    try:
        panel = load_close_panel(specs, min_overlap=uni["min_real_rows"])
        if panel.shape[1] >= uni["min_real_assets"] and len(panel) >= uni["min_real_rows"]:
            return panel, "real"
    except (FileNotFoundError, ValueError):
        pass

    syn = config["synthetic"]
    panel = synthetic_price_panel(
        n_assets=int(syn["n_assets"]),
        n_days=int(syn["n_days"]),
        seed=int(syn["seed"]),
        momentum_autocorr=float(syn["autocorr"]),
        drift_dispersion=float(syn.get("drift_dispersion", 0.10)),
        beta_low=float(syn.get("beta_low", 0.6)),
        beta_high=float(syn.get("beta_high", 1.4)),
    )
    return panel, "synthetic"


# ---------------------------------------------------------------------------
# Signal → weights
# ---------------------------------------------------------------------------
def base_param(config: dict[str, Any]) -> dict[str, Any]:
    """The base signal parameters (the centre of the sensitivity sweep)."""
    return dict(config["signal"])


def strategy_fn(param: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """No-look-ahead monthly momentum book for the given parameters.

    Daily cross-sectional momentum weights are down-sampled to a periodic
    rebalance (held flat between month-ends) so the book trades ~12×/year.
    Matches the :data:`~quant_lucky.backtest.validation.StrategyFn`
    contract used by Walk-Forward and the sensitivity sweep.
    """
    daily = cross_sectional_momentum_weights(
        prices,
        lookback=int(param.get("lookback", 252)),
        skip=int(param.get("skip", 21)),
        top_quantile=float(param.get("top_quantile", 0.2)),
        long_only=bool(param.get("long_only", False)),
        gross_leverage=float(param.get("gross_leverage", 1.0)),
    )
    return hold_between_rebalances(daily, str(param.get("rebalance", "M")))


def build_weights(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    return strategy_fn(base_param(config), prices)


def build_benchmark(config: dict[str, Any], prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight long-only 'market' proxy for attribution.

    A dollar-neutral momentum book should show ~zero beta to this — the
    attribution's job is to *demonstrate* that market-neutrality, not to
    beat the benchmark on level.
    """
    return equal_weight_weights(prices, rebalance=str(config["signal"]["rebalance"]))


# ---------------------------------------------------------------------------
# Research spec
# ---------------------------------------------------------------------------
def build_spec(config: dict[str, Any], prices: pd.DataFrame, data_source: str) -> ResearchSpec:
    base = base_param(config)
    grid = [{**base, "lookback": lb} for lb in config["sensitivity"]["lookback"]]
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
