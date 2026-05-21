"""Shared fixtures for backtest tests.

We build small wide-form ``(date × asset)`` weight / price grids here.
Synthetic panels are deterministic via the root ``_deterministic_seed``
fixture; we still pass explicit ``np.random.default_rng`` seeds where a
specific signal-to-noise ratio matters for the assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """A wide (date × asset) price grid: 3 assets × 60 business days.

    Lognormal random walk so prices stay positive. Slight positive drift
    so a long-only portfolio earns something to measure.
    """
    rng = np.random.default_rng(42)
    n_days = 60
    n_assets = 3
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]

    log_ret = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    px = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
        index=dates,
        columns=assets,
    )
    px.index.name = "date"
    px.columns.name = "asset"
    return px


@pytest.fixture()
def equal_weight(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Constant 1/3, 1/3, 1/3 weights for the 3-asset price fixture."""
    n_assets = synthetic_prices.shape[1]
    return pd.DataFrame(
        1.0 / n_assets,
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )


@pytest.fixture()
def zero_weights(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """All-zero weights — no positions, no returns, no costs."""
    return pd.DataFrame(
        0.0,
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )


@pytest.fixture()
def long_short_weights(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """A market-neutral book: +1 on A0, -1 on A1, 0 on A2."""
    w = pd.DataFrame(
        0.0,
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    w["A0"] = 1.0
    w["A1"] = -1.0
    return w
