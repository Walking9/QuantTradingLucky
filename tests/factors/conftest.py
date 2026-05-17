"""Shared fixtures for factors tests.

Synthetic panels are deterministic given the global ``_deterministic_seed``
fixture in the root conftest; we do not re-seed locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small (date, asset) OHLCV panel and matching wide-price DataFrame.

    Returns:
        (panel_long, prices_wide). The wide DataFrame is the same close
        series as ``panel_long['close'].unstack('asset')`` — provided
        separately because most tester functions take a wide price frame.
    """
    rng = np.random.default_rng(42)
    n_days = 120
    n_assets = 6
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]

    # Random walk per asset for prices. Use lognormal returns so prices stay > 0.
    log_ret = rng.normal(0.0005, 0.012, size=(n_days, n_assets))
    prices = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
        index=dates,
        columns=assets,
    )
    prices.index.name = "date"
    prices.columns.name = "asset"

    long_close = prices.stack(future_stack=True).rename("close")
    long_idx = long_close.index

    df = pd.DataFrame(
        {
            "open": long_close.values * (1 + rng.normal(0, 0.001, len(long_idx))),
            "high": long_close.values * (1 + rng.uniform(0, 0.01, len(long_idx))),
            "low": long_close.values * (1 - rng.uniform(0, 0.01, len(long_idx))),
            "close": long_close.values,
            "volume": rng.lognormal(15, 0.5, len(long_idx)),
            "amount": rng.lognormal(20, 0.5, len(long_idx)),
        },
        index=long_idx,
    )
    return df, prices


@pytest.fixture()
def panel_with_signal() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Panel where asset_0..asset_2 have positive drift, asset_3..5 negative.

    Used to verify the tester actually detects a planted Alpha — IC must
    be statistically meaningful, not just numerically valid.
    """
    rng = np.random.default_rng(7)
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"S{i}" for i in range(6)]

    drifts = np.array([0.002, 0.0015, 0.001, -0.001, -0.0015, -0.002])
    rets = rng.normal(0, 0.01, size=(n_days, 6)) + drifts
    prices = pd.DataFrame(
        np.exp(np.cumsum(rets, axis=0)) * 100.0,
        index=dates,
        columns=assets,
    )
    prices.index.name = "date"
    prices.columns.name = "asset"

    long_close = prices.stack(future_stack=True).rename("close")
    idx = long_close.index

    panel = pd.DataFrame(
        {
            "open": long_close.values,
            "high": long_close.values * 1.01,
            "low": long_close.values * 0.99,
            "close": long_close.values,
            "volume": rng.lognormal(15, 0.5, len(idx)),
            "amount": rng.lognormal(20, 0.5, len(idx)),
        },
        index=idx,
    )

    # Oracle factor: the true drift (constant per asset). Tester should
    # find a strongly positive IC against forward returns.
    oracle = pd.Series(np.tile(drifts, n_days), index=idx, name="oracle")
    return panel, prices, oracle
