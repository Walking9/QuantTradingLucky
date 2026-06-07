"""Tests for strategy price-panel assembly and synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import ParquetStore
from quant_lucky.strategies.data import (
    load_close_panel,
    synthetic_price_panel,
)


def _make_ohlcv(dates: pd.DatetimeIndex, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Build a schema-valid OHLCV frame with tz-aware UTC timestamps."""
    rng = np.random.default_rng(seed)
    close = start_price * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
    open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(dates))))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, len(dates)),
        }
    )


class TestSyntheticPanel:
    def test_deterministic(self) -> None:
        a = synthetic_price_panel(n_assets=6, n_days=300, seed=42)
        b = synthetic_price_panel(n_assets=6, n_days=300, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_seed_changes_output(self) -> None:
        a = synthetic_price_panel(n_assets=6, n_days=300, seed=1)
        b = synthetic_price_panel(n_assets=6, n_days=300, seed=2)
        assert not np.allclose(a.to_numpy(), b.to_numpy())

    def test_shape_and_labels(self) -> None:
        p = synthetic_price_panel(n_assets=5, n_days=120)
        assert p.shape == (120, 5)
        assert list(p.columns) == ["SYN00", "SYN01", "SYN02", "SYN03", "SYN04"]
        assert (p > 0).all().all()
        assert p.index.name == "date"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n_assets"):
            synthetic_price_panel(n_assets=0)
        with pytest.raises(ValueError, match="n_days"):
            synthetic_price_panel(n_days=1)

    def test_no_planted_edge_by_default(self) -> None:
        # With momentum_autocorr=0, a momentum long-short should NOT earn a
        # large positive gross Sharpe — the panel is (near) edgeless.
        from quant_lucky.backtest import VectorEngine
        from quant_lucky.strategies.momentum import cross_sectional_momentum_weights

        px = synthetic_price_panel(n_assets=20, n_days=900, seed=99)
        w = cross_sectional_momentum_weights(px, lookback=120, skip=21, top_quantile=0.2)
        gross_sharpe = VectorEngine(cost_bps=0.0).run(w, px).report.sharpe
        assert abs(gross_sharpe) < 1.5  # consistent with noise, not a planted alpha

    def test_injected_autocorr_is_detectable(self) -> None:
        from quant_lucky.backtest import VectorEngine
        from quant_lucky.strategies.momentum import cross_sectional_momentum_weights

        edged = synthetic_price_panel(n_assets=20, n_days=900, seed=99, momentum_autocorr=0.15)
        w = cross_sectional_momentum_weights(edged, lookback=120, skip=21, top_quantile=0.2)
        none = synthetic_price_panel(n_assets=20, n_days=900, seed=99, momentum_autocorr=0.0)
        w0 = cross_sectional_momentum_weights(none, lookback=120, skip=21, top_quantile=0.2)
        s_edge = VectorEngine(cost_bps=0.0).run(w, edged).report.sharpe
        s_none = VectorEngine(cost_bps=0.0).run(w0, none).report.sharpe
        # Injecting persistence raises gross momentum Sharpe above the
        # edgeless baseline (sanity check the knob does what it claims).
        assert s_edge > s_none


class TestLoadClosePanel:
    def test_inner_join_aligns_on_common_dates(self, tmp_path) -> None:
        store = ParquetStore(root=tmp_path)
        dates_a = pd.date_range("2022-01-03", periods=20, freq="B", tz="UTC")
        # B overlaps A but starts 5 days later and ends 5 days earlier.
        dates_b = dates_a[5:15]
        store.write(
            _make_ohlcv(dates_a, seed=1), provider="p1", symbol="AAA", frequency=Frequency.DAILY
        )
        store.write(
            _make_ohlcv(dates_b, seed=2), provider="p2", symbol="BBB", frequency=Frequency.DAILY
        )

        panel = load_close_panel([("p1", "AAA"), ("p2", "BBB")], store_root=tmp_path)
        assert list(panel.columns) == ["AAA", "BBB"]
        assert len(panel) == len(dates_b)  # inner join → overlap window
        assert not panel.isna().any().any()
        assert panel.index.is_monotonic_increasing

    def test_outer_join_ffills(self, tmp_path) -> None:
        store = ParquetStore(root=tmp_path)
        dates_a = pd.date_range("2022-01-03", periods=20, freq="B", tz="UTC")
        dates_b = dates_a[5:15]
        store.write(
            _make_ohlcv(dates_a, seed=1), provider="p1", symbol="AAA", frequency=Frequency.DAILY
        )
        store.write(
            _make_ohlcv(dates_b, seed=2), provider="p2", symbol="BBB", frequency=Frequency.DAILY
        )

        panel = load_close_panel([("p1", "AAA"), ("p2", "BBB")], store_root=tmp_path, how="outer")
        assert len(panel) == len(dates_a)  # union of dates
        # BBB ffilled after its last bar → no NaN past the first valid point.
        assert panel["BBB"].iloc[-1] == panel["BBB"].dropna().iloc[-1]

    def test_date_bounds(self, tmp_path) -> None:
        store = ParquetStore(root=tmp_path)
        dates = pd.date_range("2022-01-03", periods=40, freq="B", tz="UTC")
        store.write(
            _make_ohlcv(dates, seed=1), provider="p1", symbol="AAA", frequency=Frequency.DAILY
        )
        panel = load_close_panel(
            [("p1", "AAA")], store_root=tmp_path, start="2022-01-10", end="2022-01-20"
        )
        assert panel.index.min() >= pd.Timestamp("2022-01-10")
        assert panel.index.max() <= pd.Timestamp("2022-01-20")

    def test_too_little_overlap_raises(self, tmp_path) -> None:
        store = ParquetStore(root=tmp_path)
        a = pd.date_range("2022-01-03", periods=10, freq="B", tz="UTC")
        b = pd.date_range("2022-06-01", periods=10, freq="B", tz="UTC")  # disjoint
        store.write(_make_ohlcv(a, seed=1), provider="p1", symbol="AAA", frequency=Frequency.DAILY)
        store.write(_make_ohlcv(b, seed=2), provider="p2", symbol="BBB", frequency=Frequency.DAILY)
        with pytest.raises(ValueError, match="aligned rows"):
            load_close_panel([("p1", "AAA"), ("p2", "BBB")], store_root=tmp_path)

    def test_bad_how(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="how"):
            load_close_panel([("p", "X")], store_root=tmp_path, how="cross")

    def test_empty_specs(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="specs is empty"):
            load_close_panel([], store_root=tmp_path)

    def test_missing_symbol_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_close_panel([("nope", "ZZZ")], store_root=tmp_path)
