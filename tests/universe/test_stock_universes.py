"""Tests for CSI300 and SP500 universe builders.

Uses fake fetchers to avoid any network dependency.
"""

from __future__ import annotations

from datetime import date

import pytest

from quant_lucky.universe import CSI300Universe, SP500Universe
from quant_lucky.universe.base import UniverseDataUnavailableError


class TestCSI300Universe:
    def test_seed_snapshot_without_fetcher(self) -> None:
        u = CSI300Universe()
        snap = u.snapshot(date(2024, 6, 30))
        assert snap.members  # non-empty
        assert snap.metadata["source"] == "seed"

    def test_fetcher_used_when_provided(self) -> None:
        def fake(_: date) -> list[str]:
            return ["600519.SH", "000858.SZ"]

        u = CSI300Universe(fetcher=fake)
        snap = u.snapshot(date(2024, 6, 30))
        assert set(snap) == {"600519.SH", "000858.SZ"}
        assert snap.metadata["source"] == "fetcher"

    def test_fetcher_empty_falls_back_to_seed(self) -> None:
        u = CSI300Universe(fetcher=lambda _: [])
        snap = u.snapshot(date(2024, 1, 1))
        assert set(snap.members) == set(u.seed)
        assert "empty" in snap.metadata["reason"]

    def test_fetcher_empty_raises_when_strict(self) -> None:
        u = CSI300Universe(fetcher=lambda _: [], use_seed_on_error=False)
        with pytest.raises(UniverseDataUnavailableError):
            u.snapshot(date(2024, 1, 1))

    def test_members_convenience(self) -> None:
        u = CSI300Universe(fetcher=lambda _: ["A", "B"])
        assert u.members(date(2024, 1, 1)) == {"A", "B"}


class TestSP500Universe:
    def test_seed_snapshot_without_fetcher(self) -> None:
        u = SP500Universe()
        snap = u.snapshot(date(2024, 6, 30))
        assert "AAPL" in snap
        assert snap.metadata["source"] == "seed"

    def test_fetcher_used_when_provided(self) -> None:
        u = SP500Universe(fetcher=lambda _: ["AAPL", "MSFT", "NVDA"])
        snap = u.snapshot(date(2024, 6, 30))
        assert set(snap) == {"AAPL", "MSFT", "NVDA"}

    def test_fetcher_empty_strict_raises(self) -> None:
        u = SP500Universe(fetcher=lambda _: (), use_seed_on_error=False)
        with pytest.raises(UniverseDataUnavailableError):
            u.snapshot(date(2024, 1, 1))
