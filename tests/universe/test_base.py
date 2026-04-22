"""Tests for :mod:`quant_lucky.universe.base`."""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest

from quant_lucky.universe import UniverseSnapshot, coerce_date


class TestUniverseSnapshot:
    def test_members_sorted_and_deduped(self) -> None:
        snap = UniverseSnapshot(
            as_of=date(2024, 1, 1),
            members=("MSFT", "AAPL", "MSFT", "NVDA"),
        )
        assert snap.members == ("AAPL", "MSFT", "NVDA")

    def test_len_and_iter(self) -> None:
        snap = UniverseSnapshot(as_of=date(2024, 1, 1), members=("A", "B"))
        assert len(snap) == 2
        assert list(snap) == ["A", "B"]

    def test_contains(self) -> None:
        snap = UniverseSnapshot(as_of=date(2024, 1, 1), members=("A", "B"))
        assert "A" in snap
        assert "Z" not in snap

    def test_member_set_is_defensive(self) -> None:
        snap = UniverseSnapshot(as_of=date(2024, 1, 1), members=("A",))
        members = snap.member_set()
        members.add("B")
        # Original snapshot is unaffected
        assert "B" not in snap

    def test_requires_date_type(self) -> None:
        with pytest.raises(TypeError, match="as_of"):
            UniverseSnapshot(as_of="2024-01-01", members=("A",))  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        snap = UniverseSnapshot(as_of=date(2024, 1, 1), members=("A",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.members = ("Z",)  # type: ignore[misc]


class TestCoerceDate:
    def test_none_returns_today(self) -> None:
        result = coerce_date(None)
        assert isinstance(result, date)

    def test_datetime_stripped_to_date(self) -> None:
        result = coerce_date(datetime(2024, 6, 15, 12, 30, 45))
        assert result == date(2024, 6, 15)

    def test_date_passed_through(self) -> None:
        d = date(2024, 6, 15)
        assert coerce_date(d) == d
