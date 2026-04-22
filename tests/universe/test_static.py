"""Tests for :class:`StaticUniverse`."""

from __future__ import annotations

from datetime import UTC, date, datetime

from quant_lucky.universe import StaticUniverse


class TestStaticUniverse:
    def test_simple_membership(self) -> None:
        u = StaticUniverse(members_list=("AAPL", "MSFT"))
        assert u.members() == {"AAPL", "MSFT"}

    def test_snapshot_metadata(self) -> None:
        u = StaticUniverse(members_list=("AAPL",), label="my_list")
        snap = u.snapshot(date(2024, 1, 1))
        assert snap.metadata["builder"] == "my_list"
        assert snap.metadata["source"] == "default"

    def test_snapshot_uses_today_when_none(self) -> None:
        u = StaticUniverse(members_list=("A",))
        snap = u.snapshot()
        assert snap.as_of == datetime.now(tz=UTC).date()

    def test_history_resolves_correctly(self) -> None:
        history = {
            date(2020, 1, 1): ("AAPL", "MSFT"),
            date(2022, 1, 1): ("AAPL", "MSFT", "NVDA"),
            date(2024, 1, 1): ("AAPL", "NVDA"),
        }
        u = StaticUniverse(members_list=("FALLBACK",), history=history)

        # Before any history -> default
        assert u.members(date(2019, 6, 1)) == {"FALLBACK"}
        # Exactly at effective date -> that snapshot
        assert u.members(date(2020, 1, 1)) == {"AAPL", "MSFT"}
        # Between two entries -> earlier snapshot
        assert u.members(date(2021, 6, 1)) == {"AAPL", "MSFT"}
        # Latest entry
        assert u.members(date(2024, 6, 1)) == {"AAPL", "NVDA"}

    def test_history_source_label(self) -> None:
        history = {date(2022, 1, 1): ("AAPL",)}
        u = StaticUniverse(members_list=("ZZZ",), history=history)
        snap = u.snapshot(date(2023, 1, 1))
        assert snap.metadata["source"].startswith("history@2022-01-01")

    def test_accepts_datetime_as_of(self) -> None:
        u = StaticUniverse(members_list=("A", "B"))
        snap = u.snapshot(datetime(2024, 6, 1, 15, 0))
        assert snap.as_of == date(2024, 6, 1)
