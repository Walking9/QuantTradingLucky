"""Static universe — membership provided directly as a list.

Useful for:

- Unit tests that need a deterministic, network-free universe.
- Custom strategy-specific baskets (e.g. "my top-10 tickers").
- Providing a seed universe that more sophisticated builders fall
  back to when their data source is unreachable.

The static universe supports optional membership *history* so it can
mimic point-in-time correctness when the caller supplies one:

    >>> hist = {
    ...     date(2020, 1, 1): ["AAPL", "MSFT"],
    ...     date(2022, 1, 1): ["AAPL", "MSFT", "NVDA"],
    ... }
    >>> universe = StaticUniverse(members=["AAPL"], history=hist)
    >>> universe.members(as_of=date(2021, 6, 1)) == {"AAPL", "MSFT"}
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import ClassVar

from quant_lucky.universe.base import UniverseBuilder, UniverseSnapshot, coerce_date


@dataclass
class StaticUniverse(UniverseBuilder):
    """Return a hard-coded membership list.

    When ``history`` is provided the builder resolves the **largest**
    effective date ``<= as_of`` and uses that snapshot. If ``as_of``
    precedes every entry in the history, :attr:`members` is returned
    (so the constructor default acts as a floor).
    """

    members_list: tuple[str, ...] = field(default_factory=tuple)
    history: dict[date, tuple[str, ...]] = field(default_factory=dict)
    label: str = "static"
    name: ClassVar[str] = "static"

    def __post_init__(self) -> None:
        self.members_list = tuple(sorted(set(self.members_list)))
        normalised = {
            effective_date: tuple(sorted(set(syms)))
            for effective_date, syms in self.history.items()
        }
        self.history = dict(sorted(normalised.items()))

    def snapshot(self, as_of: date | datetime | None = None) -> UniverseSnapshot:
        target = coerce_date(as_of)

        applicable = [d for d in self.history if d <= target]
        if applicable:
            effective = max(applicable)
            symbols = self.history[effective]
            source = f"history@{effective.isoformat()}"
        else:
            symbols = self.members_list
            source = "default"

        return UniverseSnapshot(
            as_of=target,
            members=symbols,
            metadata={"builder": self.label, "source": source},
        )
