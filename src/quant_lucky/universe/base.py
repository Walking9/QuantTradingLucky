"""Core abstractions for universe construction.

A *universe* is the set of tradeable instruments a strategy can hold at
a given point in time. The primary defence against survivorship bias is
**point-in-time correctness**: a symbol that joined the index in 2018
must not appear in a 2015 snapshot, and a delisted symbol must still
appear on dates before its removal.

This module defines the data classes and the ``UniverseBuilder`` ABC
every concrete builder inherits from. Concrete builders live in
neighbouring modules (``static.py``, ``csi300.py``, ``sp500.py``,
``crypto_topn.py``).

The interface is intentionally minimal:

- :meth:`UniverseBuilder.snapshot` returns the members on a given date.
- :meth:`UniverseBuilder.members` is a convenience alias returning just
  the set — useful when metadata is not needed.

Subclasses decide how to source membership data. A caller requiring
point-in-time correctness should always pick a builder that documents
it (most stock builders here use a seed snapshot; wire a live data
source in production).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any


class UniverseError(Exception):
    """Base class for universe-construction errors."""


class UniverseDataUnavailableError(UniverseError):
    """Raised when the builder cannot source membership data.

    This is a distinct class because callers often want to fall back to
    a cached or seed universe rather than propagate the error.
    """


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable universe membership at a point in time.

    Attributes
    ----------
    as_of:
        The date the snapshot is valid for.
    members:
        Sorted tuple of instrument identifiers. A tuple (not a set) so
        equality comparisons and hashing are deterministic.
    metadata:
        Free-form metadata the builder may attach (e.g. index name,
        data source, freshness timestamp). Never contains secrets.
    """

    as_of: date
    members: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, date):
            raise TypeError(f"as_of must be a date, got {type(self.as_of).__name__}")
        # Stored sorted + deduplicated so callers can rely on the layout.
        object.__setattr__(self, "members", tuple(sorted(set(self.members))))

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.members

    def member_set(self) -> set[str]:
        """Return members as a mutable ``set`` (defensive copy)."""
        return set(self.members)


class UniverseBuilder(ABC):
    """Abstract base for universe construction.

    Implementations should be cheap to construct; heavy I/O belongs in
    :meth:`snapshot` so a single instance can be reused across dates.
    """

    name: str = "base"

    @abstractmethod
    def snapshot(self, as_of: date | datetime | None = None) -> UniverseSnapshot:
        """Return the universe membership as of ``as_of``.

        Passing ``None`` means "latest available". Implementations that
        do not support point-in-time queries must raise
        :class:`UniverseError` when given a non-``None`` historical
        date.
        """
        raise NotImplementedError

    def members(self, as_of: date | datetime | None = None) -> set[str]:
        """Convenience wrapper returning only the member set."""
        return self.snapshot(as_of).member_set()


def coerce_date(value: date | datetime | None) -> date:
    """Coerce ``None``/``datetime``/``date`` to ``date``.

    ``None`` is mapped to today (UTC) so default behaviour is
    deterministic. Using the caller's local timezone would make tests
    flaky; explicit UTC avoids that trap.
    """
    if value is None:
        return datetime.now(tz=UTC).date()
    if isinstance(value, datetime):
        return value.date()
    return value
