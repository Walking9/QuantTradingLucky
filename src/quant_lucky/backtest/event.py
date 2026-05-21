"""Event-driven backtest engine (placeholder).

Implementation deferred to M8 / M11 when CTA strategies and intraday
order book simulation become necessary. The contract is sketched here
so the rest of the codebase can import from ``quant_lucky.backtest``
without needing to know whether vectorised or event-driven semantics
are in play.

What event-driven adds over vectorised
--------------------------------------
* Limit orders, stop orders, trailing stops.
* Partial fills, rejections, exchange latency.
* Per-bar (or per-tick) order book interaction — required for any
  market-making, ladder, or HFT-style strategy.
* Live-trading code path: the same loop runs against a paper or live
  broker by swapping the data feed and execution handler.

Why we defer
------------
The vectorised engine answers 95% of factor / portfolio-level research
questions and runs in milliseconds. Building the event engine before
M8 would be premature: we'd guess at order-type requirements and end up
rewriting once the first CTA strategy lands. Better to build it then.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    """Marker base class for events. Subclasses live in the M8 iteration."""

    timestamp: Any  # pd.Timestamp once we wire it in
    payload: dict[str, Any]


class EventEngine(ABC):
    """Abstract base for event-driven engines.

    Concrete implementations (e.g. ``BarEventEngine`` for OHLC bar feeds,
    ``TickEventEngine`` for L1 quotes) will be added when we need them.
    Calling :meth:`run` on this stub raises ``NotImplementedError`` so
    callers cannot accidentally use it before it exists.
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """To be implemented in M8.

        Raises:
            NotImplementedError: always — this is a placeholder.
        """
        raise NotImplementedError(
            "Event-driven engine is not yet implemented. Use VectorEngine for "
            "factor / portfolio research; the event engine lands with the CTA "
            "strategies in M8."
        )


__all__ = [
    "Event",
    "EventEngine",
]
