"""Smoke tests for the event-driven engine placeholder.

The real engine ships in M8; until then we only verify the stub does
what its docstring promises (raises NotImplementedError on use, but
can still be imported and subclassed for type hints).
"""

from __future__ import annotations

import pytest

from quant_lucky.backtest.event import Event, EventEngine


def test_event_engine_is_abstract() -> None:
    """Direct instantiation must fail because ``run`` is abstract."""
    with pytest.raises(TypeError):
        EventEngine()  # type: ignore[abstract]


def test_subclass_run_still_raises_until_implemented() -> None:
    """Default implementation of run() raises NotImplementedError."""

    class _Stub(EventEngine):
        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return super().run(*args, **kwargs)

    with pytest.raises(NotImplementedError, match="M8"):
        _Stub().run()


def test_event_dataclass_is_constructible() -> None:
    """Event is a placeholder dataclass; just ensure it accepts payload."""
    e = Event(timestamp=None, payload={"kind": "bar", "asset": "A0"})
    assert e.payload["asset"] == "A0"
