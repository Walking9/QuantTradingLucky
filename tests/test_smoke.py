"""Smoke tests: verify the package imports and the skeleton is intact.

These run in CI on every commit; they should finish in milliseconds
and never touch the network.
"""

from __future__ import annotations

import importlib

import pytest


SUBPACKAGES: tuple[str, ...] = (
    "data",
    "universe",
    "costs",
    "indicators",
    "factors",
    "alpha",
    "portfolio",
    "backtest",
    "risk",
    "derivatives",
    "crypto",
    "ml",
    "utils",
)


def test_version_is_defined() -> None:
    import quant_lucky

    assert isinstance(quant_lucky.__version__, str)
    assert quant_lucky.__version__.count(".") == 2


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    """Every planned subpackage must be importable as a namespace."""
    module = importlib.import_module(f"quant_lucky.{name}")
    assert module.__doc__, f"quant_lucky.{name} is missing a module docstring"
