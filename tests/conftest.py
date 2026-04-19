"""Shared pytest fixtures and configuration.

Keep this file small; topic-specific fixtures belong next to the tests
that use them (e.g. ``tests/factors/conftest.py``).
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    """Seed all random sources so tests are reproducible."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    yield


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Directory holding committed test fixtures (small CSV/Parquet)."""
    path = project_root / "tests" / "fixtures"
    path.mkdir(parents=True, exist_ok=True)
    return path
