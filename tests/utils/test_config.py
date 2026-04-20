"""Tests for Settings defaults and project-root anchoring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from quant_lucky.utils.config import PROJECT_ROOT, Settings, _find_project_root


def test_find_project_root_from_package_file() -> None:
    """The helper locates the repo that contains pyproject.toml."""
    root = _find_project_root()
    assert (root / "pyproject.toml").is_file()
    assert root == PROJECT_ROOT


def test_data_root_is_absolute_and_under_project_root() -> None:
    """Default data_root must be CWD-independent."""
    s = Settings()
    assert s.data_root.is_absolute()
    assert s.data_root == PROJECT_ROOT / "data"


def test_data_root_env_override(monkeypatch, tmp_path: Path) -> None:
    """QUANT_DATA_ROOT env var overrides the default."""
    monkeypatch.setenv("QUANT_DATA_ROOT", str(tmp_path / "override"))
    s = Settings()
    assert s.data_root == tmp_path / "override"


def test_data_root_stable_across_cwd(tmp_path: Path, project_root: Path) -> None:
    """Running from a different CWD must resolve to the same data_root."""
    # Run settings import in a subprocess with cwd=tmp_path; should still
    # resolve to the project's data folder, not tmp_path/data.
    env = os.environ.copy()
    # Ensure we don't inherit an override.
    env.pop("QUANT_DATA_ROOT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quant_lucky.utils.config import settings; print(settings.data_root)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    reported = Path(result.stdout.strip())
    assert reported == project_root / "data"
