"""Application configuration loaded from environment variables and .env.

Uses pydantic-settings for validation + type safety. All API credentials
are wrapped in ``SecretStr`` so they never end up in log output by accident.

The singleton ``settings`` object is imported throughout the codebase:

    from quant_lucky.utils.config import settings
    token = settings.tushare_token.get_secret_value()  # explicit unwrap
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up until we find a directory containing ``pyproject.toml``.

    Falls back to the current working directory if no marker is found.
    Anchoring paths and the ``.env`` lookup at the project root keeps
    behaviour identical whether code runs from repo root, ``notebooks/``
    or a test tmpdir.
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """Project-wide settings. Override any field via environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths -------------------------------------------------------------
    data_root: Path = Field(default=PROJECT_ROOT / "data", alias="QUANT_DATA_ROOT")

    @field_validator("data_root", mode="after")
    @classmethod
    def _anchor_data_root(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not CWD.

        A user with ``QUANT_DATA_ROOT=./data`` in ``.env`` should get the
        same cache location whether they launch from the repo root or
        from ``notebooks/``.
        """
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # --- Logging -----------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- A-share credentials ----------------------------------------------
    tushare_token: SecretStr | None = Field(default=None, alias="TUSHARE_TOKEN")

    # --- US / global credentials ------------------------------------------
    alpha_vantage_key: SecretStr | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    polygon_api_key: SecretStr | None = Field(default=None, alias="POLYGON_API_KEY")

    # --- Crypto credentials ------------------------------------------------
    binance_api_key: SecretStr | None = Field(default=None, alias="BINANCE_API_KEY")
    binance_api_secret: SecretStr | None = Field(default=None, alias="BINANCE_API_SECRET")
    okx_api_key: SecretStr | None = Field(default=None, alias="OKX_API_KEY")
    okx_api_secret: SecretStr | None = Field(default=None, alias="OKX_API_SECRET")
    okx_api_passphrase: SecretStr | None = Field(default=None, alias="OKX_API_PASSPHRASE")

    # --- Derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_root / "cache"


settings = Settings()
