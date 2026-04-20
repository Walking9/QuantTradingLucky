"""Loguru-based logging with sensible defaults.

Import ``logger`` anywhere and it's ready to use:

    from quant_lucky.utils.logging import logger
    logger.info("Fetching {symbol}", symbol="AAPL")

Tests silence logging automatically via ``caplog``; no need to mock.
"""

from __future__ import annotations

import sys

from loguru import logger

from quant_lucky.utils.config import settings

_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(level: str | None = None) -> None:
    """(Re)configure the root loguru sink."""
    logger.remove()
    logger.add(sys.stderr, level=level or settings.log_level, format=_FMT)


# Auto-configure on first import
configure_logging()

__all__ = ["configure_logging", "logger"]
