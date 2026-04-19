"""Shared utilities: configuration, logging, datetime, IO helpers.

Keep this package thin. If a module grows beyond ~200 lines, promote it
to a first-class subpackage. Typical contents:

- ``config``   : pydantic-based settings loaded from .env + YAML.
- ``logging``  : loguru configuration with structured JSON option.
- ``timeutil`` : trading-calendar-aware date helpers.
- ``io``       : Parquet read/write with metadata, DuckDB helpers.
"""
