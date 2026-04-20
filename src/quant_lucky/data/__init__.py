"""Data acquisition, cleaning, and storage layer.

Responsibilities
----------------
- Download historical market data from multiple providers
  (Tushare, AKShare, yfinance, CCXT).
- Normalise schema across markets (OHLCV + adjustment factors).
- Persist to a uniform Parquet cache under ``data/`` with a partitioning
  scheme: ``<provider>/<symbol>/<frequency>.parquet``.
- Expose a single high-level :class:`Downloader` API that hides provider
  specifics.

Design rules
------------
- Raw data is immutable; transformations live in ``data/processed``.
- Always record the provider, download timestamp, and schema version
  in Parquet metadata for reproducibility.
"""

from __future__ import annotations

from quant_lucky.data.base import (
    AuthenticationError,
    DataProvider,
    DataProviderError,
    DownloadRequest,
    RateLimitError,
    SymbolNotFoundError,
)
from quant_lucky.data.downloader import Downloader
from quant_lucky.data.providers import CCXTProvider, TushareProvider, YFinanceProvider
from quant_lucky.data.schema import (
    OHLCV_COLUMNS,
    OPTIONAL_COLUMNS,
    Frequency,
    Market,
    validate_ohlcv,
)
from quant_lucky.data.store import ParquetStore

__all__ = [
    # Schema
    "Frequency",
    "Market",
    "OHLCV_COLUMNS",
    "OPTIONAL_COLUMNS",
    "validate_ohlcv",
    # Base
    "DataProvider",
    "DataProviderError",
    "DownloadRequest",
    "AuthenticationError",
    "RateLimitError",
    "SymbolNotFoundError",
    # Concrete
    "YFinanceProvider",
    "CCXTProvider",
    "TushareProvider",
    # Orchestration
    "Downloader",
    "ParquetStore",
]
