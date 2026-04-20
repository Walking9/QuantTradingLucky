"""Abstract base class for data providers.

A provider is anything that can translate a ``DownloadRequest`` (symbol +
date range + frequency) into a canonical OHLCV DataFrame. Concrete
providers live under ``quant_lucky.data.providers``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_lucky.data.schema import Frequency, Market


class DataProviderError(Exception):
    """Base class for all provider-side errors."""


class AuthenticationError(DataProviderError):
    """Provider credentials are missing or invalid."""


class SymbolNotFoundError(DataProviderError):
    """Symbol not found at the provider."""


class RateLimitError(DataProviderError):
    """Provider-imposed rate limit hit."""


@dataclass(frozen=True)
class DownloadRequest:
    """Immutable description of a single download task."""

    symbol: str
    start: datetime
    end: datetime
    frequency: Frequency
    market: Market | None = None  # optional hint


class DataProvider(ABC):
    """Base class every concrete provider inherits from.

    Subclasses must set ``name``, ``supported_markets`` and
    ``supported_frequencies`` and implement :meth:`fetch`.
    """

    name: str = "base"
    supported_markets: set[Market] = set()
    supported_frequencies: set[Frequency] = set()
    requires_credentials: bool = False

    @abstractmethod
    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        """Fetch OHLCV data for one symbol.

        Returns a DataFrame conforming to the canonical schema in
        :mod:`quant_lucky.data.schema`.
        """
        raise NotImplementedError

    def supports(self, request: DownloadRequest) -> bool:
        """Return True if this provider can serve ``request``."""
        freq_ok = request.frequency in self.supported_frequencies
        market_ok = request.market is None or request.market in self.supported_markets
        return freq_ok and market_ok
