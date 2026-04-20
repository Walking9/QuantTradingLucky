"""Concrete data providers.

Each provider encapsulates a single data source. They all implement the
:class:`quant_lucky.data.base.DataProvider` interface so callers can swap
them transparently.
"""

from __future__ import annotations

from quant_lucky.data.providers.ccxt_provider import CCXTProvider
from quant_lucky.data.providers.tushare_provider import TushareProvider
from quant_lucky.data.providers.yfinance_provider import YFinanceProvider

__all__ = ["CCXTProvider", "TushareProvider", "YFinanceProvider"]
