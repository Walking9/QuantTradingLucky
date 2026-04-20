"""Parquet-based storage for OHLCV data with schema metadata.

Layout::

    <root>/<provider>/<safe_symbol>/<frequency>.parquet

where ``safe_symbol`` replaces filesystem-unsafe characters like ``/`` and
``:`` so that e.g. ``BTC/USDT`` becomes ``BTC-USDT``.

Each file carries Parquet key/value metadata including provider name,
symbol, frequency, download timestamp and schema version, so the data is
self-describing and can be audited years later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_lucky.data.schema import Frequency, validate_ohlcv
from quant_lucky.utils.config import settings
from quant_lucky.utils.logging import logger

SCHEMA_VERSION = "1"


def _safe_symbol(symbol: str) -> str:
    """Map a market symbol to a filesystem-safe path component."""
    return symbol.replace("/", "-").replace(":", "-").replace("\\", "-")


class ParquetStore:
    """Read/write OHLCV parquet files under a configurable root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else settings.raw_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Path helpers
    # ----------------------------------------------------------------------
    def path_for(self, provider: str, symbol: str, frequency: Frequency) -> Path:
        return self.root / provider / _safe_symbol(symbol) / f"{frequency.value}.parquet"

    def exists(self, provider: str, symbol: str, frequency: Frequency) -> bool:
        return self.path_for(provider, symbol, frequency).is_file()

    # ----------------------------------------------------------------------
    # IO
    # ----------------------------------------------------------------------
    def write(
        self,
        df: pd.DataFrame,
        *,
        provider: str,
        symbol: str,
        frequency: Frequency,
        extra_metadata: dict[str, str] | None = None,
    ) -> Path:
        """Validate ``df`` then write as Parquet with enriched metadata."""
        validate_ohlcv(df)
        path = self.path_for(provider, symbol, frequency)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df, preserve_index=False)

        metadata: dict[bytes, bytes] = {
            b"provider": provider.encode(),
            b"symbol": symbol.encode(),
            b"frequency": frequency.value.encode(),
            b"downloaded_at": datetime.now(timezone.utc).isoformat().encode(),
            b"schema_version": SCHEMA_VERSION.encode(),
            b"row_count": str(len(df)).encode(),
            b"start": df["timestamp"].iloc[0].isoformat().encode(),
            b"end": df["timestamp"].iloc[-1].isoformat().encode(),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                metadata[k.encode()] = v.encode()

        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, **metadata})

        pq.write_table(table, path, compression="snappy")
        logger.info("Wrote {n} rows to {path}", n=len(df), path=path)
        return path

    def read(self, provider: str, symbol: str, frequency: Frequency) -> pd.DataFrame:
        """Load a previously-written OHLCV DataFrame."""
        path = self.path_for(provider, symbol, frequency)
        if not path.is_file():
            raise FileNotFoundError(f"No cached data at {path}")
        return pd.read_parquet(path)

    def read_metadata(
        self, provider: str, symbol: str, frequency: Frequency
    ) -> dict[str, str]:
        """Return the key/value metadata stored alongside the Parquet file."""
        path = self.path_for(provider, symbol, frequency)
        if not path.is_file():
            raise FileNotFoundError(f"No cached data at {path}")
        md = pq.ParquetFile(path).schema_arrow.metadata or {}
        # Skip pandas-internal keys (b"pandas" -> JSON blob)
        return {
            k.decode(): v.decode()
            for k, v in md.items()
            if not k.startswith(b"pandas")
        }
