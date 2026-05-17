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

from collections.abc import Iterator
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class InventoryEntry:
    """One row in the local data cache inventory.

    Captures everything the inspection UI needs to display a Parquet
    file: its on-disk identity (provider/symbol/frequency/path),
    physical size, and the user-visible metadata stored inside the file
    (row count, date range, when it was downloaded).

    When a file is missing metadata or unreadable, the optional fields
    are ``None`` and ``is_healthy`` is False — the entry is still yielded
    so the developer can see (and clean up) bad files.
    """

    provider: str
    symbol: str  # filesystem-safe form (BTC-USDT, not BTC/USDT)
    frequency: str  # raw string; may not be a valid Frequency for corrupt files
    path: Path
    size_bytes: int
    row_count: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    downloaded_at: datetime | None = None
    schema_version: str | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)
    is_healthy: bool = True
    error: str | None = None


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

    # ----------------------------------------------------------------------
    # Inventory
    # ----------------------------------------------------------------------
    def iter_inventory(
        self,
        *,
        provider: str | None = None,
        symbol_substring: str | None = None,
    ) -> Iterator[InventoryEntry]:
        """Yield one ``InventoryEntry`` per cached Parquet file.

        Walks ``<root>/<provider>/<symbol>/<frequency>.parquet`` and emits
        one entry per file, sorted deterministically by
        ``(provider, symbol, frequency)``. Corrupt or partially-written
        files surface as entries with ``is_healthy=False`` rather than
        raising — the developer wants to see them so they can clean up.

        Args:
            provider: Restrict to one provider subtree (exact match).
            symbol_substring: Case-insensitive substring match on the
                on-disk symbol component.
        """
        if not self.root.is_dir():
            return

        # Use sorted() rather than glob's filesystem order so output is
        # deterministic across runs / OS / filesystems.
        candidates = sorted(self.root.glob("*/*/*.parquet"))

        sym_needle = symbol_substring.lower() if symbol_substring else None

        for path in candidates:
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue  # symlink pointing outside root; skip defensively
            parts = rel.parts
            if len(parts) != 3 or not parts[2].endswith(".parquet"):
                # Unexpected depth — log and skip rather than mis-parse.
                logger.warning("Skipping unexpected path in cache: {p}", p=path)
                continue
            entry_provider, entry_symbol, fname = parts
            entry_frequency = fname[: -len(".parquet")]

            if provider is not None and entry_provider != provider:
                continue
            if sym_needle is not None and sym_needle not in entry_symbol.lower():
                continue

            yield self._build_inventory_entry(
                provider=entry_provider,
                symbol=entry_symbol,
                frequency=entry_frequency,
                path=path,
            )

    @staticmethod
    def _build_inventory_entry(
        *,
        provider: str,
        symbol: str,
        frequency: str,
        path: Path,
    ) -> InventoryEntry:
        """Read metadata for one file; never raises."""
        try:
            size = path.stat().st_size
        except OSError as e:
            return InventoryEntry(
                provider=provider,
                symbol=symbol,
                frequency=frequency,
                path=path,
                size_bytes=0,
                is_healthy=False,
                error=f"stat failed: {e}",
            )

        try:
            md_raw = pq.ParquetFile(path).schema_arrow.metadata or {}
        except Exception as e:
            return InventoryEntry(
                provider=provider,
                symbol=symbol,
                frequency=frequency,
                path=path,
                size_bytes=size,
                is_healthy=False,
                error=f"unreadable parquet: {e}",
            )

        md = {
            k.decode(): v.decode()
            for k, v in md_raw.items()
            if not k.startswith(b"pandas")
        }

        known_keys = {
            "provider",
            "symbol",
            "frequency",
            "downloaded_at",
            "schema_version",
            "row_count",
            "start",
            "end",
        }
        extras = {k: v for k, v in md.items() if k not in known_keys}

        return InventoryEntry(
            provider=provider,
            symbol=symbol,
            frequency=frequency,
            path=path,
            size_bytes=size,
            row_count=_parse_int(md.get("row_count")),
            start=_parse_dt(md.get("start")),
            end=_parse_dt(md.get("end")),
            downloaded_at=_parse_dt(md.get("downloaded_at")),
            schema_version=md.get("schema_version"),
            extra_metadata=extras,
            is_healthy=True,
        )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
