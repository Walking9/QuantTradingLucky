"""Command-line interface for quant_lucky.

Entry points:
    python -m quant_lucky download --provider yfinance --symbol AAPL
    quant download --provider ccxt --exchange binance --symbol BTC/USDT --frequency 1h
    quant data ls

Run ``quant --help`` for full usage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click

from quant_lucky.data.downloader import Downloader
from quant_lucky.data.providers.ccxt_provider import CCXTProvider
from quant_lucky.data.providers.yfinance_provider import YFinanceProvider
from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import InventoryEntry, ParquetStore
from quant_lucky.utils.config import settings

_PROVIDERS = ["yfinance", "ccxt", "tushare"]


@click.group()
@click.version_option(package_name="quant-lucky")
def main() -> None:
    """QuantTradingLucky CLI."""


def _default_start() -> str:
    return (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")


def _default_end() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@main.command("download")
@click.option("--provider", type=click.Choice(_PROVIDERS), default="yfinance", show_default=True)
@click.option("--exchange", default="binance", show_default=True, help="ccxt only")
@click.option("--symbol", required=True, help="Symbol e.g. AAPL, BTC/USDT, 600519.SH")
@click.option("--start", default=_default_start, show_default=True, help="YYYY-MM-DD")
@click.option("--end", default=_default_end, show_default=True, help="YYYY-MM-DD")
@click.option(
    "--frequency",
    type=click.Choice([f.value for f in Frequency]),
    default=Frequency.DAILY.value,
    show_default=True,
)
@click.option("--force", is_flag=True, help="Re-download even if cached")
def download_cmd(
    provider: str,
    exchange: str,
    symbol: str,
    start: str,
    end: str,
    frequency: str,
    force: bool,
) -> None:
    """Download OHLCV data for a single symbol."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
    freq = Frequency(frequency)

    if provider == "yfinance":
        p = YFinanceProvider()
    elif provider == "ccxt":
        p = CCXTProvider(exchange_id=exchange)
    elif provider == "tushare":
        # Lazy import: TushareProvider() raises AuthenticationError if
        # TUSHARE_TOKEN isn't set, so we only import when actually used.
        from quant_lucky.data.providers.tushare_provider import (
            TushareProvider,
        )

        p = TushareProvider()
    else:  # pragma: no cover - click validates
        raise click.ClickException(f"Unknown provider: {provider}")

    dl = Downloader(provider=p)
    df = dl.download(symbol, start_dt, end_dt, freq, force=force)

    click.echo(f"\n✅ {len(df)} rows for {symbol} [{p.name}, {freq.value}]")
    click.echo(f"   range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    click.echo("\nHead:")
    click.echo(df.head().to_string(index=False))


# ---------------------------------------------------------------------------
# `quant data` — local cache inspection
# ---------------------------------------------------------------------------
@main.group("data")
def data_grp() -> None:
    """Inspect and manage the local data cache."""


@data_grp.command("ls")
@click.option("--provider", default=None, help="Restrict to one provider (exact match).")
@click.option(
    "--symbol",
    default=None,
    help="Filter by case-insensitive symbol substring.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit a JSON array instead of the tree view.",
)
def data_ls_cmd(provider: str | None, symbol: str | None, as_json: bool) -> None:
    """Show what data is cached under ``data/raw/``."""
    store = ParquetStore(root=settings.raw_dir)
    entries = list(store.iter_inventory(provider=provider, symbol_substring=symbol))

    if as_json:
        click.echo(json.dumps([_entry_to_dict(e) for e in entries], indent=2))
        return

    _render_inventory_tree(root=store.root, entries=entries)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_inventory_tree(*, root: Path, entries: list[InventoryEntry]) -> None:
    """Print an inventory tree to the terminal using ``rich``."""
    # Lazy import so plain ``quant --help`` doesn't pay the rich startup cost.
    from rich.console import Console
    from rich.tree import Tree

    console = Console()

    if not entries:
        console.print(
            f"[dim]{root}[/dim] is empty — " "run [cyan]quant download ...[/cyan] to populate."
        )
        return

    total_size = sum(e.size_bytes for e in entries)
    header = (
        f"[bold]{root}[/bold]  "
        f"([cyan]{len(entries)}[/cyan] files · {_humanize_size(total_size)})"
    )
    tree = Tree(header, guide_style="dim")

    # Group by provider preserving sorted order.
    by_provider: dict[str, list[InventoryEntry]] = {}
    for e in entries:
        by_provider.setdefault(e.provider, []).append(e)

    for prov, prov_entries in by_provider.items():
        prov_size = sum(e.size_bytes for e in prov_entries)
        prov_node = tree.add(
            f"[bold cyan]{prov}[/bold cyan]  "
            f"({len(prov_entries)} · {_humanize_size(prov_size)})"
        )

        # Group by symbol within provider.
        by_symbol: dict[str, list[InventoryEntry]] = {}
        for e in prov_entries:
            by_symbol.setdefault(e.symbol, []).append(e)

        for sym, sym_entries in by_symbol.items():
            sym_node = prov_node.add(f"[green]{sym}[/green]")
            for e in sym_entries:
                sym_node.add(_format_leaf(e))

    console.print(tree)


def _format_leaf(e: InventoryEntry) -> str:
    """One leaf line: ``1d   2024-01-01 → 2024-12-31 · 252 rows · 51.0 KB · 28d ago``."""
    if not e.is_healthy:
        err = e.error or "unknown error"
        return f"[red]{e.frequency}[/red]  [red]✗ {err}[/red]"

    pieces: list[str] = [f"[yellow]{e.frequency}[/yellow]"]
    if e.start is not None and e.end is not None:
        pieces.append(f"{e.start.date()} → {e.end.date()}")
    if e.row_count is not None:
        pieces.append(f"{e.row_count:,} rows")
    pieces.append(_humanize_size(e.size_bytes))
    if e.downloaded_at is not None:
        pieces.append(f"[dim]{_humanize_age(e.downloaded_at)}[/dim]")
    return "  ".join(pieces)


def _humanize_size(num_bytes: int) -> str:
    """Display bytes as KB / MB / GB with one decimal place."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    # Unreachable in practice; satisfy mypy.
    return f"{num_bytes} B"


def _humanize_age(downloaded_at: datetime) -> str:
    """Display a relative age like ``4d ago`` / ``2h ago`` / ``just now``."""
    now = datetime.now(UTC)
    # Coerce tz-naive metadata (shouldn't happen, but be defensive) to UTC.
    if downloaded_at.tzinfo is None:
        downloaded_at = downloaded_at.replace(tzinfo=UTC)
    delta = now - downloaded_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _entry_to_dict(e: InventoryEntry) -> dict[str, Any]:
    """Serialize an InventoryEntry to a JSON-safe dict."""
    return {
        "provider": e.provider,
        "symbol": e.symbol,
        "frequency": e.frequency,
        "path": str(e.path),
        "size_bytes": e.size_bytes,
        "row_count": e.row_count,
        "start": e.start.isoformat() if e.start else None,
        "end": e.end.isoformat() if e.end else None,
        "downloaded_at": e.downloaded_at.isoformat() if e.downloaded_at else None,
        "schema_version": e.schema_version,
        "extra_metadata": e.extra_metadata,
        "is_healthy": e.is_healthy,
        "error": e.error,
    }


if __name__ == "__main__":
    main()
