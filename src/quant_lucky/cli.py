"""Command-line interface for quant_lucky.

Entry points:
    python -m quant_lucky download --provider yfinance --symbol AAPL
    quant download --provider ccxt --exchange binance --symbol BTC/USDT --frequency 1h

Run ``quant --help`` for full usage.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import click

from quant_lucky.data.downloader import Downloader
from quant_lucky.data.providers.ccxt_provider import CCXTProvider
from quant_lucky.data.providers.yfinance_provider import YFinanceProvider
from quant_lucky.data.schema import Frequency

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


if __name__ == "__main__":
    main()
