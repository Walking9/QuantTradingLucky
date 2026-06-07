"""Price-panel assembly for strategy backtests.

Strategies consume a wide ``date × asset`` close-price panel; the data
layer stores one tidy OHLCV frame per ``(provider, symbol)``. This module
is the seam: it reads the Parquet cache, normalises heterogeneous
timestamps (Yahoo bars stamped 05:00 UTC, AKShare 16:00 UTC, Binance
00:00 UTC) to plain calendar dates, and aligns the requested symbols onto
one index.

Cross-market alignment uses an **inner join** on dates by default: a row
survives only if *every* requested asset has a real price that day. This
deliberately drops crypto's weekend bars when equities are in the basket
— we would rather trade on days all venues are open than forward-fill a
stale equity price (a quiet look-ahead/────staleness bug). Pass
``how='outer'`` to keep the union and forward-fill instead, eyes open.

When the cache cannot satisfy a request (the S&P 500 cross-section is not
downloadable offline — see the project's 429 history), strategies fall
back to :func:`synthetic_price_panel`: a deterministic, seeded panel with
*no planted edge*. It exists to exercise the full pipeline reproducibly,
never to manufacture a flattering result — the strategy reports say so in
plain language.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_lucky.data.schema import Frequency
from quant_lucky.data.store import ParquetStore

__all__ = [
    "PriceSpec",
    "load_close_panel",
    "synthetic_price_panel",
]

#: A ``(provider, symbol)`` pair identifying one cached series. The label
#: defaults to ``symbol`` (or ``provider:symbol`` on collision).
PriceSpec = tuple[str, str]


def _to_calendar_dates(timestamps: pd.Series) -> pd.DatetimeIndex:
    """Normalise possibly tz-aware bar timestamps to tz-naive midnight."""
    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DatetimeIndex(ts.dt.normalize())


def load_close_panel(
    specs: list[PriceSpec],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    frequency: Frequency = Frequency.DAILY,
    how: str = "inner",
    store_root: Path | None = None,
    min_overlap: int = 2,
) -> pd.DataFrame:
    """Assemble a wide close-price panel from the Parquet cache.

    Args:
        specs: List of ``(provider, symbol)`` pairs to load.
        start, end: Optional inclusive date bounds (anything
            ``pd.Timestamp`` accepts).
        frequency: Bar frequency to read (daily by default).
        how: ``'inner'`` (default) keeps dates present for *all* assets;
            ``'outer'`` keeps the union and forward-fills.
        store_root: Override the cache root (defaults to the configured
            ``data/raw``).
        min_overlap: Minimum number of aligned rows required; below this
            the panel is considered unusable and ``ValueError`` is raised
            (callers typically catch this and fall back to synthetic).

    Returns:
        Wide ``date × label`` close prices, sorted by date.

    Raises:
        FileNotFoundError: a requested series is not cached.
        ValueError: fewer than ``min_overlap`` aligned rows, or ``how``
            invalid.
    """
    if how not in ("inner", "outer"):
        raise ValueError(f"how must be 'inner' or 'outer', got {how!r}")
    if not specs:
        raise ValueError("specs is empty")

    store = ParquetStore(root=store_root)
    series: dict[str, pd.Series] = {}
    label_counts: dict[str, int] = {}
    for provider, symbol in specs:
        df = store.read(provider, symbol, frequency)
        close = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=_to_calendar_dates(df["timestamp"]),
        )
        # Guard against accidental duplicate dates after normalisation.
        close = close[~close.index.duplicated(keep="last")].sort_index()
        label = symbol if label_counts.get(symbol, 0) == 0 else f"{provider}:{symbol}"
        label_counts[symbol] = label_counts.get(symbol, 0) + 1
        series[label] = close

    panel = pd.concat(series, axis=1, join=how)
    panel = panel.ffill() if how == "outer" else panel.dropna(how="any")

    if start is not None:
        panel = panel.loc[pd.Timestamp(start) :]
    if end is not None:
        panel = panel.loc[: pd.Timestamp(end)]

    panel = panel.sort_index()
    if len(panel) < min_overlap:
        raise ValueError(
            f"only {len(panel)} aligned rows for {[s for _, s in specs]} "
            f"(need >= {min_overlap}); cache too thin for this basket"
        )

    panel.index.name = "date"
    panel.columns.name = "asset"
    return panel


def synthetic_price_panel(
    *,
    n_assets: int = 40,
    n_days: int = 1260,
    seed: int = 20260601,
    start: str = "2018-01-02",
    annual_drift: float = 0.06,
    drift_dispersion: float = 0.10,
    market_vol: float = 0.16,
    idio_vol: float = 0.22,
    beta_low: float = 0.6,
    beta_high: float = 1.4,
    momentum_autocorr: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Deterministic synthetic equity panel — *no planted edge by default*.

    Returns are a one-factor model: a common market factor plus
    per-asset idiosyncratic noise, with heterogeneous betas and small
    heterogeneous drifts. With ``momentum_autocorr=0`` (default) there is
    **no** cross-sectional return persistence — a momentum strategy
    *should* earn ~0 gross and lose money net of costs on this panel. That
    is the point: it validates the pipeline's plumbing and its honesty
    signals (IS≈OOS≈noise, DSR≈0.5), not a discovered alpha.

    ``momentum_autocorr > 0`` injects a controllable amount of trailing-
    return persistence; the momentum report uses a small value purely to
    demonstrate that the machinery *can* capture an edge when one exists.
    Any such use is labelled as synthetic in the report.

    Args:
        n_assets: Number of synthetic names.
        n_days: Number of business days.
        seed: RNG seed → fully reproducible output.
        start: First business day.
        annual_drift: Mean annual drift across names.
        drift_dispersion: Std of per-name annual drift.
        market_vol: Annualised market-factor volatility.
        idio_vol: Annualised idiosyncratic volatility.
        beta_low, beta_high: Range of per-name market betas (uniform).
        momentum_autocorr: Strength of injected trailing-return
            persistence in ``[0, ~0.2]``. ``0`` = none (default).
        periods_per_year: Annualisation factor for drift/vol scaling.

    Returns:
        Wide ``date × asset`` close prices starting at 100.0, columns
        ``SYN00 … SYN{n-1}``.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if n_days < 2:
        raise ValueError(f"n_days must be >= 2, got {n_days}")

    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year

    betas = rng.uniform(beta_low, beta_high, size=n_assets)
    drifts = rng.normal(annual_drift, drift_dispersion, size=n_assets)

    market = rng.normal(annual_drift * dt, market_vol * np.sqrt(dt), size=n_days)
    idio = rng.normal(0.0, idio_vol * np.sqrt(dt), size=(n_days, n_assets))

    # Daily simple returns: idiosyncratic drift + beta·market + idio noise.
    rets = drifts * dt + np.outer(market, betas) + idio

    if momentum_autocorr > 0.0:
        # Inject mild persistence: nudge each day toward the sign of the
        # trailing 60-day cumulative return. Deterministic given the path.
        lookback = 60
        for t in range(lookback, n_days):
            window = rets[t - lookback : t].sum(axis=0)
            rets[t] += momentum_autocorr * np.sign(window) * idio_vol * np.sqrt(dt)

    dates = pd.bdate_range(start=start, periods=n_days)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    panel = pd.DataFrame(
        prices,
        index=dates,
        columns=[f"SYN{i:02d}" for i in range(n_assets)],
    )
    panel.index.name = "date"
    panel.columns.name = "asset"
    return panel
