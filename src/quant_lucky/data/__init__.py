"""Data acquisition, cleaning, and storage layer.

Responsibilities
----------------
- Download historical market data from multiple providers
  (Tushare, AKShare, yfinance, CCXT).
- Normalise schema across markets (OHLCV + adjustment factors).
- Persist to a uniform Parquet/DuckDB cache under ``data/`` with a
  partitioning scheme: ``<market>/<symbol>/<frequency>/<year>.parquet``.
- Expose a single high-level ``load()`` API that hides provider specifics.

Design rules
------------
- Raw data is immutable; transformations live in ``data/processed``.
- Always record the provider, download timestamp, and schema version
  in Parquet metadata for reproducibility.
"""
