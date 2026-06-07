#!/usr/bin/env python
"""Run the ``momentum_us_monthly_top20`` backtest end-to-end.

    python strategies/momentum_us_monthly_top20/run_backtest.py
    python strategies/momentum_us_monthly_top20/run_backtest.py --set synthetic.autocorr=0.12
    python strategies/momentum_us_monthly_top20/run_backtest.py --set signal.long_only=true

Loads ``config.yaml`` (with optional dotted ``--set`` overrides), assembles
the price panel (real cross-section if cached, else the synthetic edgeless
fallback), runs the full research bundle (full-sample backtest + IS/OOS
split + benchmark attribution + Walk-Forward/DSR + parameter sensitivity)
via :func:`quant_lucky.strategies.run_research`, prints a summary, and
writes reproducible artifacts to ``reports/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strategy  # noqa: E402  — sibling module (this dir is on sys.path)

from quant_lucky.strategies.cli import format_summary, load_config  # noqa: E402
from quant_lucky.strategies.evaluation import run_research, write_artifacts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=strategy.NAME)
    parser.add_argument("--config", type=Path, default=strategy.CONFIG_PATH)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key config override, e.g. synthetic.autocorr=0.12",
    )
    parser.add_argument("--no-write", action="store_true", help="skip writing report artifacts")
    args = parser.parse_args(argv)

    config = load_config(strategy.DEFAULT_CONFIG, config_path=args.config, overrides=args.overrides)
    prices, data_source = strategy.load_prices(config)
    spec = strategy.build_spec(config, prices, data_source)
    evaluation = run_research(spec)

    print(format_summary(evaluation))
    if not args.no_write:
        written = write_artifacts(evaluation, HERE / "reports")
        print("\nartifacts:")
        for path in written:
            print(f"  {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
