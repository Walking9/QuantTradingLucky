"""Shared CLI plumbing for the runnable ``strategies/<name>/`` packages.

Each package's ``run_backtest.py`` does the same three boring things:
load a YAML config (with dotted-key CLI overrides), and print a concise
terminal summary of the resulting :class:`StrategyEvaluation`. Those two
helpers live here so the entry points stay focused on the *orchestration*
(load prices → build spec → :func:`run_research` → write artifacts) rather
than on argument fiddling.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from quant_lucky.strategies.evaluation import StrategyEvaluation

__all__ = ["format_summary", "load_config"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config['a']['b'] = value`` from a dotted ``'a.b'`` key."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def load_config(
    default_config: dict[str, Any],
    *,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Merge defaults ← YAML file ← CLI ``--set`` overrides.

    Args:
        default_config: The package's in-code defaults (the safety net if
            the YAML file is missing a key).
        config_path: Optional path to a ``config.yaml`` to deep-merge on
            top of the defaults.
        overrides: List of ``"dotted.key=value"`` strings. Each value is
            parsed as a YAML scalar, so ``signal.long_only=true`` becomes
            a real bool and ``synthetic.autocorr=0.12`` a real float.

    Returns:
        The merged configuration dict.
    """
    config = copy.deepcopy(default_config)
    if config_path is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
        config = _deep_merge(config, loaded)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects 'key=value', got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(config, key.strip(), yaml.safe_load(raw))

    return config


def format_summary(evaluation: StrategyEvaluation) -> str:
    """A compact, terminal-friendly 'did it work?' block."""
    ev = evaluation
    full = ev.full.report
    is_r = ev.split.is_report
    oos = ev.split.oos_report
    lines = [
        f"━━━ {ev.name} ━━━",
        f"data: {ev.data_source}  |  {ev.n_assets} assets  |  "
        f"{ev.span[0].date()} → {ev.span[1].date()}  |  cost {ev.cost_bps:.1f} bps/side",
        "",
        f"{'window':<14}{'ann_ret':>10}{'ann_vol':>10}{'sharpe':>9}{'max_dd':>10}",
        f"{'full':<14}{full.annual_return:>9.2%}{full.annual_vol:>10.2%}"
        f"{full.sharpe:>9.2f}{full.max_drawdown:>10.2%}",
        f"{'in-sample':<14}{is_r.annual_return:>9.2%}{is_r.annual_vol:>10.2%}"
        f"{is_r.sharpe:>9.2f}{is_r.max_drawdown:>10.2%}",
        f"{'out-sample':<14}{oos.annual_return:>9.2%}{oos.annual_vol:>10.2%}"
        f"{oos.sharpe:>9.2f}{oos.max_drawdown:>10.2%}",
    ]
    if ev.benchmark is not None:
        bm = ev.benchmark.report
        lines.append(
            f"{'benchmark':<14}{bm.annual_return:>9.2%}{bm.annual_vol:>10.2%}"
            f"{bm.sharpe:>9.2f}{bm.max_drawdown:>10.2%}"
        )
    if full.turnover_annual is not None:
        lines.append(f"\nannual one-way turnover: {full.turnover_annual:.2f}")
    if ev.attribution is not None:
        a = ev.attribution
        lines.append(
            f"attribution vs benchmark: beta={a.beta:.3f}  "
            f"alpha={a.alpha_annual:.2%}/yr  corr={a.correlation:.3f}"
        )
    if ev.walk_forward is not None:
        wf = ev.walk_forward
        agg = wf.extra.get("aggregate_oos_sharpe", float("nan"))
        lines.append(
            f"walk-forward: {len(wf.splits)} splits, {wf.n_trials} trials  |  "
            f"agg OOS Sharpe {agg:.2f}  |  DSR {wf.deflated_sharpe:.3f}"
        )
    return "\n".join(lines)
