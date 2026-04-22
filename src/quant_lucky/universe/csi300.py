"""CSI 300 (沪深 300) universe builder.

The CSI 300 index represents the 300 largest and most liquid A-share
equities across the Shanghai and Shenzhen exchanges. Membership is
reviewed semi-annually (June + December). Index vendors (CSI, Wind) are
the authoritative source; free data providers that mirror the list
include AKShare, Tushare Pro, and choice.

Design
------
The builder separates *fetch* from *snapshot*: callers can either

- pass an explicit ``fetcher`` callable that returns ``list[str]``, or
- rely on a hard-coded :data:`SEED_UNIVERSE` that contains the most
  recent public constituent list at the time of writing.

The seed list is intentionally small (top-30 by weight) — just enough
to make unit tests and smoke demos meaningful without pinning the
library to a snapshot that will be stale within weeks. Production
pipelines should inject a real fetcher.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import ClassVar

from quant_lucky.universe.base import (
    UniverseBuilder,
    UniverseDataUnavailableError,
    UniverseSnapshot,
    coerce_date,
)
from quant_lucky.utils.logging import logger

#: Seed universe: top A-share constituents by market cap at 2025-Q4.
#: Exchange suffixes follow the common ``<code>.SH``/``.SZ`` convention.
#: This is **not** a full CSI 300 list — it is a representative subset
#: meant for offline demos and tests. Always replace with a real
#: ``fetcher`` in production.
SEED_UNIVERSE: tuple[str, ...] = (
    "600519.SH",  # 贵州茅台
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "000858.SZ",  # 五粮液
    "600900.SH",  # 长江电力
    "601888.SH",  # 中国中免
    "000333.SZ",  # 美的集团
    "600030.SH",  # 中信证券
    "601166.SH",  # 兴业银行
    "002594.SZ",  # 比亚迪
    "600276.SH",  # 恒瑞医药
    "601398.SH",  # 工商银行
    "601939.SH",  # 建设银行
    "600050.SH",  # 中国联通
    "601628.SH",  # 中国人寿
    "300750.SZ",  # 宁德时代
    "601857.SH",  # 中国石油
    "000001.SZ",  # 平安银行
    "600887.SH",  # 伊利股份
    "601088.SH",  # 中国神华
    "600016.SH",  # 民生银行
    "000002.SZ",  # 万科A
    "600000.SH",  # 浦发银行
    "002304.SZ",  # 洋河股份
    "600031.SH",  # 三一重工
    "000651.SZ",  # 格力电器
    "600048.SH",  # 保利发展
    "601668.SH",  # 中国建筑
    "600585.SH",  # 海螺水泥
    "601012.SH",  # 隆基绿能
)

Fetcher = Callable[[date], list[str] | tuple[str, ...]]


@dataclass
class CSI300Universe(UniverseBuilder):
    """CSI 300 universe builder.

    Parameters
    ----------
    fetcher:
        Optional callable ``(as_of) -> list[str]`` returning CSI 300
        members on ``as_of``. When not provided the seed list is used.
    use_seed_on_error:
        If True (default) and ``fetcher`` raises, fall back to the
        seed list and log a warning. Set False to propagate errors.
    """

    fetcher: Fetcher | None = None
    use_seed_on_error: bool = True
    seed: tuple[str, ...] = field(default_factory=lambda: SEED_UNIVERSE)
    name: ClassVar[str] = "csi300"

    def snapshot(self, as_of: date | datetime | None = None) -> UniverseSnapshot:
        target = coerce_date(as_of)

        if self.fetcher is None:
            return self._seed_snapshot(target, reason="no fetcher configured")

        try:
            members = tuple(self.fetcher(target))
        except Exception as exc:  # pragma: no cover - defensive
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError(f"CSI300 fetcher failed: {exc}") from exc
            logger.warning(
                "CSI300 fetcher failed ({err}); falling back to seed list",
                err=exc,
            )
            return self._seed_snapshot(target, reason=f"fetcher error: {exc}")

        if not members:
            if not self.use_seed_on_error:
                raise UniverseDataUnavailableError("CSI300 fetcher returned empty")
            return self._seed_snapshot(target, reason="fetcher returned empty")

        return UniverseSnapshot(
            as_of=target,
            members=members,
            metadata={"builder": self.name, "source": "fetcher"},
        )

    def _seed_snapshot(self, target: date, *, reason: str) -> UniverseSnapshot:
        return UniverseSnapshot(
            as_of=target,
            members=self.seed,
            metadata={"builder": self.name, "source": "seed", "reason": reason},
        )
