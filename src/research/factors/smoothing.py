"""
src/research/factors/smoothing.py
平滑扫描的计算层：把「变体 × 调仓周期 × 网格相位」跑成分位组合读数表。

为什么相位要跑三个：跨变体固定同一锚点（同相位）只解决可比性，解决不了**相位敏感性**——
实测 `ts_rank` 变体在 h=3 换锚点后只多腿净复利在 −12% ~ +31% 之间乱跳，而 `decay_linear`
变体稳在 10pp 内。一个只在某个相位好看的变体不是读数，所以 h>1 默认跑三个相位并把范围
带进汇总（`summarize`）。

判据口径：`break_even_bp` 是**多空价差**的平衡成本（回答「成本空间够不够」），
`long_net_compounded` 才是**可执行口径**（最高分位等权、扣该腿自己的换手成本）——
crypto 上做不了空头腿（借币成本 + 无上限），两者不能混读。详见 docs/factors/README.md 第二十八节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel
from src.research.factors.portfolio import (
    PortfolioOptions,
    PortfolioStats,
    leg_stats,
    portfolio_stats,
)

PHASES_PER_HORIZON: int = 3


@dataclass(frozen=True)
class ScanConfig:
    """扫描口径：跨变体必须一致，否则表内读数不可比。"""

    horizons: tuple[int, ...]
    quantiles: int
    cost_one_way: float
    single_anchor: bool = False

    def anchors(self, horizon: int) -> tuple[int, ...]:
        """每个调仓周期的网格锚点：h=1 只有一种网格，h>1 跑三个相位。"""
        if self.single_anchor or horizon == 1:
            return (0,)
        step = max(horizon // PHASES_PER_HORIZON, 1)
        return tuple(sorted({0, step, 2 * step}))


def scan(
    specs: list[tuple[str, FactorSpec]],
    panel: Panel,
    mask: pd.DataFrame,
    config: ScanConfig,
    anchor: int | None,
) -> pd.DataFrame:
    """逐（变体 × 调仓周期 × 相位）跑分位组合，收成一行一个读数的长表。"""
    records: list[dict[str, object]] = []
    for kind, spec in specs:
        for horizon in config.horizons:
            for phase in config.anchors(horizon):
                stats = portfolio_stats(
                    spec,
                    panel,
                    mask,
                    PortfolioOptions(
                        horizon=horizon,
                        quantiles=config.quantiles,
                        cost_one_way=config.cost_one_way,
                        # 拼体锚点（anchor=none）时不加相位偏移，否则就不是文档口径了
                        anchor=None if anchor is None else anchor + phase,
                    ),
                )
                records.append(
                    _record(kind, spec, horizon, phase, stats, config.cost_one_way)
                )
    return pd.DataFrame.from_records(records)


def _record(
    kind: str,
    spec: FactorSpec,
    horizon: int,
    phase: int,
    stats: PortfolioStats,
    cost_one_way: float,
) -> dict[str, object]:
    """一行读数：多空价差口径与可执行的只多腿口径并列（两者不能混用）。"""
    leg_net = stats.long_returns - 2.0 * stats.long_turnover * cost_one_way
    long_leg = leg_stats(leg_net, stats.periods_per_year)
    return {
        "kind": kind,
        "factor": spec.key,
        "base": spec.key.split("@")[0],
        "horizon": horizon,
        "phase": phase,
        "periods": stats.periods,
        "gross_bp": stats.gross_spread * 1e4,
        "net_bp": stats.net_spread * 1e4,
        "gross_annual": stats.annual_gross,
        "net_annual": stats.annual_net,
        "turnover": stats.turnover,
        "break_even_bp": stats.break_even_cost * 1e4,
        "over_cost": stats.break_even_cost >= cost_one_way,
        "long_net_annual": stats.long_only_annual,
        "long_net_compounded": long_leg.annual_compounded,
        "btc_compounded": stats.btc_leg.annual_compounded,
        "beats_btc": long_leg.annual_compounded > stats.btc_leg.annual_compounded,
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """把逐相位长表压成一行一个候选：换手/平衡取首相位，只多腿报中位与范围。"""
    first = "first"
    grouped = frame.sort_values("phase").groupby(
        ["kind", "factor", "base", "horizon"], as_index=False
    )
    out = cast(
        pd.DataFrame,
        grouped.agg(
            frames=("periods", first),
            turnover=("turnover", first),
            break_even_bp=("break_even_bp", first),
            gross_bp=("gross_bp", first),
            net_bp=("net_bp", first),
            long_med=("long_net_compounded", "median"),
            long_low=("long_net_compounded", "min"),
            long_high=("long_net_compounded", "max"),
            btc=("btc_compounded", first),
            beats=("beats_btc", "sum"),
            phases=("phase", "size"),
        ),
    )
    out["beats_all"] = out["beats"] == out["phases"]
    return out
