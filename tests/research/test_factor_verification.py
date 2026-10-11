"""收益层验证：候选池构造、定向、网格锚点与逐期限 PBO。

历史教训（两条都是本轮真踩到的）：

1. 扫描不预设方向，做空高分位的候选原始价差为负。拿**未定向**的价差算平衡成本会
   把「107bp 的宽边际」错报成 0（`PortfolioStats.break_even_cost` 在毛价差为负时
   按设计返回 0，那个设计对单因子回测是对的，对定向后的候选是错的）。
2. 分位网格原来锚在 `spec.warmup` 上，不同窗口的**相位不同**；h=2 时奇偶窗口的
   网格完全不相交，PBO 的 `dropna(how="any")` 会把整个矩阵清空、静默报「不可算」。
   给同一个锚点才让候选落在同一时间轴上。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.research.factors.panel import Panel
from src.research.factors.portfolio import (
    PortfolioOptions,
    break_even_cost,
    portfolio_stats,
)
from src.research.factors.sweep import SWEEP_FAMILIES
from src.research.factors.verification import (
    PlateauPick,
    _row,
    oriented_spread,
    picks_from_candidates,
    picks_from_plateaus,
    representative_window,
    verify,
)

type PanelFactory = Callable[[dict[str, list[float]]], Panel]

NAMES = 20
QUANTILES = 5
SEED = 20261012


def _panel(panel_factory: PanelFactory, periods: int, rate: float = 0.002) -> Panel:
    """第 i 个标的带确定性扰动地上行：排名随窗口变化，候选之间不会退化成同一条线。"""
    rng = np.random.default_rng(SEED)
    shocks = rng.normal(0.0, 0.01, size=(periods, NAMES))
    growth = rate * np.arange(NAMES)
    closes = 100.0 * np.cumprod(1.0 + growth + shocks, axis=0)
    return panel_factory(
        {f"N{index:02d}-USDT": closes[:, index].tolist() for index in range(NAMES)}
    )


def _plateaus(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_representative_window_is_the_midpoint() -> None:
    """代表窗口取中点：平台的端点最容易被人事后挑出来。"""
    assert representative_window(10, 20) == 15
    assert representative_window(7, 7) == 7


def test_picks_from_plateaus_skips_spikes_and_reads_direction() -> None:
    """尖峰不参与收益层验证；方向由平台自己的 IC 符号决定。"""
    table = _plateaus(
        [
            {
                "family": "mom",
                "horizon": 1,
                "start": 10,
                "end": 20,
                "mean_ic": -0.03,
                "kind": "平台",
            },
            {
                "family": "vol",
                "horizon": 1,
                "start": 5,
                "end": 5,
                "mean_ic": 0.08,
                "kind": "尖峰",
            },
        ]
    )
    picks = picks_from_plateaus(table)
    assert [pick.family for pick in picks] == ["mom"]
    assert picks[0].window == 15
    assert picks[0].direction == -1.0


def test_picks_from_candidates_filters_horizons() -> None:
    ic = pd.DataFrame(
        [
            {"family": "mom", "horizon": 1, "window": 3, "ic": -0.04},
            {"family": "vol", "horizon": 30, "window": 60, "ic": 0.05},
        ]
    )
    assert len(picks_from_candidates(ic)) == 2
    only_long = picks_from_candidates(ic, [30])
    assert [pick.family for pick in only_long] == ["vol"]
    assert only_long[0].direction == 1.0


def test_oriented_spread_follows_direction(panel_factory: PanelFactory) -> None:
    panel = _panel(panel_factory, periods=60)
    stats = portfolio_stats(
        SWEEP_FAMILIES["mom"].spec(3),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )
    short = PlateauPick("mom", 1, 3, -1.0)
    long = PlateauPick("mom", 1, 3, 1.0)
    flipped = oriented_spread(short, stats)
    straight = oriented_spread(long, stats)
    pd.testing.assert_series_equal(flipped, -straight)


def test_break_even_uses_oriented_spread(panel_factory: PanelFactory) -> None:
    """平衡成本必须用定向后的价差算。

    原始价差为负（做空高分位）而定向后为正时，用原始值算会得到 0——修复前
    `_row` 就是这么错的，把 mom/vol 两族上百 bp 的边际全报成了 0。
    """
    panel = _panel(panel_factory, periods=60)
    base = portfolio_stats(
        SWEEP_FAMILIES["mom"].spec(3),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )
    stats = dataclasses.replace(base, gross_spread=-0.02, turnover=0.5)
    row = _row(PlateauPick("mom", 1, 3, -1.0), stats)
    assert row["gross_per_period"] == pytest.approx(0.02)
    assert row["break_even_cost"] == pytest.approx(0.02 / (4 * 0.5))
    assert row["break_even_cost"] == pytest.approx(break_even_cost(0.02, 0.5))


def test_anchor_puts_candidates_on_one_phase(panel_factory: PanelFactory) -> None:
    """不给锚点时 h=2 的奇偶窗口网格完全不相交；给了锚点才共用同一相位。"""
    panel = _panel(panel_factory, periods=120)
    mask = panel.close.notna()
    even = SWEEP_FAMILIES["mom"].spec(2)
    odd = SWEEP_FAMILIES["mom"].spec(3)
    bare = PortfolioOptions(horizon=2, quantiles=QUANTILES, cost_one_way=0.0)
    anchored = dataclasses.replace(bare, anchor=0)

    def dates(spec_index: int, options: PortfolioOptions) -> pd.Index:
        spec = even if spec_index == 0 else odd
        return portfolio_stats(spec, panel, mask, options).spread_returns.index

    assert dates(0, bare).intersection(dates(1, bare)).empty
    assert len(dates(0, anchored).intersection(dates(1, anchored))) > 10


def test_verify_reports_pbo_per_horizon(panel_factory: PanelFactory) -> None:
    """同一期限的候选才是直接竞争关系，裁决按期限分开给。"""
    panel = _panel(panel_factory, periods=300)
    picks = [PlateauPick("mom", 1, window, -1.0) for window in (2, 3, 4, 5)]
    result = verify(picks, panel, panel.close.notna())
    assert len(result.by_horizon) == 1
    row = result.by_horizon.iloc[0].to_dict()
    assert int(row["horizon"]) == 1
    assert int(row["candidates"]) == 4
    assert 0.0 <= float(row["pbo"]) <= 1.0


def test_verify_keeps_readings_when_pbo_is_impossible(
    panel_factory: PanelFactory,
) -> None:
    """期限太长时分不出 4 块：报「不可算」，但收益层读数照给。"""
    panel = _panel(panel_factory, periods=300)
    picks = [PlateauPick("mom", 30, window, -1.0) for window in (2, 3, 4, 5)]
    result = verify(picks, panel, panel.close.notna())
    assert result.by_horizon.empty
    assert len(result.picks) == 4
    assert (result.picks["periods"] > 0).all()


def test_verify_is_empty_for_no_picks(panel_factory: PanelFactory) -> None:
    panel = _panel(panel_factory, periods=60)
    result = verify([], panel, panel.close.notna())
    assert result.picks.empty
    assert result.by_horizon.empty
