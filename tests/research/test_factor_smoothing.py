"""
平滑扫描计算层的两条不变量：相位汇总与「胜 BTC」必须按相位数判定。

历史教训：只报单一锚点时，`ts_rank` 变体在 h=3 换锚点后能从 −12% 跳到 +31%——
汇总若丢掉相位信息，筛选就把「某个相位好看」当成读数。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.factors.library import resolve_factors
from src.research.factors.panel import Panel
from src.research.factors.smoothing import (
    ScanConfig,
    summarize,
    verify_grid,
)
from src.research.factors.variants import resolve_smoothing

HORIZON = 9
# 判定路径要真跑分位组合：得跨过 5 档 × 每档最少 3 只的门槛，且期数要够算 DSR
VERIFY_SYMBOLS = 40
VERIFY_BARS = 240


def _frame(phases: list[float], beats: list[bool]) -> pd.DataFrame:
    """再造一张长表：同一候选的三个相位读数。"""
    return pd.DataFrame(
        {
            "kind": ["平滑"] * len(phases),
            "factor": ["x@decay5"] * len(phases),
            "base": ["x"] * len(phases),
            "horizon": [HORIZON] * len(phases),
            "phase": [0, 3, 6][: len(phases)],
            "periods": [100] * len(phases),
            "turnover": [0.1, 0.12, 0.11][: len(phases)],
            "break_even_bp": [20.0, 21.0, 19.0][: len(phases)],
            "gross_bp": [5.0] * len(phases),
            "net_bp": [3.0] * len(phases),
            "long_net_compounded": phases,
            "btc_compounded": [0.3] * len(phases),
            "beats_btc": beats,
            "over_cost": [True] * len(phases),
        }
    )


def test_anchors_cover_three_phases_for_multi_bar_horizon() -> None:
    """h>1 默认三相位（0/h÷3/2h÷3），h=1 只有一种网格，--single-anchor 关掉扩展。"""
    config = ScanConfig(horizons=(1, HORIZON), quantiles=5, cost_one_way=0.001)
    assert config.anchors(1) == (0,)
    assert config.anchors(HORIZON) == (0, 3, 6)

    single = ScanConfig(
        horizons=(HORIZON,), quantiles=5, cost_one_way=0.001, single_anchor=True
    )
    assert single.anchors(HORIZON) == (0,)


def test_summarize_reports_median_and_range_across_phases() -> None:
    """汇总取相位中位与范围，换手/平衡成本取首相位。"""
    out = summarize(_frame([0.10, 0.60, 0.20], [True, True, True]))
    row = out.iloc[0]

    assert row["long_med"] == pytest.approx(0.20)
    assert row["long_low"] == pytest.approx(0.10)
    assert row["long_high"] == pytest.approx(0.60)
    assert row["turnover"] == pytest.approx(0.10)
    assert row["break_even_bp"] == pytest.approx(20.0)
    assert row["phases"] == 3


def test_beats_all_requires_every_phase() -> None:
    """「胜 BTC」必须三个相位全过：只在部分相位赢的候选不算。"""
    all_win = summarize(_frame([0.4, 0.5, 0.6], [True, True, True])).iloc[0]
    partial = summarize(_frame([0.4, 0.5, 0.6], [True, True, False])).iloc[0]

    assert all_win["beats"] == 3
    assert bool(all_win["beats_all"]) is True
    assert partial["beats"] == 2
    assert bool(partial["beats_all"]) is False


def _verify_panel() -> Panel:
    """合成多标的局面板：40 只 × 240 根，收益确定可由种子公式重算。"""
    index = pd.date_range("2020-01-01", periods=VERIFY_BARS, freq="1D", tz="UTC")
    columns = [f"S{symbol:02d}-USDT" for symbol in range(VERIFY_SYMBOLS)]
    close = pd.DataFrame(
        {
            name: [
                100.0 * 1.01 ** (bar % 7 - 3) * (1.0 + 0.001 * (bar + int(name[1:3])))
                for bar in range(VERIFY_BARS)
            ]
            for name in columns
        },
        index=index,
    )
    volume = pd.DataFrame(1.0e6, index=index, columns=pd.Index(columns, dtype="object"))
    return Panel(timeframe="1d", close=close, volume=volume, notional=close * volume)


def test_verify_grid_runs_each_horizon_and_echoes_declared_trials() -> None:
    """判定路径：逐调仓周期各出一条 verdict，候选齐、N 用声明值（不是候选数）。"""
    panel = _verify_panel()
    mask = pd.DataFrame(True, index=panel.index, columns=panel.close.columns)
    specs = [("基准", spec) for spec in resolve_factors("rev_3")]
    specs += [("平滑", spec) for spec in resolve_smoothing("rev_3", "decay5")]
    config = ScanConfig(horizons=(1, 3), quantiles=5, cost_one_way=0.001)
    declared = 123

    verdicts = verify_grid(specs, panel, mask, config, declared)

    assert [verdict.horizon for verdict in verdicts] == [1, 3]
    for verdict in verdicts:
        assert verdict.trials == declared
        assert verdict.candidates == len(specs)
        # 候选一个不能少，且每个都得带出可裁定的 DSR 字段
        assert set(verdict.rows["factor"]) == {spec.key for _, spec in specs}
        assert bool(verdict.rows["sufficient"].all())
        assert verdict.rows["dsr"].between(0.0, 1.0).all()
        assert verdict.observations > 0
    # 判定用的是只多腿净收益序列，不是多空价差：年化必须在只多腿的合理量级内
    assert verdicts[0].rows["annual"].abs().max() < 10.0
