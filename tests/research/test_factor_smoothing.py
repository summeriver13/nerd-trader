"""
平滑扫描计算层的两条不变量：相位汇总与「胜 BTC」必须按相位数判定。

历史教训：只报单一锚点时，`ts_rank` 变体在 h=3 换锚点后能从 −12% 跳到 +31%——
汇总若丢掉相位信息，筛选就把「某个相位好看」当成读数。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.factors.smoothing import ScanConfig, summarize

HORIZON = 9


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
