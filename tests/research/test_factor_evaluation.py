"""评估台的口径：前向收益对齐与 RankIC 的独立样本校正。

历史教训：IC 用重叠样本算 t 值时，有效观测数会被放大 h 倍，噪声能被读成显著。
用合成数据钉死。
"""

from collections.abc import Callable

import numpy as np
import pytest

from src.research.factors.evaluation import (
    MIN_NAMES_PER_PERIOD,
    forward_returns,
    ic_stats,
)
from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel


def test_forward_returns_align_to_next_bars(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """h 根前向收益对齐到未来第 h 根；末尾 h 根没有前向收益。"""
    panel = panel_factory({"A-USDT": [100.0, 110.0, 121.0]})
    forward = forward_returns(panel, 1)

    assert forward["A-USDT"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(forward["A-USDT"].iloc[-1])


def test_rank_ic_is_one_for_perfect_ranking(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """因子排名与未来收益完全一致时 IC = 1，期数 = 有效截面数。"""
    panel, _ = ranked_panel
    stats = ic_stats(rank_spec(), panel, panel.close.notna(), horizon=1)

    assert stats.mean == pytest.approx(1.0)
    assert stats.periods == len(panel.index) - 1
    assert stats.independent == len(panel.index) - 1  # horizon=1 时重叠与非重叠等价
    assert stats.names == len(panel.symbols)


def test_rank_ic_flips_sign_with_ranking(
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """反向排名的 IC = -1：符号由因子排序决定，不受实现细节影响。"""
    panel, _ = ranked_panel
    stats = ic_stats(rank_spec(reverse=True), panel, panel.close.notna(), horizon=1)

    assert stats.mean == pytest.approx(-1.0)


def test_rank_ic_skips_thin_cross_sections(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """截面标的数不足时整期跳过，而不是拿几个标的算出一个 IC。"""
    thin = MIN_NAMES_PER_PERIOD - 2
    closes = {
        f"N{index}-USDT": [100.0 + index * bar for bar in range(3)]
        for index in range(thin)
    }
    panel = panel_factory(closes)
    stats = ic_stats(rank_spec(), panel, panel.close.notna(), horizon=1)

    assert stats.periods == 0
    assert stats.mean == 0.0
