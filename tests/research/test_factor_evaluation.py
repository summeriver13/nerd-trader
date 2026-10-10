"""评估台的口径：前向收益对齐、RankIC 的独立样本校正、分位价差与成本公式。

历史教训：IC 用重叠样本算 t 值时，有效观测数会被放大 h 倍，噪声能被读成显著；
分位组合的成本若与换手口径不一致，扣出来的净收益就没有意义。这两件事都用
手算的合成数据钉死。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.research.factors.evaluation import (
    MIN_NAMES_PER_PERIOD,
    PortfolioOptions,
    forward_returns,
    ic_stats,
    portfolio_stats,
)
from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel

NAMES = 20
PERIODS = 4
STEP = 0.01  # 第 i 个标的每根收益 = STEP × i
QUANTILES = 4


def _ranked_panel(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> tuple[Panel, list[float]]:
    """确定性行情：收益随标的下标线性递增，便于手算分位价差。"""
    rates = [STEP * index for index in range(NAMES)]
    closes = {
        f"N{index:02d}-USDT": [100.0 * (1 + rate) ** bar for bar in range(PERIODS)]
        for index, rate in enumerate(rates)
    }
    return panel_factory(closes), rates


def _rank_spec(reverse: bool = False) -> FactorSpec:
    """因子值 = 标的下标（越大越看多）；reverse 用于验证符号方向。"""
    sign = -1.0 if reverse else 1.0

    def compute(panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(
            {name: float(index) * sign for index, name in enumerate(panel.symbols)},
            index=panel.index,
        )

    return FactorSpec(
        key="rank",
        label="手算排名因子",
        family="测试",
        direction=1,
        warmup=0,
        compute=compute,
    )


def _alternating_spec(panel: Panel) -> FactorSpec:
    """每期排名翻转的因子：成分全换，用于钉住换手与成本公式。"""
    frame = pd.DataFrame(
        {
            name: [
                float(index) * (1 if bar % 2 == 0 else -1)
                for bar in range(len(panel.index))
            ]
            for index, name in enumerate(panel.symbols)
        },
        index=panel.index,
    )
    return FactorSpec(
        key="flip",
        label="翻转因子",
        family="测试",
        direction=1,
        warmup=0,
        compute=lambda _: frame,
    )


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
) -> None:
    """因子排名与未来收益完全一致时 IC = 1，期数 = 有效截面数。"""
    panel, _ = _ranked_panel(panel_factory)
    stats = ic_stats(_rank_spec(), panel, panel.close.notna(), horizon=1)

    assert stats.mean == pytest.approx(1.0)
    assert stats.periods == PERIODS - 1
    assert stats.independent == PERIODS - 1  # horizon=1 时重叠与非重叠等价
    assert stats.names == NAMES


def test_rank_ic_flips_sign_with_ranking(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """反向排名的 IC = -1：符号由因子排序决定，不受实现细节影响。"""
    panel, _ = _ranked_panel(panel_factory)
    stats = ic_stats(_rank_spec(reverse=True), panel, panel.close.notna(), horizon=1)

    assert stats.mean == pytest.approx(-1.0)


def test_rank_ic_skips_thin_cross_sections(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """截面标的数不足时整期跳过，而不是拿几个标的算出一个 IC。"""
    thin = MIN_NAMES_PER_PERIOD - 2
    closes = {
        f"N{index}-USDT": [100.0 + index * bar for bar in range(3)]
        for index in range(thin)
    }
    panel = panel_factory(closes)
    stats = ic_stats(_rank_spec(), panel, panel.close.notna(), horizon=1)

    assert stats.periods == 0
    assert stats.mean == 0.0


def test_quantile_spread_matches_hand_calculation(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """价差 = 最高分位平均收益 - 最低分位平均收益；成分不变时换手为 0。"""
    panel, rates = _ranked_panel(panel_factory)
    stats = portfolio_stats(
        _rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )
    expected = float(
        np.mean(rates[-NAMES // QUANTILES :]) - np.mean(rates[: NAMES // QUANTILES])
    )

    assert stats.gross_spread == pytest.approx(expected)
    assert stats.net_spread == pytest.approx(expected)
    assert stats.turnover == pytest.approx(0.0)
    assert stats.periods == PERIODS - 1
    assert stats.quantile_returns.is_monotonic_increasing


def test_turnover_and_cost_formula(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """成分每期全换：单腿换手 1，成本 = 4 × 换手 × 单边费率。"""
    panel, _ = _ranked_panel(panel_factory)
    cost_one_way = 0.001
    stats = portfolio_stats(
        _alternating_spec(panel),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=cost_one_way),
    )

    assert stats.turnover == pytest.approx(1.0)
    assert stats.net_spread == pytest.approx(stats.gross_spread - 4 * cost_one_way)


def test_long_only_net_equals_top_quantile_when_no_turnover(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """成分不变时只多腿没有换手成本，等于最高分位的毛收益。"""
    panel, _ = _ranked_panel(panel_factory)
    stats = portfolio_stats(
        _rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.001),
    )

    assert stats.long_turnover == pytest.approx(0.0)
    assert stats.long_only_net == pytest.approx(float(stats.quantile_returns.iloc[-1]))
    assert stats.long_only_annual == pytest.approx(
        stats.long_only_net * stats.periods / stats.span_years
    )


def test_long_only_leg_uses_its_own_turnover(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """只多腿成本 = 2ρ（不是多空价差的 4ρ）：用错会把成本算高一倍。"""
    panel, _ = _ranked_panel(panel_factory)
    cost_one_way = 0.001
    stats = portfolio_stats(
        _alternating_spec(panel),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=cost_one_way),
    )

    assert stats.long_turnover == pytest.approx(1.0)
    assert stats.short_turnover == pytest.approx(1.0)
    assert stats.turnover == pytest.approx(1.0)  # 两腿均值
    assert stats.long_only_net == pytest.approx(
        float(stats.quantile_returns.iloc[-1]) - 2 * cost_one_way
    )


def test_annualization_uses_realized_periods(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """年化按期数简单折算（spread 不是可复利曲线），期数少时不外推。"""
    panel, _ = _ranked_panel(panel_factory)
    stats = portfolio_stats(
        _rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )

    assert stats.span_years > 0
    assert stats.annual_gross == pytest.approx(
        stats.gross_spread * stats.periods / stats.span_years
    )
