"""旁证口径：因子相关、容量上限、幸存者偏差代理。

这三项都是「结论能不能用」的前提：重复因子不该当两个因子用、容量决定收益能不能落地、
缓存里 0 个退市标的（实测）意味着小市值类因子的收益里混着幸存者偏差。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.research.factors.diagnostics import (
    capacity_usd,
    factor_correlation,
    survivorship_probe,
)
from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel, UniverseRule

NAMES = 20
PERIODS = 6
STEP = 0.01


def _ranked_panel(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> Panel:
    closes = {
        f"N{index:02d}-USDT": [
            100.0 * (1 + STEP * index) ** bar for bar in range(PERIODS)
        ]
        for index in range(NAMES)
    }
    return panel_factory(closes)


def _rank_spec(key: str, reverse: bool = False) -> FactorSpec:
    """因子值 = 标的下标（越大越看多）；label 故意与 key 不同，防止张冠李戴。"""
    sign = -1.0 if reverse else 1.0

    def compute(panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(
            {name: float(index) * sign for index, name in enumerate(panel.symbols)},
            index=panel.index,
        )

    return FactorSpec(
        key=key,
        label=f"标签-{key}",
        family="测试",
        direction=1,
        warmup=0,
        compute=compute,
    )


def test_correlation_matrix_is_indexed_by_key(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """相关矩阵必须按 key 索引（label 会随文案变，按它索引会 KeyError）。"""
    panel = _ranked_panel(panel_factory)
    specs = [_rank_spec("a"), _rank_spec("b", reverse=True)]
    matrix = factor_correlation(specs, panel, panel.close.notna())

    assert list(matrix.index) == ["a", "b"]
    assert matrix.loc["a", "b"] == pytest.approx(-1.0)


def test_correlation_of_identical_and_reversed_factors(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """同一条信号的两个副本相关为 1，反向信号为 -1，自身对角恒为 1。"""
    panel = _ranked_panel(panel_factory)
    specs = [
        _rank_spec("a"),
        _rank_spec("a_copy"),
        _rank_spec("reversed", reverse=True),
    ]
    matrix = factor_correlation(specs, panel, panel.close.notna())

    assert matrix.loc["a", "a"] == pytest.approx(1.0)
    assert matrix.loc["a", "a_copy"] == pytest.approx(1.0)
    assert matrix.loc["a", "reversed"] == pytest.approx(-1.0)
    assert matrix.loc["reversed", "a"] == pytest.approx(-1.0)  # 对称


def _constant_notional_panel(periods: int = 30, notional: float = 1e6) -> Panel:
    """名义额恒定的面板：容量公式可以手算。"""
    index = pd.date_range("2020-01-01", periods=periods, freq="1D", tz="UTC")
    close = pd.DataFrame({f"N{i:02d}-USDT": 100.0 for i in range(NAMES)}, index=index)
    frame = pd.DataFrame(notional, index=index, columns=close.columns)
    return Panel(timeframe="1d", close=close, volume=frame / 100.0, notional=frame)


def test_capacity_scales_with_notional_and_participation() -> None:
    """容量 = 2 × 每腿标的数 × 参与率 × 最小中位名义额：手算值钉住。"""
    panel = _constant_notional_panel()
    capacity = capacity_usd(_rank_spec("rank"), panel, panel.close.notna(), quantiles=4)

    assert capacity == pytest.approx(1e6 * 0.01 * (NAMES // 4) * 2)


def test_capacity_is_limited_by_illiquid_name() -> None:
    """持有集合里流动性最差的那只决定容量：它只有 1 万美元/日。"""
    panel = _constant_notional_panel()
    thin = panel.notional.copy()
    thin["N00-USDT"] = 1e4  # 因子值最低的标的，落在空头腿
    illiquid = Panel(
        timeframe="1d", close=panel.close, volume=panel.volume, notional=thin
    )

    capacity = capacity_usd(
        _rank_spec("rank"), illiquid, panel.close.notna(), quantiles=4
    )
    assert capacity == pytest.approx(1e4 * 0.01 * (NAMES // 4) * 2)


def test_capacity_uses_median_notional_not_single_day() -> None:
    """单个低成交日不能把容量打下来：口径必须与池子规则同一把尺子（滚动中位）。"""
    panel = _constant_notional_panel()
    spiky = panel.notional.copy()
    spiky.iloc[-1, spiky.columns.get_loc("N00-USDT")] = 1e3  # 最后一天只成交 1000 美元
    with_spike = Panel(
        timeframe="1d", close=panel.close, volume=panel.volume, notional=spiky
    )

    capacity = capacity_usd(
        _rank_spec("rank"), with_spike, panel.close.notna(), quantiles=4
    )
    assert capacity == pytest.approx(1e6 * 0.01 * (NAMES // 4) * 2)


def test_capacity_needs_full_liquidity_window(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """流动性窗口不满时给 0：宁可说不可估，也不要拿半截窗口估容量。"""
    short = _ranked_panel(panel_factory)  # 只有 6 根，窗口（30）不满

    assert (
        capacity_usd(_rank_spec("rank"), short, short.close.notna(), quantiles=4) == 0.0
    )


def test_survivorship_probe_flags_young_underperformance(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """上市初期下跌、成熟期上涨时，代理指标必须给出负缺口。"""
    young_bars, mature_bars = 180, 20
    prices = [100.0 * 0.99**bar for bar in range(young_bars)]
    last = prices[-1]
    prices += [last * 1.01**bar for bar in range(1, mature_bars + 1)]
    panel = panel_factory({"A-USDT": prices})

    probe = survivorship_probe(panel, UniverseRule(min_listed_days=180), horizon=1)

    assert probe.gap < 0  # 年轻观测更差 → 死亡集中在生命周期早期
    assert probe.young_count > 0
    assert probe.mature_count > 0
    assert probe.young_mean < 0 < probe.mature_mean


def test_survivorship_probe_needs_liquid_observations(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """流动性不达标的观测不参与代理统计（它们本就不在池子里）。"""
    panel = panel_factory({"A-USDT": [100.0 + bar for bar in range(200)]})
    tiny = panel.notional * 0 + 1.0  # 名义额 1 美元，远低于门槛
    illiquid = Panel(timeframe="1d", close=panel.close, volume=tiny, notional=tiny)

    probe = survivorship_probe(illiquid, UniverseRule(min_listed_days=180), horizon=1)
    assert probe.young_count == 0
    assert probe.mature_count == 0


def test_ranked_panel_fixture_is_deterministic(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """合成行情必须可复现：两次构造的结果逐点相等。"""
    first = _ranked_panel(panel_factory)
    second = _ranked_panel(panel_factory)
    pd.testing.assert_frame_equal(first.close, second.close)
    assert np.isclose(first.notional.iloc[0, 0], second.notional.iloc[0, 0])
