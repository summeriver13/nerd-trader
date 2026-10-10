"""波动率目标：测试只覆盖会静默出错的地方。

三个高风险点（错了都不会报错，只会让结论变好看）：

1. **等暴露对照的暴露反解**——反解错了，整份研究测的就不是同一个平均暴露，
   而对照看起来仍然「正常」，没有对照的直觉会直接把偏差读成效应。
2. **暴露路径的前视**——第 t 根的暴露只能用第 t 根及以前的价格。引擎虽然会把
   signal 右移一根，但暴露本身若用了未来数据，右移也救不回来。
3. **块自举的下标**——块自举的意义是保留自相关；退化成独立同分布抽样会把置信
   区间算窄，从而把噪声读成显著。
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.research.vol_target import (
    VolTargetSpec,
    build_grid,
    matched_control,
    run_constant,
    run_spec,
)
from src.research.vol_target_report import _block_sample, _segment_edges

BARS = 400
SEED = 7
# 前段低波动、后段高波动：让暴露路径真的在动，否则前视测试无意义
CALM_VOL = 0.01
WILD_VOL = 0.04


@pytest.fixture
def bars() -> pd.DataFrame:
    """合成行情：固定种子、波动率两段式，可复现且暴露路径有变化。"""
    rng = np.random.default_rng(SEED)
    half = BARS // 2
    scale = np.concatenate([np.full(half, CALM_VOL), np.full(BARS - half, WILD_VOL)])
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, scale, size=BARS))
    index = pd.date_range("2020-01-01", periods=BARS, freq="1D", tz="UTC")
    return pd.DataFrame({"close": close}, index=index)


@pytest.fixture
def spec() -> VolTargetSpec:
    """出厂默认参数（目标 40%、回看 30 根、每周调仓、只降杠杆）。"""
    return VolTargetSpec(0.40, 30, 7, 1.0)


def test_control_matches_average_exposure(
    bars: pd.DataFrame, spec: VolTargetSpec
) -> None:
    """等暴露对照必须真的跑出同一个平均暴露（端到端验证反解，而非重算公式）。"""
    run = run_spec(bars, spec)
    control = matched_control(bars, run)
    assert control.avg_exposure == pytest.approx(run.avg_exposure, abs=1e-9)
    # 对照只在首根建仓一次：总换手 = 仓位水平，而平均暴露 = 仓位水平 × (n−1)/n
    # 两条都是引擎算出来的，相等就证明持仓路径确实只有一笔建仓
    bars_count = bars.index.size
    assert control.turnover * bars_count == pytest.approx(
        control.avg_exposure * bars_count / (bars_count - 1), rel=1e-9
    )


def test_exposure_ignores_future_prices(
    bars: pd.DataFrame, spec: VolTargetSpec
) -> None:
    """篡改最后一根的价格，前面所有根的暴露必须一字不变。"""
    close = cast(pd.Series, bars["close"])
    base = spec.sizing().exposure(close)
    tampered = close.copy()
    tampered.iloc[-1] *= 10.0
    after = spec.sizing().exposure(tampered)
    assert np.array_equal(after.iloc[:-1].to_numpy(), base.iloc[:-1].to_numpy())


def test_constant_exposure_keeps_sharpe(bars: pd.DataFrame) -> None:
    """恒定缩放不改变夏普——这是「等暴露对照」成立的理论依据。

    对照若连这条都不满足，说明成本或口径没跟着仓位一起缩放。
    """
    full = run_constant(bars, 1.0, "full")
    half = run_constant(bars, 0.5, "half")
    assert half.stats.sharpe == pytest.approx(full.stats.sharpe, rel=1e-9)


def test_grid_is_complete_and_unique() -> None:
    """网格漏组合会低估试验次数（门槛放宽），重组合会高估。"""
    specs = build_grid()
    assert len(specs) == 5 * 5 * 3 * 3
    assert len({item.label() for item in specs}) == len(specs)


def test_segment_edges_tile_the_sample() -> None:
    """分段必须首尾相接、无重叠、无遗漏，否则分段统计会漏掉或重复算某段。"""
    edges = _segment_edges(1000, 4)
    assert edges[0][0] == 0
    assert edges[-1][1] == 1000
    assert all(
        left[1] == right[0] for left, right in zip(edges, edges[1:], strict=False)
    )
    assert all(stop > start for start, stop in edges)


def test_block_sample_keeps_runs_contiguous() -> None:
    """块自举必须抽连续段：相邻下标绝大多数应满足「前一根 +1」。"""
    index = _block_sample(1000, 50, np.random.default_rng(0))
    assert index.size == 1000
    assert index.min() >= 0
    assert index.max() < 1000
    # 20 块 × 每块 49 次块内连续 = 980，总步数 999，故下界取 0.95 留足余量
    assert float(((np.diff(index) % 1000) == 1).mean()) >= 0.95
