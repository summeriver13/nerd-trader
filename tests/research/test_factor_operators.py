"""
Alpha101 算子与派生因子的不变量：手算对照、预热下标、只吃历史。

历史教训同 test_factor_library.py：混入未来信息的变换不会报错，只会让 IC 好看得
不像话，唯一可靠的防线是「篡改未来、过去必须不变」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.operators import correlation, decay_linear, ts_rank
from src.research.factors.panel import Panel
from src.research.factors.variants import (
    resolve_interaction,
    resolve_smoothing,
    smoothing_spec,
)

BARS = 120
DECAY_PROBE = 5


def _panel(values: list[float], volumes: list[float] | None = None) -> Panel:
    """合成面板：成交量可给（量价相关需要非恒定序列，否则方差为 0）。"""
    index = pd.date_range("2020-01-01", periods=len(values), freq="1D", tz="UTC")
    close = pd.DataFrame({"A-USDT": values}, index=index)
    volume = pd.DataFrame(
        {"A-USDT": values if volumes is None else volumes}, index=index
    )
    return Panel(timeframe="1d", close=close, volume=volume, notional=close * volume)


def test_decay_linear_matches_hand_calculation() -> None:
    """线性衰减：最近一根权重 window、最远一根权重 1，分母是权重和。"""
    frame = pd.DataFrame({"A-USDT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = decay_linear(frame, DECAY_PROBE)["A-USDT"]

    assert out.iloc[: DECAY_PROBE - 1].isna().all()
    # 首个满窗落在下标 4：窗内 [1,2,3,4,5]，最近一根权重 5、最远一根权重 1
    weights = np.arange(DECAY_PROBE, 0, -1, dtype=float)
    expected = float(np.dot(weights, [5.0, 4.0, 3.0, 2.0, 1.0]) / weights.sum())
    assert out.iloc[DECAY_PROBE - 1] == pytest.approx(expected)


def test_decay_linear_flat_series_is_flat() -> None:
    """常数序列经过平滑仍是同一常数：平滑不能凭空造出振幅。"""
    out = decay_linear(pd.DataFrame({"A": [7.0] * 10}), 5)["A"].dropna()
    assert np.allclose(out.to_numpy(), 7.0)


def test_ts_rank_returns_within_window_percentile() -> None:
    """上升序列的 ts_rank 恒为 1（每根都是自身近期最高）。"""
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = ts_rank(frame, 3)["A"]
    assert out.iloc[:2].isna().all()
    assert np.allclose(out.dropna().to_numpy(), 1.0)

    drop = pd.DataFrame({"A": [5.0, 4.0, 3.0, 2.0]})
    # 百分位排名 = rank / n（最小值 1/3，不是 0）
    assert ts_rank(drop, 3)["A"].iloc[-1] == pytest.approx(1 / 3)


def test_correlation_with_itself_is_one() -> None:
    """序列与自身的滚动相关恒为 1；与常数序列相关为 NaN（方差为 0）。"""
    values = pd.DataFrame({"A": [1.0, 3.0, 2.0, 5.0, 4.0, 7.0]})
    same = correlation(values, values, 3)["A"].dropna()
    assert np.allclose(same.to_numpy(), 1.0)

    constant = pd.DataFrame({"A": [2.0] * 6})
    assert bool(correlation(values, constant, 3)["A"].isna().all())


def test_operator_rejects_too_small_window() -> None:
    """窗口 <2 当场报错：窗口 1 的「平滑」是恒等变换，属调用方写错。"""
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    for operator in (decay_linear, ts_rank):
        with pytest.raises(ValueError, match="窗口至少"):
            operator(frame, 1)
    with pytest.raises(ValueError, match="窗口至少"):
        correlation(frame, frame, 1)


def test_variant_warmup_is_base_plus_window_minus_one() -> None:
    """派生因子的预热 = 基因子预热 + 窗口 − 1（前 window−1 根必为 NaN）。"""
    for key, spec in FACTOR_LIBRARY.items():
        panel = _panel(list(np.linspace(100.0, 300.0, BARS)))
        for operator in ("decay5", "tsrank20"):
            derived = smoothing_spec(spec, operator)
            assert derived.key == f"{key}@{operator}"
            assert derived.direction == spec.direction
            values = derived.values(panel)["A-USDT"]
            assert values.iloc[: derived.warmup].isna().all()
            assert values.iloc[derived.warmup :].notna().all()


def test_variants_read_only_the_past() -> None:
    """篡改第 60 根之后的价格，前 60 根的派生因子值必须一模一样。"""
    base = [100.0 + step for step in range(BARS)]
    cut = 60
    intact = _panel(base)
    tampered = _panel(base[:cut] + [value * 3 for value in base[cut:]])

    for spec in resolve_smoothing() + resolve_interaction():
        left = spec.values(intact).iloc[:cut]
        right = spec.values(tampered).iloc[:cut]
        pd.testing.assert_frame_equal(left, right)


def test_interaction_spec_lookahead_free_and_warmup() -> None:
    """交互因子：warmup = 窗口 − 1，且与成交量相关必须在满窗后才出数。"""
    values = [float(item) for item in np.linspace(100.0, 260.0, BARS)]
    volumes = [float(item) for item in 1e6 + 1e5 * np.sin(np.arange(BARS) / 3.0)]
    spec = next(item for item in resolve_interaction() if item.key == "corr_pv_10")
    series = spec.values(_panel(values, volumes))["A-USDT"]

    assert series.iloc[: spec.warmup].isna().all()
    assert series.iloc[spec.warmup :].notna().all()
    # 方向已并入：量价同向（价升量升那段）时因子值应为负
    assert series.iloc[spec.warmup] < 0
