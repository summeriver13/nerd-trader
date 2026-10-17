"""
tests/research/test_leverage_cap.py
容量口径（杠杆额度）的合成数据测试：数值断言都建立在可手算的面板上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.backtest.leverage_cap import (
    CapacitySpec,
    average_dollar_volume,
    capacity_frame,
    dollar_volume,
    run_capacity,
)
from src.research.trend.tsmom import TsmomOptions


def make_bars(closes: np.ndarray, volume: float = 1.0) -> pd.DataFrame:
    """构造日线：high/low 由 close 派生，volume 恒定（成交额 = close × volume）。"""
    index = pd.date_range("2024-01-01", periods=closes.size, freq="D", tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    volume_series = pd.Series(np.full(closes.size, volume), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume_series,
        }
    )


def test_dollar_volume_is_close_times_volume() -> None:
    """成交额就是收盘价 × 成交量的乘积。"""
    bars = make_bars(np.array([10.0, 20.0, 30.0]), volume=5.0)
    assert dollar_volume(bars).tolist() == pytest.approx([50.0, 100.0, 150.0])


def test_average_dollar_volume_uses_only_past_readings() -> None:
    """滞后一根：当根的成交额尖峰不出现在当根的 ADV 里。"""
    closes = np.array([10.0, 10.0, 10.0, 100.0, 10.0, 10.0])
    adv = average_dollar_volume(make_bars(closes), window=2, lag=1)
    assert np.isnan(adv.iloc[0]) and np.isnan(adv.iloc[1])  # 暖机期无读数
    assert adv.iloc[2] == pytest.approx(10.0)  # 用 t=0,1
    assert adv.iloc[3] == pytest.approx(10.0)  # 用 t=1,2：尖峰尚未进窗口
    assert adv.iloc[4] == pytest.approx(55.0)  # 用 t=2,3：尖峰从 t=3 起才被看到


def test_capacity_multiplier_follows_the_participation_formula() -> None:
    """倍率 = 参与率 × ADV × N_t / M：两只标的、成交额 10、账本 100 → 0.01。"""
    bars = make_bars(np.full(4, 10.0))
    bars_map = {"a": bars, "b": bars}
    counts = pd.Series(2.0, index=bars.index)
    book = pd.Series(1.0, index=bars.index)
    spec = CapacitySpec(notional=100.0, participation=0.05, window=2, lag=1)
    capacity = capacity_frame(bars_map, counts, book, spec)
    assert capacity.mult["a"].tolist() == pytest.approx([0.0, 0.0, 0.01, 0.01])
    assert capacity.mean_multiplier == pytest.approx(0.01)
    assert capacity.constrained_share == pytest.approx(1.0)  # 全部有读数的格子都被压住


def test_capacity_multiplier_never_exceeds_book_or_cap() -> None:
    """小账本下允许倍率很大：最终倍率被账本倍率（1.0）与上限（3 倍）夹住。"""
    bars = make_bars(np.full(4, 10.0))
    bars_map = {"a": bars}
    counts = pd.Series(1.0, index=bars.index)
    book = pd.Series(1.0, index=bars.index)
    small = CapacitySpec(notional=0.001, participation=0.05, window=2, lag=1)
    capacity = capacity_frame(bars_map, counts, book, small)
    assert capacity.mult["a"].max() == pytest.approx(1.0)
    assert capacity.constrained_share == pytest.approx(0.0)
    loose = CapacitySpec(notional=0.001, participation=0.05, window=2, lag=1, cap=0.25)
    assert capacity_frame(bars_map, counts, book, loose).mult[
        "a"
    ].max() == pytest.approx(0.25)


def test_missing_volume_leaves_the_symbol_without_quota() -> None:
    """成交额为 0 的标的允许倍率为 0：容量口径下它根本进不了仓。"""
    bars = make_bars(np.full(4, 10.0), volume=0.0)
    counts = pd.Series(1.0, index=bars.index)
    book = pd.Series(1.0, index=bars.index)
    spec = CapacitySpec(notional=100.0, participation=0.05, window=2, lag=1)
    capacity = capacity_frame({"a": bars}, counts, book, spec)
    assert capacity.mult["a"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_run_capacity_scales_positions_by_the_multiplier() -> None:
    """引擎仓位 = 信号 × 倍率（再滞后一根），故倍率越小仓位越小。"""
    # 漂移 2% 必须大于 make_bars 里 high 的 1% 上影，否则永不突破（信号恒 0）
    closes = 100.0 * np.cumprod(np.full(30, 1.02))
    bars = make_bars(closes)
    bars_map = {"a": bars}
    counts = pd.Series(1.0, index=bars.index)
    book = pd.Series(1.0, index=bars.index)
    spec = CapacitySpec(notional=1e4, participation=0.05, window=2, lag=1)
    capacity = capacity_frame(bars_map, counts, book, spec)
    runs = run_capacity(bars_map, TsmomOptions(entry_periods=(3,)), capacity)
    pos = runs["a"].pos
    mult = capacity.mult["a"]
    assert pos.iloc[2] == pytest.approx(0.0)  # 暖机期无候选仓位
    for index in range(6, len(bars)):
        # 单调上行下唐奇安信号为 1，故仓位应等于滞后一根的倍率
        assert pos.iloc[index] == pytest.approx(mult.iloc[index - 1])
