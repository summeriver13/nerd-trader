"""carry_risk 的口径：强平线公式、信号防前视、切换成本与最坏涨幅窗口。"""

import numpy as np
import pandas as pd

from src.research.funding.carry import CAPITAL_MULTIPLE, ROUND_TRIP_COST
from src.research.funding.carry_risk import (
    MAINTENANCE_MARGIN,
    PERP_MARGIN_RATIO,
    apply_rule,
    liquidation_gain,
    trailing_signal,
)


def _daily(values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_liquidation_gain_matches_hand_calculation() -> None:
    """2x 杠杆（保证金率 0.5）配 0.5% 维持保证金 → 涨约 49.25% 强平。"""
    gain = liquidation_gain()
    expected = (PERP_MARGIN_RATIO - MAINTENANCE_MARGIN) / (1 + MAINTENANCE_MARGIN)
    assert np.isclose(gain, expected)
    assert 0.49 < gain < 0.50


def test_trailing_signal_has_no_lookahead() -> None:
    """信号只看已完成的历史：第一天必然不在场，哪怕当天费率为正。"""
    daily = _daily([0.01, 0.01, 0.01])
    signal = trailing_signal(daily, 1)
    assert not bool(signal.iloc[0])
    assert bool(signal.iloc[1])


def test_trailing_signal_stays_out_when_trailing_is_negative() -> None:
    """过去窗口累计为负就离场，转正后才回来。"""
    daily = _daily([0.01, 0.01, -0.03, -0.03, 0.02, 0.02, 0.02])
    # 窗口 2 的累计：-0.02, -0.06, -0.01, 0.04, 0.06 → 移一位后为信号
    signal = trailing_signal(daily, 2)
    assert bool(signal.iloc[2])  # 前两日 +0.02
    assert not bool(signal.iloc[3])  # 前两日 -0.02
    assert not bool(signal.iloc[5])  # 前两日 -0.01
    assert bool(signal.iloc[6])  # 前两日 +0.04


def test_apply_rule_always_pays_one_round_trip() -> None:
    """一直持有恰好两次仓位变化（建仓 + 平仓）= 一次完整往返。"""
    daily = _daily([0.001] * 10)
    always = pd.Series(True, index=daily.index)
    result = apply_rule(daily, always, "always")
    assert result.changes == 2
    assert np.isclose(result.cost, ROUND_TRIP_COST)
    assert np.isclose(
        result.total_return,
        ((1.001**10 - 1) - ROUND_TRIP_COST) / CAPITAL_MULTIPLE,
    )


def test_apply_rule_charges_each_switch() -> None:
    """两次进出共四次仓位变化，成本是基准的两倍。"""
    daily = _daily([0.001] * 10)
    signal = pd.Series(
        [True, True, False, False, True, True, False, False, False, False],
        index=daily.index,
    )
    result = apply_rule(daily, signal, "twice")
    assert result.changes == 4
    assert np.isclose(result.cost, ROUND_TRIP_COST * 2)
    assert result.days_in_market == 4
