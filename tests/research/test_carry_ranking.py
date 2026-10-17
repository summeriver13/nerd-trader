"""carry_ranking 的横截面口径：成本摊销、组合成员门槛、分段边界。"""

import numpy as np
import pandas as pd

from src.research.funding.carry import cost_drag
from src.research.funding.carry_ranking import (
    MIN_PORTFOLIO_MEMBERS,
    net_recent_daily,
    portfolio_daily,
    segment_rows,
)


def _daily(values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_net_recent_daily_respects_window_and_amortizes_cost() -> None:
    """近期窗口只取最后 N 天，且把该窗口的持有成本按日摊掉。"""
    daily = _daily([0.001] * 400)
    recent = net_recent_daily(daily, 365)
    assert len(recent) == 365
    amortized, rebalance = cost_drag(365 / 365)
    expected = 0.001 - (amortized + rebalance) / 365
    assert np.allclose(recent.to_numpy(), expected)


def test_net_recent_daily_drops_missing_days() -> None:
    """缺日（NaN）不进序列，否则会被当成 0 费率拉低均值。"""
    daily = _daily([0.001, np.nan, 0.002])
    assert len(net_recent_daily(daily, 3)) == 2


def test_portfolio_requires_minimum_members() -> None:
    """成分数不足的日子剔除，避免早期只有一两只标的时组合被单标的代表。"""
    index = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    wide = pd.DataFrame(
        {
            "A": [0.001] * 10,
            "B": [0.001] * 10,
            "C": [np.nan] * 5 + [0.001] * 5,
            "D": [np.nan] * 5 + [0.001] * 5,
            "E": [np.nan] * 5 + [0.001] * 5,
            "F": [np.nan] * 5 + [0.001] * 5,
            "G": [np.nan] * 5 + [0.001] * 5,
        },
        index=index,
    )
    portfolio = portfolio_daily(wide)
    assert len(portfolio) == 5  # 前 5 天只有 2 只成员，被剔除
    assert MIN_PORTFOLIO_MEMBERS == 5


def test_segment_rows_split_by_calendar_and_count_positive() -> None:
    """分段按日历等分；positive 统计段内日均费率为正的标的数。"""
    index = pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC")
    wide = pd.DataFrame(
        {
            "A": [0.001] * 40,
            "B": [-0.001] * 40,
            "C": [0.001] * 20 + [-0.001] * 20,
            "D": [0.001] * 40,
            "E": [0.001] * 40,
        },
        index=index,
    )
    rows = segment_rows(wide, count=2)
    assert len(rows) == 2
    assert sum(row.days for row in rows) == len(portfolio_daily(wide))
    assert rows[0].positive == 4  # 前半段 C 还是正的
    assert rows[1].positive == 3  # 后半段 C 转负
