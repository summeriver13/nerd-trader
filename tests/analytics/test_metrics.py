"""指标口径的黄金样本：手算值锚定，防止有人顺手改公式。

这些数字是整个项目产出（收益率/回撤/夏普）的唯一来源，
改口径必须同时改这里，改不动就说明口径变了。
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    annualize,
    annualized_sharpe,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    max_drawdown_from_equity,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def test_max_drawdown_golden_sample() -> None:
    """手算：净值 1 → 1.2 → 0.9 → 1.05，最大回撤 = 0.9/1.2 - 1 = -25%"""
    net_ret = pd.Series([0.2, -0.25, 0.16666666666666666])
    assert max_drawdown(net_ret) == pytest.approx(-0.25)

    equity = pd.Series([1.0, 1.2, 0.9, 1.05])
    assert max_drawdown_from_equity(equity) == pytest.approx(-0.25)
    assert drawdown_series(equity).min() == pytest.approx(-0.25)


def test_drawdown_starts_at_zero_and_never_positive() -> None:
    """单调上涨时回撤恒为 0，且序列没有正值。"""
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown_from_equity(equity) == pytest.approx(0.0)
    assert drawdown_series(equity).max() == pytest.approx(0.0)


def test_total_return_compounds() -> None:
    """手算：(1.1)(0.9) - 1 = -1%"""
    assert total_return(pd.Series([0.1, -0.1])) == pytest.approx(-0.01)


def test_annualize_known_values() -> None:
    """一年 365 天翻倍 → 年化 100%；两年翻倍 → 41.42%。"""
    assert annualize(1.0, 365) == pytest.approx(1.0)
    assert annualize(1.0, 730) == pytest.approx(2**0.5 - 1)
    assert annualize(-1.0, 365) == pytest.approx(-1.0)


def test_sharpe_matches_manual_formula() -> None:
    """逐根夏普 = 均值 / 样本标准差（ddof=1），年化版再乘 √每年根数。"""
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    manual = float(returns.mean() / returns.std(ddof=1))
    assert sharpe_ratio(returns) == pytest.approx(manual)
    assert annualized_sharpe(returns, 365) == pytest.approx(manual * np.sqrt(365))


def test_sortino_and_calmar() -> None:
    """索提诺下行偏差取 min(r,0) 的均方根；卡尔玛 = 年化 / |回撤|。"""
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    downside = float(np.sqrt((np.minimum(returns.to_numpy(), 0.0) ** 2).mean()))
    assert sortino_ratio(returns, 365) == pytest.approx(
        float(returns.mean()) / downside * np.sqrt(365)
    )
    assert calmar_ratio(0.30, -0.15) == pytest.approx(2.0)
    assert calmar_ratio(0.30, 0.0) == 0.0
