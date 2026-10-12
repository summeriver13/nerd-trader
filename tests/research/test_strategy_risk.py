"""第 15 章策略风险的用例：binSR / binFreq 的往返、单调性、失败概率的方向。

关键断言是「往返」与「方向」：二次方程解错根、增根不检验、精度与次数弄反，
都会让某条断言当场变红。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.research.strategy_risk import (
    required_bets_per_year,
    required_precision,
    strategy_failure_probability,
)

STOP_LOSS = -0.05
PROFIT_TARGET = 0.05
FREQUENCY = 260.0
TARGET_SHARPE = 2.0


def test_required_precision_matches_the_book_example() -> None:
    """原文参数（sl=−5%、pt=+5%、260 次/年、目标夏普 2）要求约 56% 的胜率。"""
    assert required_precision(STOP_LOSS, PROFIT_TARGET, FREQUENCY, TARGET_SHARPE) == (
        pytest.approx(0.5615, abs=1e-3)
    )


def test_required_bets_round_trips_through_required_precision() -> None:
    """原文自带的检查：解出的频率必须能反解回同一目标夏普。"""
    precision = required_precision(STOP_LOSS, PROFIT_TARGET, FREQUENCY, TARGET_SHARPE)
    assert required_bets_per_year(
        STOP_LOSS, PROFIT_TARGET, precision, TARGET_SHARPE
    ) == pytest.approx(FREQUENCY, rel=1e-6)


def test_required_precision_rises_with_the_target_sharpe() -> None:
    low = required_precision(STOP_LOSS, PROFIT_TARGET, FREQUENCY, 1.0)
    high = required_precision(STOP_LOSS, PROFIT_TARGET, FREQUENCY, 3.0)
    assert high > low


def test_required_precision_falls_with_more_bets() -> None:
    """广度换精度：下注次数越多，所需胜率越低。"""
    few = required_precision(STOP_LOSS, PROFIT_TARGET, 52.0, TARGET_SHARPE)
    many = required_precision(STOP_LOSS, PROFIT_TARGET, 2600.0, TARGET_SHARPE)
    assert many < few


def test_required_precision_penalises_asymmetric_payoffs() -> None:
    """止损比止盈大（输得多赢得少）时所需胜率更高——这就是「赢少赔多」的代价。"""
    symmetric = required_precision(-0.05, 0.05, FREQUENCY, TARGET_SHARPE)
    worse = required_precision(-0.10, 0.05, FREQUENCY, TARGET_SHARPE)
    assert worse > symmetric


def test_required_bets_returns_none_when_the_root_is_extraneous() -> None:
    """胜率低于盈亏比所允许的水平时，加多少次数都到不了目标夏普。

    代数上仍有一个根，但它反解不回目标夏普，属增根——必须报 None，
    不能报一个看起来像答案的数字。
    """
    assert required_bets_per_year(STOP_LOSS, PROFIT_TARGET, 0.02, 5.0) is None


def test_impossible_inputs_raise() -> None:
    with pytest.raises(ValueError):
        required_precision(0.05, -0.05, FREQUENCY, TARGET_SHARPE)
    with pytest.raises(ValueError):
        required_bets_per_year(STOP_LOSS, PROFIT_TARGET, 1.5, TARGET_SHARPE)


def test_failure_probability_falls_when_the_sample_is_better() -> None:
    """胜率更高的一组下注，失败概率必须更低。"""
    rng = np.random.default_rng(23)
    good = np.where(rng.random(400) < 0.75, 0.05, -0.05)
    bad = np.where(rng.random(400) < 0.45, 0.05, -0.05)
    good_risk = strategy_failure_probability(good, FREQUENCY, TARGET_SHARPE)
    bad_risk = strategy_failure_probability(bad, FREQUENCY, TARGET_SHARPE)
    assert good_risk < bad_risk
    assert 0.0 < good_risk < 1.0


def test_failure_probability_needs_both_signs() -> None:
    """只有一边收益时算不出盈亏比 ⇒ nan，不能当成 0（「不会失败」）或 1。"""
    assert math.isnan(
        strategy_failure_probability([0.05] * 100, FREQUENCY, TARGET_SHARPE)
    )


def test_failure_probability_needs_enough_bets() -> None:
    assert math.isnan(
        strategy_failure_probability([0.05, -0.05] * 5, FREQUENCY, TARGET_SHARPE)
    )
