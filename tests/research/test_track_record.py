"""第 14 章记录质量工具的用例：HHI 的两个端点、回撤的精确值、PSR/MinTRL 的往返。

这些断言的价值在于「能被算错」：集中度写成平方份额、回撤把末段丢掉、MinTRL 忽略
非正态项，都会让某条断言当场变红。
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import sharpe_ratio
from src.research.study.track_record import (
    concentration,
    concentration_by_sign,
    drawdown_and_time_under_water,
    max_drawdown,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_z_score,
    track_record,
)
from src.research.validation.overfitting import deflated_sharpe


def test_concentration_is_zero_when_every_bet_is_equal() -> None:
    """均匀下注 ⇒ 0（原文「positive_concentration == 0 ⇔ returns are uniform」）。"""
    assert concentration([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) == pytest.approx(
        0.0, abs=1e-12
    )


def test_concentration_is_one_when_only_one_bet_is_nonzero() -> None:
    """只有一笔非零 ⇒ 1。"""
    assert concentration([0.0, 0.0, 5.0]) == pytest.approx(1.0)


def test_concentration_needs_at_least_three_bets() -> None:
    """少于三笔给 nan，而不是 0——「算不出」与「很均匀」不是一回事。"""
    assert math.isnan(concentration([1.0, 2.0]))


def test_concentration_is_scale_invariant() -> None:
    """整体放大 1000 倍不改变集中度（权重按 Σr 归一）。"""
    returns = [0.5, 0.1, 0.1, 0.1, 0.2]
    assert concentration(returns) == pytest.approx(
        concentration(np.array(returns) * 1000)
    )


def test_concentration_rejects_mixed_signs() -> None:
    """混号收益会让权重超 1、结果出 3.755 这种假数，必须显式拒绝而不是给个数。"""
    with pytest.raises(ValueError, match="同号"):
        concentration([0.5, -0.3, 0.2, -0.1, 0.05])


def test_concentration_by_sign_handles_mixed_input() -> None:
    """混号收益的正确入口：分两支后各自落在 [0, 1]。"""
    positive, negative = concentration_by_sign([0.5, -0.3, 0.2, -0.1, 0.05, -0.02])
    assert 0.0 <= positive <= 1.0
    assert 0.0 <= negative <= 1.0


def test_concentration_by_sign_returns_nan_for_a_thin_side() -> None:
    """某一侧不足三笔时给 nan（而不是把两侧混在一起算）。"""
    positive, negative = concentration_by_sign([0.5, 0.3, 0.2, -0.1, -0.02])
    assert 0.0 <= positive <= 1.0
    assert math.isnan(negative)


def test_concentration_does_not_care_about_the_sign() -> None:
    """全负的收益与全正的收益给出同一个数（权重按 Σr 归一后都为正）。"""
    assert concentration([-0.5, -0.1, -0.1, -0.3]) == pytest.approx(
        concentration([0.5, 0.1, 0.1, 0.3])
    )


def test_concentration_separates_the_two_signs() -> None:
    """一笔暴亏 + 多笔小赢：负侧集中度远高于正侧（原文分两支算的用处）。

    手算：负侧 [−0.9, −0.05, −0.05] → HHI 0.815 → 归一化 0.7225；
    正侧 [0.05, 0.02, 0.01, 0.01] → HHI 0.3827 → 归一化 0.1769。
    """
    positive, negative = concentration_by_sign(
        [0.05, 0.02, 0.01, 0.01, -0.9, -0.05, -0.05]
    )
    assert negative == pytest.approx(0.7225, abs=1e-4)
    assert positive == pytest.approx(0.1769, abs=1e-4)


def test_max_drawdown_is_zero_on_a_monotone_level() -> None:
    """一路新高没有回撤。"""
    assert max_drawdown(pd.Series([0.0, 1.0, 2.0, 3.0])) == 0.0


def test_drawdown_and_time_under_water_match_a_hand_computed_dip() -> None:
    """净值 [0,1,2,1,0,3]：2 → 0 的回撤是 2，水下 3 根（到收复它的新高）。"""
    depths, spans = drawdown_and_time_under_water(
        pd.Series([0.0, 1.0, 2.0, 1.0, 0.0, 3.0]), periods_per_year=1.0
    )
    assert depths.tolist() == [2.0]
    assert spans.tolist() == [3.0]


def test_unrecovered_drawdown_is_measured_to_the_end() -> None:
    """还没收复的高水位算到观测结束（原文明确加这一段）。"""
    depths, spans = drawdown_and_time_under_water(
        pd.Series([0.0, 2.0, 1.0, 1.5]), periods_per_year=1.0
    )
    assert depths.tolist() == [1.0]
    assert spans.tolist() == [3.0]


def test_zero_sharpe_returns_half_probability() -> None:
    """对称的零均值收益：SR = 0 ⇒ PSR = 0.5（偏度 0、峰度恰好让分母为 1）。"""
    returns = np.array([1.0, -1.0] * 20)
    assert probabilistic_sharpe_ratio(returns) == pytest.approx(0.5)


def test_psr_rises_with_the_same_sharpe_over_more_observations() -> None:
    """同样的逐期夏普，记录越长越显著——这正是 MinTRL 要回答的问题。"""
    pattern = np.concatenate([np.full(20, 0.01), np.full(20, -0.005)])
    short = probabilistic_sharpe_ratio(pattern)
    long = probabilistic_sharpe_ratio(np.tile(pattern, 4))
    assert long > short > 0.5


def test_psr_reports_nan_when_observations_are_too_few() -> None:
    assert math.isnan(probabilistic_sharpe_ratio([0.1, -0.05] * 5))


def test_deflated_sharpe_and_psr_share_the_same_formula() -> None:
    """DSR 就是「基准换成零假设期望最大夏普」的 PSR——两个模块必须给出同一个数。"""
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.001, 0.02, 300))
    test = deflated_sharpe("x", returns, expected_max=0.05)
    assert test.dsr == pytest.approx(
        probabilistic_sharpe_ratio(returns, benchmark=0.05)
    )


def test_minimum_track_record_is_infinite_below_the_benchmark() -> None:
    """夏普没超过基准时不是「需要更久」，而是「多久都证明不了」。"""
    rng = np.random.default_rng(5)
    returns = rng.normal(-0.002, 0.02, 500)
    assert math.isinf(minimum_track_record_length(returns, benchmark=0.0))


def test_minimum_track_record_is_the_exact_inverse_of_the_psr_formula() -> None:
    """MinTRL 是 PSR 公式对观测数的**精确反解**：到这个长度时检验量恰好跨过置信分位数。

    刻意用右偏、肥尾样本，并按「固定矩」口径比较——截断后重估矩会让均值动 1σ，
    而 MinTRL 与夏普平方成反比，那种比法出的是噪声而不是不变量。
    """
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0015, 0.02, 4000) + 0.02 * (rng.exponential(1.0, 4000) - 1.0)
    series = pd.Series(returns)
    observed = sharpe_ratio(returns)
    skew, kurtosis = float(series.skew()), float(series.kurt()) + 3.0
    needed = minimum_track_record_length(returns, confidence=0.95)
    assert needed > 100.0  # 否则这条测试落在「样本不足」区间里，断言就失去意义

    def test_value(at: int) -> float:
        value = sharpe_z_score(observed, at, skew, kurtosis, 0.0)
        assert value is not None
        return value

    quantile = NormalDist().inv_cdf(0.95)
    assert test_value(math.floor(needed)) < quantile <= test_value(math.ceil(needed))
    # 非正态项必须真的在起作用：MinTRL = 1 + 系数 × (忽略该系数时的值 − 1)
    naive = 1.0 + (quantile / observed) ** 2
    coefficient = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    assert needed == pytest.approx(1.0 + coefficient * (naive - 1.0), rel=1e-12)


def test_minimum_track_record_grows_with_confidence() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.002, 0.02, 400)
    assert minimum_track_record_length(returns, confidence=0.99) > (
        minimum_track_record_length(returns, confidence=0.90)
    )


def test_track_record_bundles_the_chapter_fourteen_checks() -> None:
    """报告用的一次算齐：字段齐、样本不足会明说、说明里带 PSR 与集中度。"""
    rng = np.random.default_rng(13)
    returns = pd.Series(rng.normal(0.002, 0.02, 300))
    record = track_record("演示策略", returns)
    assert record.observations == 300
    assert record.sufficient
    assert 0.0 < record.psr <= 1.0
    assert record.max_drawdown > 0.0
    assert 0.0 <= record.positive_concentration <= 1.0
    assert 0.0 <= record.negative_concentration <= 1.0
    assert "PSR" in record.describe() and "集中度" in record.describe()
    thin = track_record("薄记录", returns.iloc[:10])
    assert not thin.sufficient and "样本不足" in thin.describe()
