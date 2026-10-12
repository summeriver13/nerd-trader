"""
tests/research/test_overlap.py
样本重叠的两件工具：唯一性权重与 purge/embargo。

钉的是「非重叠采样下必须是空操作」这条不变量——它同时是「现有 PBO 数字没被
标签重叠污染」的依据，也是将来改用重叠采样时 purge 会真的生效的前提。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.research import overlap


def test_concurrency_ramps_up_when_labels_overlap() -> None:
    """逐根采样 + 3 根标签：并发数 1,2,3 爬升后保持 3。"""
    starts = np.arange(8)
    result = overlap.concurrency(8, starts, horizon=3)
    assert result.tolist() == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


def test_concurrency_is_one_when_labels_abut_without_overlap() -> None:
    """每 h 根取一个观测：标签首尾相接不重叠，并发恒为 1。"""
    starts = np.arange(0, 9, 3)
    result = overlap.concurrency(9, starts, horizon=3)
    assert result.tolist() == [1.0] * 9


def test_average_uniqueness_falls_to_one_over_horizon() -> None:
    """逐根采样时，稳态唯一性趋近 1/h（同一段收益被算了 h 次）。"""
    result = overlap.average_uniqueness(8, np.arange(8), horizon=3)
    # 头三个观测的标签期内并发数逐根爬升 1,2,3，取均值即 (1 + 1/2 + 1/3)/3
    assert result[0] == pytest.approx((1.0 + 0.5 + 1 / 3) / 3, rel=1e-12)
    assert result[1] == pytest.approx((0.5 + 1 / 3 + 1 / 3) / 3, rel=1e-12)
    assert result[3] == pytest.approx(1 / 3, rel=1e-12)
    assert result[7] == pytest.approx(1 / 3, rel=1e-12)


def test_average_uniqueness_is_one_when_not_overlapping() -> None:
    """非重叠采样的唯一性恒为 1——这正是抽稀法「不需要权重」的原因。"""
    result = overlap.average_uniqueness(9, np.arange(0, 9, 3), horizon=3)
    assert result.tolist() == [1.0, 1.0, 1.0]


def test_average_uniqueness_of_nothing_is_empty() -> None:
    assert overlap.average_uniqueness(10, np.zeros(0, dtype=int), horizon=3).size == 0


def test_purge_mask_without_rules_only_drops_the_test_rows() -> None:
    is_test = np.array(
        [False, False, True, True, False, False, False, True, True, False]
    )
    assert overlap.purge_mask(is_test, purge=0, embargo=0).tolist() == (
        (~is_test).tolist()
    )


def test_purge_drops_training_rows_whose_label_reaches_the_test_block() -> None:
    """purge=1：位置 1 与 6 的标签跨进了测试块（2 与 7），必须剔除。"""
    is_test = np.array(
        [False, False, True, True, False, False, False, True, True, False]
    )
    result = overlap.purge_mask(is_test, purge=1, embargo=0)
    assert result.tolist() == [
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
    ]


def test_embargo_drops_training_rows_right_after_the_test_block() -> None:
    """embargo=1：位置 4 与 9 紧跟在测试块之后，序列相关会让它们泄漏。"""
    is_test = np.array(
        [False, False, True, True, False, False, False, True, True, False]
    )
    result = overlap.purge_mask(is_test, purge=0, embargo=1)
    assert result.tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_purge_never_keeps_a_test_row() -> None:
    """无论规则怎么设，测试行本身都不能留在训练集里。"""
    is_test = np.array([True, False, True, False, False])
    for purge, embargo in ((0, 0), (1, 0), (0, 2), (2, 2)):
        result = overlap.purge_mask(is_test, purge=purge, embargo=embargo)
        assert not (result & is_test).any()


def test_embargo_from_ratio_matches_authors_one_percent() -> None:
    assert overlap.embargo_from_ratio(1147) == 11
    assert overlap.embargo_from_ratio(76) == 0


def test_uniqueness_from_ends_reduces_to_the_fixed_horizon_version() -> None:
    """不变量：长度固定时两条路径必须逐位相等。

    不相等就说明两处对区间的开闭约定不一致，Ch 4 的接线会静默错位。
    """
    starts = np.arange(0, 60, 3)
    for horizon in (1, 3, 5, 30):
        ends = np.minimum(starts + horizon, 60)
        fixed = overlap.average_uniqueness(60, starts, horizon)
        variable = overlap.uniqueness_from_ends(60, starts, ends)
        np.testing.assert_allclose(fixed, variable)


def test_variable_length_uniqueness_is_hand_computable() -> None:
    """手算样本：10 根、3 条标签，starts=[0,0,5]、ends=[3,6,8]。

    逐根并发数：bar 0/1/2 → 2，bar 3/4 → 1，bar 5 → 2，bar 6/7 → 1。
    所以逐根 1/并发 = [.5,.5,.5, 1,1, .5, 1,1, 1,1]，前缀和
    = [0,.5,1,1.5,2.5,3.5,4,5,6,7,8]。
    三个观测分别是 0.5、4/6、2.5/3。
    """
    starts = np.array([0, 0, 5])
    ends = np.array([3, 6, 8])
    unique = overlap.uniqueness_from_ends(10, starts, ends)
    np.testing.assert_allclose(unique, [0.5, 4.0 / 6.0, 2.5 / 3.0])


def test_variable_length_labels_do_not_collapse_to_a_constant() -> None:
    """Ch 4 的核心：长度可变时唯一性才有内容。

    固定期限下唯一性恒为 1/h（与抽稀数学等价，见上一条不变量）；
    三重障碍下短标签与长标签混在一起，唯一性必须真的散开。
    注意长短与唯一性**不成单调关系**——长标签跳得更长，但覆盖的拥挤区也不同。
    """
    starts = np.arange(0, 50, 2)
    ends = starts + np.tile([2, 9], 13)[: len(starts)]
    unique = overlap.uniqueness_from_ends(120, starts, ends)
    assert unique.std() > 0.0
    assert unique.max() - unique.min() > 0.3


def test_uniqueness_from_ends_of_nothing_is_empty() -> None:
    assert (
        overlap.uniqueness_from_ends(
            10, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        ).size
        == 0
    )


def test_uniqueness_never_exceeds_one() -> None:
    """并发标签数至少为 1（自己），所以唯一性不可能大于 1。"""
    starts = np.array([0, 1, 2, 10])
    ends = np.array([5, 6, 3, 12])
    unique = overlap.uniqueness_from_ends(20, starts, ends)
    assert (unique > 0).all()
    assert (unique <= 1.0).all()
