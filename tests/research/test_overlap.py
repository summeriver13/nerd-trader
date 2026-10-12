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
    is_test = np.array([False, False, True, True, False, False, False, True, True, False])
    assert overlap.purge_mask(is_test, purge=0, embargo=0).tolist() == (
        (~is_test).tolist()
    )


def test_purge_drops_training_rows_whose_label_reaches_the_test_block() -> None:
    """purge=1：位置 1 与 6 的标签跨进了测试块（2 与 7），必须剔除。"""
    is_test = np.array([False, False, True, True, False, False, False, True, True, False])
    result = overlap.purge_mask(is_test, purge=1, embargo=0)
    assert result.tolist() == [True, False, False, False, True, True, False, False, False, True]


def test_embargo_drops_training_rows_right_after_the_test_block() -> None:
    """embargo=1：位置 4 与 9 紧跟在测试块之后，序列相关会让它们泄漏。"""
    is_test = np.array([False, False, True, True, False, False, False, True, True, False])
    result = overlap.purge_mask(is_test, purge=0, embargo=1)
    assert result.tolist() == [True, True, False, False, False, True, True, False, False, False]


def test_purge_never_keeps_a_test_row() -> None:
    """无论规则怎么设，测试行本身都不能留在训练集里。"""
    is_test = np.array([True, False, True, False, False])
    for purge, embargo in ((0, 0), (1, 0), (0, 2), (2, 2)):
        result = overlap.purge_mask(is_test, purge=purge, embargo=embargo)
        assert not (result & is_test).any()


def test_embargo_from_ratio_matches_authors_one_percent() -> None:
    assert overlap.embargo_from_ratio(1147) == 11
    assert overlap.embargo_from_ratio(76) == 0
