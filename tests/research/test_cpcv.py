"""CPCV（de Prado 第 12 章）的不变量：切分穷尽性、purge/embargo、与 K 折的等价、路径分配。"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.research.cpcv import CombinatorialPurgedCv, path_series
from src.research.tuning import PurgedKFold

GROUPS = 6
K_TEST = 2
DATES = 120


def _cv(**overrides: int) -> CombinatorialPurgedCv:
    params = {"n_groups": GROUPS, "k_test": K_TEST, "purge": 0, "embargo": 0}
    params.update(overrides)
    return CombinatorialPurgedCv(**params)


def test_counts_follow_the_book_formulas() -> None:
    cv = _cv()
    assert cv.n_splits == math.comb(GROUPS, K_TEST) == 15
    assert cv.n_paths == math.comb(GROUPS, K_TEST) * K_TEST // GROUPS == 5
    # 每组恰好出现在 φ 次切分里——这是「路径能填满」的前提，不是巧合
    assert math.comb(GROUPS - 1, K_TEST - 1) == cv.n_paths


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="组数至少 3"):
        CombinatorialPurgedCv(n_groups=2, k_test=1)
    with pytest.raises(ValueError, match="测试组数"):
        CombinatorialPurgedCv(n_groups=5, k_test=5)
    with pytest.raises(ValueError, match="不能为负"):
        CombinatorialPurgedCv(n_groups=5, k_test=2, purge=-1)
    with pytest.raises(ValueError, match="唯一日期数少于组数"):
        _cv().split(GROUPS - 1)


def test_splits_are_exactly_the_combinations() -> None:
    cv = _cv()
    splits = cv.split(DATES)
    assert [split.test_groups for split in splits] == list(
        combinations(range(GROUPS), K_TEST)
    )
    groups = cv.group_dates(DATES)
    for split in splits:
        expected = np.concatenate([groups[group] for group in split.test_groups])
        assert split.test_dates.tolist() == expected.tolist()
        assert not set(split.test_dates.tolist()) & set(split.train_dates.tolist())
        assert len(split.train_dates) == DATES - expected.size


def test_purge_and_embargo_windows_are_excluded_exactly() -> None:
    """不只是「挖掉了一些」——多挖一天少挖一天都算错，这里按集合严等。"""
    cv = _cv(purge=3, embargo=4)
    groups = cv.group_dates(DATES)
    for split in cv.split(DATES):
        banned = set(split.test_dates.tolist())
        for group in split.test_groups:
            start = int(groups[group][0])
            stop = int(groups[group][-1]) + 1
            banned |= set(range(max(start - cv.purge, 0), start))
            banned |= set(range(stop, min(stop + cv.embargo, DATES)))
        assert not banned & set(split.train_dates.tolist())
        assert len(split.train_dates) == DATES - len(banned)


def test_train_keeps_the_dates_after_the_test_block() -> None:
    """训练集是「测试组之外的全部」，不是只看历史——这是 k=1 能与 K 折等价的原因。"""
    cv = CombinatorialPurgedCv(n_groups=4, k_test=1)
    split = next(item for item in cv.split(40) if item.test_groups == (1,))
    assert split.train_dates.min() < split.test_dates.min()
    assert split.train_dates.max() > split.test_dates.max()


def test_k_test_one_equals_the_purged_k_fold() -> None:
    """k=1 时 CPCV 必须与第 9 章的 PurgedKFold 逐行等价（同一切法、同一套 purge/embargo）。"""
    codes = np.repeat(np.arange(DATES), 3)  # 一天三行，模拟面板
    frame = pd.DataFrame({"x": np.arange(codes.size)})
    cv = CombinatorialPurgedCv(n_groups=5, k_test=1, purge=2, embargo=3)
    splits = cv.split(DATES)
    folds = list(PurgedKFold(n_splits=5, purge=2, embargo=3).split(frame, groups=codes))
    assert len(folds) == len(splits)
    for split, (train_rows, test_rows) in zip(splits, folds, strict=True):
        assert set(np.unique(codes[test_rows]).tolist()) == set(
            split.test_dates.tolist()
        )
        assert set(np.unique(codes[train_rows]).tolist()) == set(
            split.train_dates.tolist()
        )


def test_paths_cover_every_group_exactly_once() -> None:
    cv = _cv()
    assignment = cv.paths()
    splits = cv.split(DATES)
    assert assignment.shape == (cv.n_paths, GROUPS)
    assert (assignment >= 0).all() and (assignment < cv.n_splits).all()
    for path in assignment:
        # 同一条路径的多个组**可以**来自同一次切分（贪心分配本就如此，参照实现亦然）；
        # 要求的是「每组只有一个来源」与「来源确实把该组当测试」。
        assert len(path.tolist()) == GROUPS
        for group, split_index in enumerate(path):
            assert group in splits[int(split_index)].test_groups


def test_every_group_is_offered_by_exactly_n_paths_splits() -> None:
    cv = _cv()
    splits = cv.split(DATES)
    for group in range(GROUPS):
        assert sum(group in split.test_groups for split in splits) == cv.n_paths


def test_path_series_stitches_the_assigned_segments() -> None:
    cv = _cv()
    splits = cv.split(DATES)
    groups = cv.group_dates(DATES)
    assignment = cv.paths()
    index = pd.date_range("2020-01-01", periods=DATES, tz="UTC")
    # 每次切分的测试段填常数（=切分序号），训练段留 NaN：路径里读数就等于分配矩阵
    returns = []
    for number, split in enumerate(splits):
        series = pd.Series(np.nan, index=index)
        series.iloc[split.test_dates] = float(number)
        returns.append(series)
    frame = path_series(returns, paths=assignment, group_dates=groups)
    assert list(frame.columns) == [f"路径{index + 1}" for index in range(cv.n_paths)]
    assert not frame.isna().to_numpy().any()
    for column in range(cv.n_paths):
        for group in range(GROUPS):
            positions = groups[group]
            expected = float(assignment[column, group])
            assert (frame.iloc[positions, column] == expected).all()


def test_path_series_rejects_unaligned_input() -> None:
    cv = CombinatorialPurgedCv(n_groups=4, k_test=2)
    with pytest.raises(ValueError, match="必须都对齐"):
        path_series(
            [pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0, 3.0])],
            paths=cv.paths(),
            group_dates=cv.group_dates(12),
        )
