"""
`src/research/labeling/tuning.py` 的用例（一）：**折的切分与打分口径**。

断言分两类：**口径不变式**（折不跨日、purge/embargo 精确挖掉哪几天、打分与 sklearn
逐值相同）与**行为标定**（先跑探针量出再写死）。搜索部分的用例在
`test_tuning_search.py`——分开是因为两处的失败含义不同：这里的红是「折切错了」，
那里的红是「搜索选错了」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.labeling.tuning import (
    PurgedKFold,
    log_uniform,
    positive_f1,
    scoring_for,
)

type PanelCodes = tuple[np.ndarray, np.ndarray]


def test_purged_kfold_keeps_same_date_together(coded_panel: PanelCodes) -> None:
    """同一天的所有标的必须落在同一折——这是本仓吃过的那次「同日泄漏」。"""
    codes, features = coded_panel
    splitter = PurgedKFold(n_splits=4, purge=1, embargo=1)
    for train_index, test_index in splitter.split(features, groups=codes):
        assert not set(codes[train_index]) & set(codes[test_index])


def test_purged_kfold_embargo_removes_the_next_days(coded_panel: PanelCodes) -> None:
    """embargo=k 时，测试块**之后**的 k 天必须不在训练集里。"""
    codes, features = coded_panel
    splitter = PurgedKFold(n_splits=4, purge=0, embargo=2)
    second_train, second_test = list(splitter.split(features, groups=codes))[1]
    stop = int(np.max(codes[second_test]))
    trained = set(codes[second_train])
    assert {stop + 1, stop + 2} & trained == set()
    assert stop + 3 in trained  # 只挖两天，第三天要回来


def test_purged_kfold_purge_removes_the_previous_days(coded_panel: PanelCodes) -> None:
    """purge=k 时，测试块**之前**的 k 天必须不在训练集里（标签往回跨）。"""
    codes, features = coded_panel
    train_index, test_index = list(
        PurgedKFold(n_splits=4, purge=3, embargo=0).split(features, groups=codes)
    )[1]
    start = int(np.min(codes[test_index]))
    trained = set(codes[train_index])
    assert {start - 1, start - 2, start - 3} & trained == set()
    assert start - 4 in trained


def test_purged_kfold_without_gaps_partitions_every_date(
    coded_panel: PanelCodes,
) -> None:
    """purge=embargo=0 时退化成普通时间 K 折：训练与测试合起来正好是全部日期。"""
    codes, features = coded_panel
    every = set(range(int(codes.max()) + 1))
    covered: set[int] = set()
    for train_index, test_index in PurgedKFold(n_splits=4).split(
        features, groups=codes
    ):
        assert set(codes[train_index]) | set(codes[test_index]) == every
        covered |= set(codes[test_index])
    assert covered == every


def test_purged_kfold_requires_groups_and_enough_dates(coded_panel: PanelCodes) -> None:
    codes, features = coded_panel
    with pytest.raises(ValueError, match="groups"):
        list(PurgedKFold(n_splits=3).split(features))
    with pytest.raises(ValueError, match="唯一日期数"):
        list(PurgedKFold(n_splits=codes.max() + 2).split(features, groups=codes))


def test_purged_kfold_validates_arguments() -> None:
    with pytest.raises(ValueError, match="折数"):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError, match="不能为负"):
        PurgedKFold(n_splits=3, purge=-1)


def test_positive_f1_matches_sklearn_on_a_normal_case() -> None:
    from sklearn.metrics import f1_score

    labels = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 0, 1, 1, 0])
    expected = float(
        f1_score(labels, predicted)
    )  # 非退化用例：不触发 zero_division 警告
    assert positive_f1(labels, predicted) == pytest.approx(expected)


def test_positive_f1_is_zero_when_nothing_is_predicted_positive() -> None:
    """退化折给 0 而不是抛错/发警告——本仓把警告升级成错误，这条是那堵墙的护栏。"""
    labels = np.array([0, 1, 0, 0])
    assert positive_f1(labels, np.zeros(4, dtype=int)) == 0.0
    assert positive_f1(np.zeros(4, dtype=int), np.zeros(4, dtype=int)) == 0.0
    assert positive_f1(np.ones(4, dtype=int), np.ones(4, dtype=int)) == 1.0


def test_positive_f1_is_zero_on_a_dead_fold_whatever_you_predict() -> None:
    """某折没有正类时 f1 恒为 0：预测全 0 与预测全 1 都得 0（探针实测）。

    所以正类稀少的任务里，这种折对调参**零信息**，只会把折均值往 0 拉。这条是模块
    docstring 第 4 点的护栏：跑出来才发现两者相同，不是凭直觉能想到的。
    """
    no_positives = np.zeros(6, dtype=int)
    assert positive_f1(no_positives, np.zeros(6, dtype=int)) == 0.0
    assert positive_f1(no_positives, np.ones(6, dtype=int)) == 0.0
    assert positive_f1(np.array([1, 0]), np.array([1, 0])) == 1.0
    assert positive_f1(np.array([1, 0]), np.array([0, 0])) == 0.0


def test_scoring_for_picks_f1_for_meta_labels() -> None:
    assert scoring_for(np.array([0, 1, 1, 0])) == "f1"
    assert scoring_for(pd.Series([0, 1])) == "f1"


def test_scoring_for_picks_log_loss_for_multi_class() -> None:
    assert scoring_for(np.array([0, 1, 2])) == "neg_log_loss"
    assert scoring_for(np.array([-1, 1])) == "neg_log_loss"


def test_log_uniform_rejects_degenerate_bounds() -> None:
    with pytest.raises(ValueError, match="0 < low < high"):
        log_uniform(0.0, 10.0)
    with pytest.raises(ValueError, match="0 < low < high"):
        log_uniform(10.0, 1.0)


def test_log_uniform_puts_half_the_mass_below_the_geometric_mean() -> None:
    """对数均匀的**判别性**性质：几何均值两侧各一半。

    均匀分布在同一区间上只把 ln 尺度的一小段放在低端——两者一比就分得开，而「样本落在
    [low, high] 内」这种弱断言均匀分布也满足，测不出东西。
    """
    low, high = 1.0, 1024.0
    samples = np.asarray(log_uniform(low, high).rvs(size=20000, random_state=0))
    geometric_mean = np.sqrt(low * high)
    assert float(np.mean(samples < geometric_mean)) == pytest.approx(0.5, abs=0.02)
    uniform_share = (geometric_mean - low) / (high - low)
    assert uniform_share < 0.04  # 均匀分布只有 3%，判别力足够
    assert samples.min() >= low and samples.max() <= high
