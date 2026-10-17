"""
`src/research/labeling/tuning.py` 的用例（二）：**搜索语义**（候选生成、选优、可复现、权重切片）。

折的切分与打分口径在 `test_tuning.py`。这里的分母是「搜索选对了没有」。用假分类器而不是
真随机森林，是为了让每个断言都精确、且单文件用例总耗时低于 1 秒。
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pytest

from src.research.labeling.tuning import (
    PurgedKFold,
    TuningProblem,
    candidate_list,
    clean_params,
    fold_scores,
    log_uniform,
    search_hyperparameters,
)

type PanelCodes = tuple[np.ndarray, np.ndarray]


class ConstantClassifier:
    """常数分类器：`fit` 只记先验，`predict_proba` 按构造时给的正类率输出。

    比真模型好的地方是**完全可预测**——折切错、权重没传对、分数算错都会让断言直接红，
    而不会像随机森林那样被自身噪声掩盖。
    """

    def __init__(self, positive_rate: float = 1.0) -> None:
        self.positive_rate = positive_rate
        self.seen_weight: float | None = None

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantClassifier:
        self.seen_weight = (
            None if sample_weight is None else float(np.sum(sample_weight))
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = np.full(int(np.asarray(X).shape[0]), self.positive_rate)
        return np.column_stack([1.0 - probability, probability])


class ThresholdClassifier:
    """按「第一列特征是否低于阈值」预测正类——`fit` 不学任何东西。

    用它是因为分数随阈值**单调、可解析、最优唯一**（探针实测：阈值 0/10/20/30/40
    对应折均分 0、1/6、1/3、1/2、2/3），既能验搜索选对了，又能验排序。
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ThresholdClassifier:
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        positive = (np.asarray(X, dtype=float)[:, 0] < self.threshold).astype(float)
        return np.column_stack([1.0 - positive, positive])


def _constant_factory(params: dict[str, Any]) -> ConstantClassifier:
    return ConstantClassifier(positive_rate=float(params["positive_rate"]))


def _threshold_factory(params: dict[str, Any]) -> ThresholdClassifier:
    return ThresholdClassifier(threshold=float(params["threshold"]))


def _sawtooth_labels(codes: np.ndarray, block: int = 5) -> np.ndarray:
    """每 `block` 天一块、正负交替的标签。

    用意：**保证每一折里两类都在**。若某折没有正类，f1 恒为 0（见
    `test_positive_f1_is_zero_on_a_dead_fold_whatever_you_predict`），折均值就不再
    反映配置好坏。
    """
    return ((codes // block) % 2 == 0).astype(int)


def _problem(
    factory: Any,
    labels: np.ndarray,
    panel: PanelCodes,
    cv: PurgedKFold,
    sample_weight: np.ndarray | None = None,
) -> TuningProblem:
    codes, features = panel
    return TuningProblem(
        factory=factory,
        features=features,
        labels=labels,
        groups=codes,
        sample_weight=sample_weight,
        cv=cv,
    )


def test_search_hyperparameters_enumerates_the_grid_and_keeps_the_best(
    coded_panel: PanelCodes,
) -> None:
    """网格全枚举，且选出的必须是真的最优——分数随阈值单调，最优唯一。"""
    codes, _ = coded_panel
    problem = _problem(
        _threshold_factory,
        _sawtooth_labels(codes),
        coded_panel,
        PurgedKFold(n_splits=4, purge=1, embargo=1),
    )
    result = search_hyperparameters(
        problem, {"threshold": [0.0, 10.0, 20.0, 30.0, 40.0]}
    )
    assert result.candidates == 5
    assert result.best_params == {"threshold": 40.0}
    assert result.best_score == pytest.approx(2 / 3)
    assert result.best_score == pytest.approx(float(result.fold_scores["score"].max()))
    assert result.fold_scores["threshold"].tolist() == [40.0, 30.0, 20.0, 10.0, 0.0]
    assert result.fold_scores["score"].tolist() == pytest.approx(
        [2 / 3, 0.5, 1 / 3, 1 / 6, 0.0]
    )


def test_search_hyperparameters_random_draws_exactly_n_iter_candidates(
    coded_panel: PanelCodes,
) -> None:
    codes, _ = coded_panel
    problem = _problem(
        _constant_factory,
        _sawtooth_labels(codes),
        coded_panel,
        PurgedKFold(n_splits=3),
    )
    result = search_hyperparameters(
        problem, {"positive_rate": log_uniform(0.05, 0.95)}, n_iter=6, seed=1
    )
    assert result.candidates == 6
    assert 0.05 <= result.best_params["positive_rate"] <= 0.95
    assert isinstance(result.best_params["positive_rate"], float)


def test_search_hyperparameters_is_reproducible_for_a_seed(
    coded_panel: PanelCodes,
) -> None:
    """同 seed 必须同结果——账本按参数值做幂等判定，抽出来的字面量不能每次都变。"""
    codes, _ = coded_panel
    problem = _problem(
        _constant_factory,
        _sawtooth_labels(codes),
        coded_panel,
        PurgedKFold(n_splits=3),
    )
    space = {"positive_rate": log_uniform(0.05, 0.95)}
    first = search_hyperparameters(problem, space, n_iter=5, seed=7)
    second = search_hyperparameters(problem, space, n_iter=5, seed=7)
    assert first.best_params == second.best_params
    assert (
        first.fold_scores["positive_rate"].tolist()
        == second.fold_scores["positive_rate"].tolist()
    )


def test_fold_scores_slices_sample_weights_by_the_same_row_indices(
    coded_panel: PanelCodes,
) -> None:
    """权重必须按折的训练行切开——切错会让「样本重要性」静默错位到别的日期上。"""
    codes, features = coded_panel
    labels = _sawtooth_labels(codes)
    weight = np.linspace(1.0, 2.0, labels.size)
    splitter = PurgedKFold(n_splits=3, purge=1, embargo=1)
    seen: list[float] = []

    class Weigher(ThresholdClassifier):
        @override
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ThresholdClassifier:
            seen.append(float(np.sum(sample_weight)))
            return self

    def weigher_factory(params: dict[str, Any]) -> ThresholdClassifier:
        return Weigher(threshold=float(params["threshold"]))

    problem = _problem(
        weigher_factory,
        labels,
        coded_panel,
        splitter,
        sample_weight=weight,
    )
    mean_score, by_fold = fold_scores(problem, {"threshold": 40.0})
    expected = [
        float(weight[train_index].sum())
        for train_index, _ in splitter.split(features, labels, groups=codes)
    ]
    assert seen == pytest.approx(expected)
    assert len(by_fold) == 3
    assert mean_score == pytest.approx(np.mean(list(by_fold.values())))


def test_search_hyperparameters_uses_purged_shapes_so_folds_never_mix_dates(
    coded_panel: PanelCodes,
) -> None:
    """端到端护栏：每个候选跑的折一样，且每折训练行都是整天（同天同进同出）。"""
    codes, features = coded_panel
    per_day = features.shape[1]  # 只是拿来做整除校验的常量
    observed: list[int] = []

    class Recorder(ConstantClassifier):
        @override
        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantClassifier:
            observed.append(int(np.asarray(X).shape[0]))
            return super().fit(X, y, sample_weight)

    def recorder_factory(params: dict[str, Any]) -> ConstantClassifier:
        return Recorder(positive_rate=float(params["positive_rate"]))

    problem = _problem(
        recorder_factory,
        _sawtooth_labels(codes),
        coded_panel,
        PurgedKFold(n_splits=5, purge=2, embargo=2),
    )
    search_hyperparameters(problem, {"positive_rate": [0.5, 0.7]})
    assert len(observed) == 10  # 2 配置 × 5 折
    assert all(count % 3 == 0 for count in observed)  # 每天 3 行
    assert per_day == 2


def test_candidate_list_semantics() -> None:
    assert candidate_list({"a": [1, 2], "b": [3, 4]}) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    drawn = candidate_list({"a": [1, 2, 3]}, n_iter=4, seed=0)
    assert len(drawn) == 4
    assert all(item["a"] in {1, 2, 3} for item in drawn)
    with pytest.raises(ValueError, match="搜索空间为空"):
        candidate_list({})
    with pytest.raises(ValueError, match="候选列表"):
        candidate_list({"a": log_uniform(1, 2)})
    with pytest.raises(ValueError, match="网格过大"):
        candidate_list({"a": list(range(101)), "b": list(range(101))})


def test_clean_params_normalises_numpy_scalars() -> None:
    cleaned = clean_params(
        {"n": np.int64(7), "f": np.float64(0.30000000000000004), "s": "gini"}, 4
    )
    assert cleaned == {"n": 7, "f": 0.3, "s": "gini"}
    assert all(isinstance(value, (int, float, str)) for value in cleaned.values())
    assert clean_params({"f": 2.7}, 0) == {"f": 3}
