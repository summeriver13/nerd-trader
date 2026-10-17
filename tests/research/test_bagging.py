"""袋化集成的用例。

四条口径各自对应用例：① 同种子逐位可复现、袋子按标的分层；② 与「按同一种子逐袋重拟合」
的朴素实现逐位等价（否则「采样器的差异」会混进「实现的差异」）；③ 袋子只用得到它抽中的行
（未抽中的行改到爆也不该动预测）；④ 顺序臂真的换了袋子（低并发梯子上唯一性必须更高）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.research.labeling import bagging, meta

OPTIONS = bagging.BaggingOptions(
    n_estimators=3, max_depth=3, min_samples_leaf=2, seed=1
)


def _spans(
    per_symbol: int = 40, symbols: int = 2, step: int = 3
) -> bagging.SpanCoordinates:
    """低并发梯子：每只标的每 `step` 根起一条标签，跨度 1~5 轮转。

    这是顺序采样该赢的场合（Ch 4.5 的读数）：标签几乎不重叠，抽签权重的杠杆最大。
    """
    starts = np.tile(np.arange(0, per_symbol * step, step), symbols)
    spans = np.resize(np.array([0, 1, 2, 3, 4]), starts.size)
    return bagging.SpanCoordinates(
        groups=np.repeat(np.arange(symbols), per_symbol),
        sizes=np.full(symbols, per_symbol * step, dtype=np.int64),
        starts=starts,
        ends=starts + spans,
    )


def _data(rows: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, 3))
    target = (values[:, 0] + 0.1 * generator.normal(size=rows) > 0).astype(int)
    return values, target, np.ones(rows)


def _fitted(spans: bagging.SpanCoordinates, sampler: str = bagging.SAMPLER_SEQUENTIAL):
    values, target, weight = _data(int(spans.groups.size))
    return bagging.BaggedClassifier(OPTIONS, spans, sampler).fit(values, target, weight)


def test_draw_is_deterministic_and_stratified_by_symbol() -> None:
    """同种子逐位相同、袋子总量等于训练行数、每只标的各交自己那一份。"""
    spans = _spans()
    first = spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=7)
    second = spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=8))
    assert first.size == spans.groups.size
    for group in np.unique(spans.groups):
        expected = int(np.sum(spans.groups == group))
        assert int(np.sum(spans.groups[first] == group)) == expected
    assert spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=7).size == first.size


def test_bags_are_drawn_per_symbol_not_from_the_pool() -> None:
    """袋子按标的分层：标的一的行永远进不了标的二的袋位。"""
    spans = _spans(symbols=3)
    drawn = spans.draw(bagging.SAMPLER_UNIFORM, seed=3)
    assert int(np.unique(spans.groups[drawn]).size) == int(spans.sizes.size)
    # 逐标的的抽中次数与它自己的行数相等（分层抽样的定义）
    counts = np.bincount(spans.groups[drawn], minlength=int(spans.sizes.size))
    np.testing.assert_array_equal(counts, np.bincount(spans.groups))


def test_sequential_bags_are_more_unique_on_a_low_concurrency_ladder() -> None:
    """顺序臂真的换了袋子：低并发梯子上，顺序袋的袋唯一性必须更高。

    逐标的、逐个种子地比会翻（本夹具 16 个组合里有 2 个略负），所以断言的是「先按标的
    取均值」那一层——它在 8 颗种子上全为正，与 Ch 4.5 的读数同向。
    """
    spans = _spans()
    for seed in range(8):
        sequential = float(
            spans.uniqueness_by_group(bagging.SAMPLER_SEQUENTIAL, seed=seed).mean()
        )
        uniform = float(
            spans.uniqueness_by_group(bagging.SAMPLER_UNIFORM, seed=seed).mean()
        )
        assert sequential > uniform


def test_bagged_forest_matches_a_literal_refit() -> None:
    """与朴素实现逐位等价：同一批袋子、同一批种子、逐袋重拟合一棵树再取平均。"""
    spans = _spans()
    values, target, weight = _data(int(spans.groups.size))
    model = bagging.BaggedClassifier(OPTIONS, spans).fit(values, target, weight)

    trees = []
    for index in range(OPTIONS.n_estimators):
        seed = bagging.tree_seed(index, OPTIONS.seed)
        rows = spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=seed)
        tree = DecisionTreeClassifier(
            max_depth=OPTIONS.max_depth,
            min_samples_leaf=OPTIONS.min_samples_leaf,
            max_features=OPTIONS.max_features,
            random_state=seed,
        )
        tree.fit(values[rows], target[rows], sample_weight=weight[rows])
        trees.append(tree)
    expected = np.asarray(
        [np.asarray(tree.predict_proba(values))[:, 1] for tree in trees]
    ).mean(axis=0)
    np.testing.assert_array_equal(
        np.asarray(model.predict_proba(values))[:, 1], expected
    )


def test_untouched_rows_do_not_affect_predictions() -> None:
    """无前视：袋子没抽中的行改成爆值，预测必须逐位不变。"""
    spans = _spans()
    values, target, weight = _data(int(spans.groups.size))
    model = bagging.BaggedClassifier(OPTIONS, spans).fit(values, target, weight)

    drawn: set[int] = set()
    for index in range(OPTIONS.n_estimators):
        seed = bagging.tree_seed(index, OPTIONS.seed)
        drawn.update(
            int(row) for row in spans.draw(bagging.SAMPLER_SEQUENTIAL, seed=seed)
        )
    untouched = [row for row in range(values.shape[0]) if row not in drawn]
    assert untouched  # 袋子是抽样，必然留下没抽中的行

    poisoned = values.copy()
    poisoned[untouched] = 1e6
    poisoned_model = bagging.BaggedClassifier(OPTIONS, spans).fit(
        poisoned, target, weight
    )
    np.testing.assert_array_equal(
        np.asarray(model.predict_proba(values)),
        np.asarray(poisoned_model.predict_proba(values)),
    )


def test_rejects_unknown_sampler_and_mismatched_rows() -> None:
    spans = _spans()
    with pytest.raises(ValueError, match="未知袋子抽法"):
        spans.draw("resample", seed=0)
    values, target, weight = _data(int(spans.groups.size) - 1)
    with pytest.raises(ValueError, match="行数与跨度坐标不一致"):
        bagging.BaggedClassifier(OPTIONS, spans).fit(values, target, weight)


def test_feature_importances_average_the_trees() -> None:
    """MDI 与随机森林同定义：各棵树的 `feature_importances_` 取平均。"""
    model = _fitted(_spans())
    expected = np.asarray([tree.feature_importances_ for tree in model.trees]).mean(
        axis=0
    )
    np.testing.assert_allclose(model.feature_importances_, expected)


def test_span_columns_round_trip_from_a_shuffled_frame() -> None:
    """`from_frame` 按行读四列，帧的行序（研究脚本是标的主序）不该影响结果。"""
    spans = _spans(per_symbol=6, symbols=2, step=2)
    frame = pd.DataFrame(
        {
            bagging.GROUP_COLUMN: spans.groups,
            bagging.START_COLUMN: spans.starts,
            bagging.END_COLUMN: spans.ends,
            bagging.SIZE_COLUMN: spans.sizes[spans.groups],
        }
    ).sample(frac=1.0, random_state=0)
    restored = bagging.SpanCoordinates.from_frame(frame)
    np.testing.assert_array_equal(restored.sizes, spans.sizes)
    for group in np.unique(spans.groups):
        rows = np.flatnonzero(spans.groups == group)
        order = np.argsort(spans.starts[rows])
        chosen = np.flatnonzero(restored.groups == group)
        reorder = np.argsort(restored.starts[chosen])
        np.testing.assert_array_equal(
            restored.starts[chosen][reorder], spans.starts[rows][order]
        )
        np.testing.assert_array_equal(
            restored.ends[chosen][reorder], spans.ends[rows][order]
        )


def test_bagged_arm_runs_through_walk_forward_meta() -> None:
    """端到端：手写集成接进滚动预测的扩展点，概率落在 [0, 1] 且确实有折。"""
    spans = _spans(per_symbol=120, symbols=2, step=2)
    rows = int(spans.groups.size)
    dates = np.repeat(pd.date_range("2024-01-01", periods=spans.groups.size // 2), 2)
    values, target, _ = _data(rows, seed=5)
    frame = pd.DataFrame(
        {
            meta.DATE_COLUMN: dates,
            "signal": values[:, 0],
            meta.LABEL_COLUMN: target,
            meta.WEIGHT_COLUMN: np.ones(rows),
            bagging.GROUP_COLUMN: spans.groups,
            bagging.START_COLUMN: spans.starts,
            bagging.END_COLUMN: spans.ends,
            bagging.SIZE_COLUMN: spans.sizes[spans.groups],
        }
    )
    options = meta.MetaLabelOptions(
        min_train=50,
        test_span=15,
        embargo=0,
        n_estimators=3,
        max_depth=3,
        min_samples_leaf=2,
    )
    prediction = meta.walk_forward_meta(
        frame,
        ["signal"],
        options,
        fit=bagging.bagged_factory(bagging.SAMPLER_SEQUENTIAL),
    )
    assert prediction.folds > 0
    observed = prediction.probability.dropna()
    assert observed.size > 0
    assert float(observed.min()) >= 0.0 and float(observed.max()) <= 1.0
