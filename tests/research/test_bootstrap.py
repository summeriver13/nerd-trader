"""
tests/research/test_bootstrap.py
顺序 bootstrap 采样器（de Prado Ch 4.5）：与朴素重建法逐位对拍，并钉住它有效的条件。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.research.validation import bootstrap


def _spans() -> tuple[np.ndarray, np.ndarray]:
    """三段重叠区间：[0,10)、[0,1)、[5,6)。"""
    return np.array([0, 0, 5]), np.array([10, 1, 6])


def _ladder() -> tuple[np.ndarray, np.ndarray, int]:
    """300 根上每 3 根起一条、跨度 1~5 根轮转的标签——低并发带自身重复的场景。"""
    starts = np.arange(0, 300, 3)
    return starts, starts + np.resize(np.array([1, 2, 3, 4, 5]), starts.size), 300


def _naive_bag_uniqueness(
    size: int, starts: np.ndarray, ends: np.ndarray, draws: list[int]
) -> float:
    """原书 getAvgUniqueness 的直译：重建袋矩阵，取 (根, 标签) 单元的 1/并发 均值。"""
    picked = np.asarray(draws, dtype=np.int64)
    matrix = bootstrap.indicator_matrix(size, starts[picked], ends[picked])
    cells = matrix / np.maximum(matrix.sum(axis=1), 1.0)[:, None]
    return float(cells[matrix > 0].mean())


def _naive_candidate(
    size: int, starts: np.ndarray, ends: np.ndarray, draws: list[int], index: int
) -> float:
    """原书 `getAvgUniqueness(indM_).iloc[-1]`：只看候选项自己那一列的均值。"""
    picked = np.asarray([*draws, index], dtype=np.int64)
    matrix = bootstrap.indicator_matrix(size, starts[picked], ends[picked])
    cells = matrix / np.maximum(matrix.sum(axis=1), 1.0)[:, None]
    column = cells[:, -1]
    return float(column[column > 0].mean())


def _naive_sequential(
    size: int, starts: np.ndarray, ends: np.ndarray, length: int, seed: int
) -> np.ndarray:
    """原书伪码直译：每步为每个候选重建一次袋矩阵（O(length·N·T)），只用于对拍。"""
    rng = np.random.default_rng(seed)
    draws: list[int] = []
    for _ in range(length):
        weights = np.array(
            [_naive_candidate(size, starts, ends, draws, i) for i in range(starts.size)]
        )
        draws.append(int(rng.choice(starts.size, p=weights / weights.sum())))
    return np.asarray(draws, dtype=np.int64)


def test_indicator_matrix_marks_every_bar_in_span() -> None:
    starts, ends = _spans()
    matrix = bootstrap.indicator_matrix(10, starts, ends)
    assert matrix.shape == (10, 3)
    assert matrix[:, 0].sum() == 10
    assert matrix[:, 1].sum() == 1
    assert matrix[4, 2] == 0.0
    assert matrix[5, 2] == 1.0


def test_bag_uniqueness_matches_the_matrix_definition() -> None:
    starts, ends = _spans()
    draws = [0, 0, 1, 2]
    assert bootstrap.bag_uniqueness(10, starts, ends, np.array(draws)) == pytest.approx(
        _naive_bag_uniqueness(10, starts, ends, draws)
    )


def test_candidate_uniqueness_matches_the_naive_rebuild() -> None:
    """抽签权重必须等于「把候选加进袋子后重建矩阵，取候选自己那列」。"""
    starts, ends = _spans()
    concurrency = bootstrap.bag_concurrency(10, starts, ends, np.array([0]))
    weights = bootstrap.candidate_uniqueness(10, starts, ends, concurrency)
    for index in range(starts.size):
        assert weights[index] == pytest.approx(
            _naive_candidate(10, starts, ends, [0], index)
        )


def test_sequential_bootstrap_matches_the_naive_reference() -> None:
    """同一颗种子下，前缀和版与原书逐候选重建版抽出完全相同的序列。"""
    starts, ends = _spans()
    assert np.array_equal(
        bootstrap.sequential_bootstrap(10, starts, ends, 12, seed=7),
        _naive_sequential(10, starts, ends, 12, 7),
    )


def test_sequential_bag_is_more_unique_on_low_concurrency_labels() -> None:
    """低并发标签上，顺序袋的平均唯一性逐颗种子都高于均匀袋（实测 30/30）。"""
    starts, ends, size = _ladder()
    length = int(starts.size)
    for seed in range(30):
        sequential = bootstrap.bag_uniqueness(
            size,
            starts,
            ends,
            bootstrap.sequential_bootstrap(size, starts, ends, length, seed=seed),
        )
        uniform = bootstrap.bag_uniqueness(
            size,
            starts,
            ends,
            bootstrap.uniform_bootstrap(int(starts.size), length, seed=seed),
        )
        assert sequential > uniform


def test_uniform_bootstrap_draws_each_label_about_equally() -> None:
    """对照臂本身是均匀的：100 条各抽约 100 次。"""
    draws = bootstrap.uniform_bootstrap(100, 10_000, seed=0)
    counts = np.bincount(draws, minlength=100)
    assert counts.mean() == pytest.approx(100.0, rel=0.05)
    assert int(draws.min()) >= 0
    assert int(draws.max()) < 100


def test_sequential_bootstrap_is_deterministic_and_in_range() -> None:
    starts, ends = _spans()
    first = bootstrap.sequential_bootstrap(10, starts, ends, 20, seed=3)
    second = bootstrap.sequential_bootstrap(10, starts, ends, 20, seed=3)
    assert np.array_equal(first, second)
    assert first.shape == (20,)
    assert int(first.min()) >= 0
    assert int(first.max()) < starts.size


def test_empty_inputs_and_nonpositive_length() -> None:
    empty = np.zeros(0, dtype=int)
    starts, ends = _spans()
    assert bootstrap.sequential_bootstrap(10, empty, empty, 5, seed=0).size == 0
    assert bootstrap.sequential_bootstrap(10, starts, ends, 0, seed=0).size == 0
    assert bootstrap.sequential_bootstrap(10, starts, ends, -1, seed=0).size == 0
    assert bootstrap.uniform_bootstrap(0, 5, seed=0).size == 0
    assert bootstrap.bag_uniqueness(10, starts, ends, empty) == pytest.approx(1.0)


def test_degenerate_span_still_covers_one_bar() -> None:
    """t1 == t0 的标签按 1 根算，否则它的抽签概率（乃至全体的）会归零。"""
    starts = np.array([0, 0])
    ends = np.array([0, 4])
    assert bootstrap.indicator_matrix(6, starts, ends)[0, 0] == 1.0
    draws = bootstrap.sequential_bootstrap(6, starts, ends, 6, seed=0)
    concurrency = bootstrap.bag_concurrency(6, starts, ends, draws)
    assert (
        float(bootstrap.candidate_uniqueness(6, starts, ends, concurrency).min()) > 0.0
    )
