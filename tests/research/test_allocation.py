"""第 16 章 HRP 的用例：距离口径、准对角化与原算法逐字等价、权重不变量、退化输入。

准对角化那条最重要：本仓用深度优先展开代替原文的迭代插入法，**对着原文算法逐字比对**
才算证明等价，否则「看起来排对了」在簇边界上很容易挑出不同的树序。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.allocation import (
    correlation_distance,
    hrp_weights,
    inverse_variance_weights,
    quasi_diagonal,
    recursive_bisection,
    single_linkage,
)


def _book_quasi_diagonal(link: np.ndarray) -> list[int]:
    """原文 getQuasiDiag 的逐字重写，只作对照用（本仓实现是深度优先展开）。

    原文用 pandas 反复重排索引来「给簇让位」；那一步的**语义**就是：每一轮把当前序列
    里每个簇条目原地换成它的两个孩子。这里用列表写同一个语义，避开桩的类型联合。
    """
    tree = link.astype(int)
    items = int(tree[-1, 3])
    order = [int(tree[-1, 0]), int(tree[-1, 1])]
    while max(order) >= items:
        expanded: list[int] = []
        for value in order:
            if value >= items:
                expanded.append(int(tree[value - items, 0]))
                expanded.append(int(tree[value - items, 1]))
            else:
                expanded.append(value)
        order = expanded
    return order


def _random_correlation(size: int, seed: int) -> np.ndarray:
    """随机相关矩阵：先造因子结构，再归一化成相关。"""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(size, max(1, size // 3)))
    cov = loadings @ loadings.T + np.diag(rng.uniform(0.2, 1.0, size=size))
    scale = np.sqrt(np.diag(cov))
    return cov / np.outer(scale, scale)


def test_distance_metric_matches_the_book_formula() -> None:
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    distance = correlation_distance(corr)
    assert distance[0, 1] == pytest.approx(np.sqrt((1.0 - 0.5) / 2.0))
    assert distance[0, 0] == 0.0
    assert distance[1, 1] == 0.0


def test_quasi_diagonal_matches_the_book_implementation() -> None:
    """本仓的深度优先展开必须与原文的迭代插入法给出**同一个序列**。"""
    for seed in (0, 1, 2, 7):
        for size in (2, 3, 5, 12):
            corr = _random_correlation(size, seed)
            tree = single_linkage(corr)
            assert quasi_diagonal(tree) == _book_quasi_diagonal(tree), (seed, size)


def test_quasi_diagonal_is_a_permutation() -> None:
    corr = _random_correlation(9, 3)
    order = quasi_diagonal(single_linkage(corr))
    assert sorted(order) == list(range(9))


def test_quasi_diagonal_keeps_clusters_contiguous() -> None:
    """两个明显分离的簇：准对角化后同一簇的成员必须连在一起。"""
    block = np.array(
        [
            [1.0, 0.9, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.9, 0.0, 0.0],
            [0.9, 0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.0, 0.9, 1.0],
        ]
    )
    order = quasi_diagonal(single_linkage(block))
    first = [position for position, value in enumerate(order) if value < 3]
    second = [position for position, value in enumerate(order) if value >= 3]
    assert max(first) - min(first) == len(first) - 1
    assert max(second) - min(second) == len(second) - 1


def test_inverse_variance_weights_are_inverse_variance() -> None:
    cov = np.diag([0.04, 0.01, 0.02])
    weights = inverse_variance_weights(cov)
    expected = np.array([1 / 0.04, 1 / 0.01, 1 / 0.02])
    assert weights == pytest.approx(expected / expected.sum())
    assert float(weights.sum()) == pytest.approx(1.0)


def test_inverse_variance_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="严格为正"):
        inverse_variance_weights(np.diag([0.04, 0.0]))


def test_two_assets_reproduce_the_inverse_variance_split() -> None:
    """两只资产时 HRP 就是逆方差组合——原文的退化情形，可手算。"""
    frame = pd.DataFrame(
        {
            "low_vol": np.tile([0.01, -0.01], 60),
            "high_vol": np.tile([0.02, -0.02], 60),
        }
    )
    weights = hrp_weights(frame)
    covariance = frame.cov().to_numpy(dtype=float)
    assert weights.to_numpy() == pytest.approx(inverse_variance_weights(covariance))


def test_weights_are_long_only_and_sum_to_one() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(400, 12)) * 0.01)
    weights = hrp_weights(frame)
    assert weights.min() >= 0.0
    assert float(weights.sum()) == pytest.approx(1.0)
    assert list(weights.index) == list(frame.columns)


def test_two_identical_clusters_split_evenly() -> None:
    """两个簇各自内部完全相关、簇间不相关：每簇各拿一半（簇方差相同时可手算）。

    簇**内**不再等分见下一个用例——那是递归二分的顺序依赖，不是这里要说的事。
    """
    rng = np.random.default_rng(5)
    common = rng.normal(size=400) * 0.01
    other = rng.normal(size=400) * 0.01
    # 把两簇缩放到同一方差：否则簇方差不等，「各拿一半」就不是这条用例要说的事了
    other = other / float(other.std()) * float(common.std())
    frame = pd.DataFrame(
        {
            "a1": common,
            "a2": common,
            "a3": common,
            "b1": other,
            "b2": other,
            "b3": other,
        }
    )
    weights = hrp_weights(frame)
    assert float(weights[["a1", "a2", "a3"]].sum()) == pytest.approx(0.5)
    assert float(weights[["b1", "b2", "b3"]].sum()) == pytest.approx(0.5)


def test_odd_cluster_size_breaks_symmetry() -> None:
    """三只**完全相同**的资产拿到 1/4 、1/2 、1/4 而不是等权——这是 HRP 固有的顺序依赖。

    递归二分把 [甲、乙、丙] 切成 [甲] 与 [乙、丙]，再切 [乙] 与 [丙]；方差全相等时
    三个切分都是 50/50，但切分**层次**不同，于是甲丙拿 1/4、乙拿 1/2（差一倍）。
    意味着「同一簇内哪只拿得多」由准对角顺序决定，而不是由风险决定。

    原文 getRecBipart 就是如此（不是实现 bug）；本用例把它钉住，避免日后被当成
    实现错误「修」掉，也提醒：簇内权重会因列顺序与样本变化而漂移。
    """
    rng = np.random.default_rng(5)
    common = rng.normal(size=400) * 0.01
    frame = pd.DataFrame({"a1": common, "a2": common, "a3": common})
    weights = hrp_weights(frame)
    assert float(weights.max() / weights.min()) == pytest.approx(2.0)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_hrp_is_less_concentrated_than_inverse_variance_on_a_correlated_block() -> None:
    """原文的核心主张：簇内高度相关时，逆方差会把权重押到「方差最小的那只」上，HRP 靠聚类分散。"""
    rng = np.random.default_rng(17)
    common = rng.normal(size=500) * 0.02
    noisy = np.column_stack(
        [common + rng.normal(size=500) * 0.001 * (index + 1) for index in range(4)]
    )
    frame = pd.DataFrame(noisy, columns=pd.Index([f"c{index}" for index in range(4)]))
    hrp = hrp_weights(frame)
    inverse_variance = pd.Series(
        inverse_variance_weights(frame.cov().to_numpy(dtype=float)), index=frame.columns
    )

    def concentration(series: pd.Series) -> float:
        return float((series**2).sum())

    assert concentration(hrp) < concentration(inverse_variance)


def test_recursive_bisection_weights_follow_the_order() -> None:
    """权重与传入的 order 同序（调用方负责映射回原始资产顺序）。"""
    cov = np.diag([0.01, 0.04])
    weights = recursive_bisection(cov, [1, 0])
    assert weights[0] > weights[1]  # order 的第一位是下标 1，它方差大、应拿更少


def test_hrp_rejects_degenerate_input() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="至少需要两个资产"):
        hrp_weights(pd.DataFrame({"only": rng.normal(size=50)}))
    with pytest.raises(ValueError, match="至少需要两期"):
        hrp_weights(pd.DataFrame({"a": [0.01], "b": [0.02]}))
