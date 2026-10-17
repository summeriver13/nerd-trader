"""
NCO（嵌套聚类优化，de Prado 第 16 章）的不变量用例。

与 HRP 的用例分开放：HRP 钉的是「递归二分的顺序依赖」，NCO 钉的是「两层次最小方差的
恒等式」与两个退化情形（一簇、每资产一簇）。共用工具都在 src/research/backtest/allocation.py。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.backtest.allocation import (
    cluster_labels,
    covariance_pair,
    minimum_variance_weights,
    nco_weights,
)


def test_cluster_labels_partition_the_assets() -> None:
    """簇编号是 0..K−1 的一个划分，且近乎共线的两只必落进同一簇。"""
    rng = np.random.default_rng(23)
    common = rng.normal(size=400) * 0.01
    other = rng.normal(size=400) * 0.01
    # 两组各自共享一个共同因子（彼此独立）：簇结构才无歧义——b 组若彼此独立，
    # 单链接在「b₁b₂」与「跨簇」同量级的距离上会任意选一条，切分不稳定
    frame = pd.DataFrame(
        {
            "a1": common + rng.normal(size=400) * 0.0005,
            "a2": common + rng.normal(size=400) * 0.0005,
            "b1": other + rng.normal(size=400) * 0.0005,
            "b2": other + rng.normal(size=400) * 0.0005,
        }
    )
    correlation = frame.corr().to_numpy(dtype=float)
    for clusters in (1, 2, 3, 4):
        labels = cluster_labels(correlation, clusters)
        assert labels.shape == (4,)
        assert labels.min() == 0
        # fcluster 的 maxclust 只保证「至多」K 个簇，树结构不支持时会更少
        assert labels.max() <= min(clusters, 4) - 1
    two = cluster_labels(correlation, 2)
    assert two[0] == two[1]
    assert two[2] == two[3]
    assert two[0] != two[2]
    with pytest.raises(ValueError, match="至少为 1"):
        cluster_labels(correlation, 0)


def test_minimum_variance_weights_match_the_closed_form() -> None:
    """两只资产的最小方差解有闭式：`w₁ = (σ₂² − ρσ₁σ₂)/(σ₁² + σ₂² − 2ρσ₁σ₂)`。"""
    sigma_one, sigma_two, rho = 0.02, 0.03, 0.4
    cov = np.array(
        [
            [sigma_one**2, rho * sigma_one * sigma_two],
            [rho * sigma_one * sigma_two, sigma_two**2],
        ]
    )
    expected = (sigma_two**2 - rho * sigma_one * sigma_two) / (
        sigma_one**2 + sigma_two**2 - 2 * rho * sigma_one * sigma_two
    )
    weights = minimum_variance_weights(cov)
    assert weights[0] == pytest.approx(expected)
    assert weights[1] == pytest.approx(1.0 - expected)
    with pytest.raises(ValueError, match="必须是方阵"):
        minimum_variance_weights(np.zeros((2, 3)))


def test_nco_with_one_cluster_is_minimum_variance() -> None:
    """只有一簇时簇内权重就是全局最小方差、簇间权重为 1（退化情形可手算）。"""
    rng = np.random.default_rng(29)
    frame = pd.DataFrame(rng.normal(size=(500, 5)) * 0.01)
    covariance = frame.cov().to_numpy(dtype=float)
    weights = nco_weights(frame, clusters=1)
    assert weights.to_numpy() == pytest.approx(minimum_variance_weights(covariance))
    assert list(weights.index) == list(frame.columns)


def test_nco_with_each_asset_alone_is_minimum_variance() -> None:
    """每个资产自成一簇时，簇内权重全为 1、簇间解就是全局最小方差。"""
    rng = np.random.default_rng(37)
    frame = pd.DataFrame(rng.normal(size=(400, 4)) * 0.01)
    covariance = frame.cov().to_numpy(dtype=float)
    weights = nco_weights(frame, clusters=frame.shape[1])
    assert weights.to_numpy() == pytest.approx(minimum_variance_weights(covariance))


def test_nco_weights_sum_to_one() -> None:
    """权重和为 1（允许为负——两层次都解无约束最小方差，这是原文 optPort 的口径）。"""
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(rng.normal(size=(300, 6)) * 0.01)
    for clusters in (1, 2, 3, 6):
        weights = nco_weights(frame, clusters=clusters)
        assert float(weights.sum()) == pytest.approx(1.0)


def test_nco_two_clusters_matches_the_book_algorithm() -> None:
    """两簇时按原文手算一遍：簇内最小方差 × 簇间最小方差（簇级矩阵由簇内组合收缩）。

    这是 NCO 的核心恒等式，手写一遍能直接看出「只对小块求逆」发生在哪一步。
    """
    rng = np.random.default_rng(43)
    common = rng.normal(size=500) * 0.02
    other = rng.normal(size=500) * 0.02
    frame = pd.DataFrame(
        {
            "a1": common + rng.normal(size=500) * 0.002,
            "a2": common + rng.normal(size=500) * 0.003,
            "b1": other + rng.normal(size=500) * 0.004,
            "b2": other + rng.normal(size=500) * 0.006,
        }
    )
    correlation, covariance = covariance_pair(frame)
    labels = cluster_labels(correlation, 2)
    assert list(labels[:2]) == [labels[0], labels[0]]
    assert list(labels[2:]) == [labels[2], labels[2]]
    left = [index for index, label in enumerate(labels) if label == labels[0]]
    right = [index for index, label in enumerate(labels) if label != labels[0]]
    block_left = covariance[np.ix_(left, left)]
    block_right = covariance[np.ix_(right, right)]
    intra_left = minimum_variance_weights(block_left)
    intra_right = minimum_variance_weights(block_right)
    cross = covariance[np.ix_(left, right)]
    reduced = np.array(
        [
            [float(intra_left @ block_left @ intra_left), float(intra_left @ cross @ intra_right)],
            [float(intra_right @ cross.T @ intra_left), float(intra_right @ block_right @ intra_right)],
        ]
    )
    inter = minimum_variance_weights(reduced)
    expected = np.zeros(4, dtype=float)
    expected[left] = intra_left * inter[0]
    expected[right] = intra_right * inter[1]
    assert nco_weights(frame, clusters=2).to_numpy() == pytest.approx(expected)


def test_nco_rejects_degenerate_input() -> None:
    rng = np.random.default_rng(47)
    with pytest.raises(ValueError, match="至少需要两个资产"):
        nco_weights(pd.DataFrame({"only": rng.normal(size=50)}))
    with pytest.raises(ValueError, match="至少需要两期"):
        nco_weights(pd.DataFrame({"a": [0.01], "b": [0.02]}))
