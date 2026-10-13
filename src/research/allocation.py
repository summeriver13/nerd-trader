"""
分层风险平价（de Prado 第 16 章）：先聚类、再在树的每一层做风险预算，不用协方差求逆。

马科维茨的解是 `w ∝ Σ⁻¹μ`，而 Σ 的条件数在有相关簇时极差：簇内资产近乎共线，
估计误差在求逆时被放大成极端权重（作者的「马科维茨诅咒」）。HRP 的对策是**把协方差
矩阵变成一棵树**：先把相近的资产聚成簇，再自顶向下按簇的方差分配权重——全程不碰逆矩阵。

原文口径（Snippet 16.4 全套，逐条核对）：

- **距离**：`d(i,j) = √((1 − ρ(i,j)) / 2)`——这是原文给的把相关变成度量的式子
  （不是 `1 − ρ`，两者的差别在簇边界附近会挑出不同的树）。本仓与
  `importance.tools.feature_clusters` 共用同一个实现。
- **聚类**：`scipy.cluster.hierarchy.linkage(squareform(d), method="single")`。
- **准对角化**（`getQuasiDiag`）：把树按叶子顺序展开成一维排列，使同一簇的资产相邻。
- **递归二分**（`getRecBipart`）：反复把当前簇对半切，`alpha = 1 − Var₀/(Var₀+Var₁)`，
  两个子簇分别乘 `alpha` 与 `1−alpha`。**簇方差用簇内逆方差组合算**（`getClusterVar`），
  不是等权——这是原文最容易被抄错的一处。
- **逆方差组合**（`getIVP`）：`w ∝ 1/diag(Σ)`，是 HRP 的退化情形（不聚类），
  本仓留作对照基线。

已知限制：HRP 只回答「给多少权重」，不回答「该不该持有」——它没有预期收益这一项，
所以拿它做多空组合时，方向必须由别的信号（本仓是因子值）给定。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# 单链接聚类：与原文一致，也是「链式相关」最自然的度量
LINKAGE_METHOD: Final[str] = "single"
# 方差必须为正：逆方差组合在 0 方差上会发散，宁可在入口拦住
MIN_VARIANCE: Final[float] = 0.0
# NCO 默认簇数：七个策略切成三簇（原文 maxClusters 是个可调参数，
# scripts/cpcv_study.py 会扫 2~6 报整个平台）
DEFAULT_CLUSTERS: Final[int] = 3


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """把相关矩阵变成度量矩阵 `√((1 − ρ)/2)`，对角为 0。

    nan / inf 先归零（缺失相关的保守处理是「当作不相关」），再夹到 [0, 1] 以免
    浮点误差让 `ρ` 略超 1 时开根号出 nan。
    """
    clean = np.nan_to_num(
        np.asarray(corr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    distance = np.sqrt(np.clip((1.0 - clean) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def single_linkage(corr: np.ndarray) -> np.ndarray:
    """相关矩阵 → 层次聚类树（scipy 的 linkage 矩阵）。"""
    distance = correlation_distance(corr)
    return linkage(squareform(distance, checks=False), method=LINKAGE_METHOD)


def quasi_diagonal(link: np.ndarray, items: int | None = None) -> list[int]:
    """准对角化：把树展开成一维排列，使同一簇的资产相邻（原文 getQuasiDiag）。

    `link` 是 scipy 的 linkage 矩阵；`items` 是叶子数（默认从矩阵里读）。
    返回原始资产的下标序列。原文用迭代插入展开，本仓写成语义相同的深度优先
    遍历（左孩子在前）——占用栈而非逐次重排 Series，等价性由用例对着原文算法鞂死。
    """
    tree = np.asarray(link, dtype=float).astype(int)
    if tree.ndim != 2 or tree.shape[0] == 0:
        raise ValueError("linkage 矩阵不能为空")
    leaves = int(tree[-1, 3]) if items is None else int(items)
    # scipy 约定：第 i 行把两个簇合并成编号 leaves + i 的簇，故根编号如下
    stack: list[int] = [leaves + tree.shape[0] - 1]
    order: list[int] = []
    while stack:
        node = stack.pop()
        if node < leaves:
            order.append(int(node))
            continue
        row = tree[node - leaves]
        stack.append(int(row[1]))
        stack.append(int(row[0]))
    return order


def covariance_pair(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """收益表 →（相关矩阵，协方差矩阵），按 `returns.columns` 顺序，含缺失检查。

    HRP 与 NCO 共用同一份校验：全空行先丢掉，相关或协方差含 nan 时直接抛错——
    相关矩阵估不出来时给「看起来像答案」的等权是最坏的结果。
    """
    frame = returns.dropna(how="all")
    correlation = frame.corr().to_numpy(dtype=float)
    covariance = frame.cov().to_numpy(dtype=float)
    if not np.isfinite(correlation).all():
        remaining = frame.dropna(axis=1)
        if remaining.shape[1] < 2:
            raise ValueError("收益表里的缺失太多，相关矩阵估不出来")
        raise ValueError("相关矩阵含 nan：请先去掉全空的资产或补数据")
    if not np.isfinite(covariance).all():
        raise ValueError("协方差矩阵含 nan：请先去掉全空的资产或补数据")
    return correlation, covariance


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """逆方差组合 `w ∝ 1/diag(Σ)`（原文 getIVP）。对角线非正时抛错。"""
    matrix = np.asarray(cov, dtype=float)
    variances = np.diag(matrix)
    if variances.size == 0 or np.any(variances <= MIN_VARIANCE):
        raise ValueError("协方差对角线必须严格为正——逆方差组合在 0 方差上发散")
    inverse = 1.0 / variances
    return inverse / float(inverse.sum())


def cluster_variance(cov: np.ndarray, items: Sequence[int]) -> float:
    """簇方差：先算簇内逆方差组合，再算组合方差（原文 getClusterVar）。"""
    matrix = np.asarray(cov, dtype=float)
    indices = np.asarray(list(items), dtype=int)
    block = matrix[np.ix_(indices, indices)]
    weights = inverse_variance_weights(block)
    return float(weights @ block @ weights)


def cluster_labels(corr: np.ndarray, clusters: int) -> np.ndarray:
    """把层次聚类树切成 `clusters` 个簇，返回每个资产（原序）的 0 基簇编号。

    簇数不小于资产数时每个资产自成一簇（退化情形由用例钉住）。
    """
    matrix = np.asarray(corr, dtype=float)
    assets = matrix.shape[0]
    if clusters < 1:
        raise ValueError("簇数至少为 1")
    if clusters >= assets:
        return np.arange(assets)
    link = single_linkage(matrix)
    labels = fcluster(link, t=clusters, criterion="maxclust")
    return np.asarray(labels, dtype=int) - 1


def minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """解析最小方差解 `w ∝ Σ⁻¹1`（原文 optPort，**不约束做空**）。

    只在小块上求逆正是 NCO 与「直接对全矩阵求逆」的区别：本仓实测的七策略协方差
    条件数中位 29、最大 53，正常求解即可，不额外加 ridge（矩阵真的奇异时让它抛错，
    而不是偷偷改问题）。
    """
    matrix = np.asarray(cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("协方差矩阵必须是方阵")
    solution = np.linalg.solve(matrix, np.ones(matrix.shape[0], dtype=float))
    total = float(solution.sum())
    if total == 0.0:
        raise ValueError("最小方差解退化成全零——协方差矩阵不可用")
    return solution / total


def recursive_bisection(cov: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """递归二分（原文 getRecBipart）：按准对角顺序自顶向下分配，权重和为 1。

    返回的权重与 `order` 同序——调用方负责映射回原始资产顺序。
    """
    sequence = [int(index) for index in order]
    if not sequence:
        return np.zeros(0, dtype=float)
    weights = np.ones(len(sequence), dtype=float)
    clusters: list[list[int]] = [sequence]
    while clusters:
        bisected: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) > 1:
                half = len(cluster) // 2
                bisected.append(cluster[:half])
                bisected.append(cluster[half:])
        clusters = bisected
        for position in range(0, len(clusters), 2):
            left = clusters[position]
            right = clusters[position + 1]
            left_variance = cluster_variance(cov, left)
            right_variance = cluster_variance(cov, right)
            total = left_variance + right_variance
            # 方差大的一支拿更少权重；两支方差都为 0（常数序列）时退化成等分
            alpha = 0.5 if total <= 0.0 else 1.0 - left_variance / total
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """分层风险平价权重，按 `returns.columns` 顺序返回。

    `returns` 是逐期收益的宽表（行=期、列=资产）。样本太短或列数不足两个时抛错——
    相关矩阵估不出来时给「看起来像答案」的等权是最坏的结果。
    """
    if returns.shape[1] < 2:
        raise ValueError("HRP 至少需要两个资产")
    if returns.shape[0] < 2:
        raise ValueError("HRP 至少需要两期收益")
    correlation, covariance = covariance_pair(returns)
    order = quasi_diagonal(single_linkage(correlation))
    weights = np.zeros(len(order), dtype=float)
    weights[np.asarray(order, dtype=int)] = recursive_bisection(covariance, order)
    return pd.Series(weights, index=returns.columns, dtype=float)


def nco_weights(
    returns: pd.DataFrame, clusters: int = DEFAULT_CLUSTERS
) -> pd.Series:
    """嵌套聚类优化（原文 Snippet 16.5）：先簇内求最优，再在簇之间求最优。

    两个层次都解最小方差（`optPort`，允许做空策略），只对簇内小块与簇级矩阵求逆——
    这正是「马科维茨诅咒」的对策：全矩阵求逆会把相关簇的估计误差放大成极端权重。

    簇级矩阵用簇内组合把协方差收缩：`Σ̃(i,j) = wᵢ' Σ(i,j) wⱼ`（i=j 时就是簇方差），
    再对 `Σ̃` 解最小方差，最后 `w = 簇内权重 × 簇间权重`。
    """
    if returns.shape[1] < 2:
        raise ValueError("NCO 至少需要两个资产")
    if returns.shape[0] < 2:
        raise ValueError("NCO 至少需要两期收益")
    correlation, covariance = covariance_pair(returns)
    labels = cluster_labels(correlation, clusters)
    count = int(labels.max()) + 1
    blocks: list[list[int]] = []
    intra_blocks: list[np.ndarray] = []
    intra = np.zeros(returns.shape[1], dtype=float)
    for index in range(count):
        rows = [position for position, label in enumerate(labels) if label == index]
        block = covariance[np.ix_(rows, rows)]
        weights = minimum_variance_weights(block)
        intra[rows] = weights
        blocks.append(rows)
        intra_blocks.append(weights)
    reduced = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left, count):
            cross = covariance[np.ix_(blocks[left], blocks[right])]
            value = float(intra_blocks[left] @ cross @ intra_blocks[right])
            reduced[left, right] = value
            reduced[right, left] = value
    inter = minimum_variance_weights(reduced)
    return pd.Series(intra * inter[labels], index=returns.columns, dtype=float)
