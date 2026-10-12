"""Ch 8 的正交化与确认性秩相关（原文 Snippet 8.5 / 8.6）。

作者给 PCA 的定位是**确认性**的：把无监督的方差结构（PCA）与有监督的特征重要性
（MDI / MDA）各自排一次序，再看两者是否一致。一致说明模型抓到的是数据里**主导的
变异结构**（更可能是真实模式）；不一致说明模型在追自己的东西。

原文（Snippet 8.5）的做法是手搓的：标准化特征 → 对 `dfZ.T @ dfZ` 做谱分解 → 特征值
降序 → 保留累计解释方差**首次**达到 `variance_threshold` 的主成分 → `loadings`（原文
`eVec`）的行是**原始特征**、列是主成分，正交特征是 `dfZ @ eVec`。本仓库改用 sklearn 的
`PCA`（数学等价、少一段自己维护的线代），并保留原文的截断口径：`n_components` 传比例，
取**最小**的、使累计解释方差达到阈值的那几个主成分。

**原文没给的是这一步**：`pcRank`（每个特征的 PCA 排名）怎么算——Snippet 8.6 只给了默认值
`[1, 2, 4, 3]`。本仓库的定义是「保留子空间里的载荷加权」：

    pca_importance_j = Σ_k (λ_k / Σλ) × |loading_jk|

即「数据的主导变异方向压在这个特征上多少」。两个候选定义里选它是因为**判别力**：把载荷
**平方**再按 λ 加权（写成份额后所有特征求和为 1）看着更自然，实测却几乎没有判别力——
标准化之后每个特征自带 1/p 的方差，一份「5 个高度相关 + 1 个正交」的合成数据里六个
特征的份额全落在 0.1625 ~ 0.1737，极差只有均值的 2%；换成绝对载荷后同一份数据的极差是
均值的 26%，并且把「主导方差但无预测力」的那簇排到了会预测的 `signal` 前面——这正是这个
工具存在的意义。只保留一个主成分时，它退化成「按 PC1 载荷大小排序」（有用例钉住）。

**原文的一处参数陷阱**（照抄会静默改变语义）：Snippet 8.6 写的是
`weightedtau(featImp, pcRank**-1.)[0]`，而 scipy 的第二个**位置**参数不是权重、是布尔量
`rank`（是否先做秩变换）。`pcRank**-1.` 是个非空数组、恒为真，所以实际生效的是**秩变换**；
真正的加权来自 scipy 的默认 weigher——按重要性的位置做双曲加权 `1/(r+1)`，即「越重要
越算数」。这个语义本身合理，`rank_agreement` 就按它实现，并写在这里，免得后来者以为
`pcRank**-1.` 被当成了权重。

参考：López de Prado (2018) AFML Ch 8（Snippet 8.5 / 8.6）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.decomposition import PCA

DEFAULT_VARIANCE_THRESHOLD = 0.95
MIN_FEATURES_FOR_RANK = 3


@dataclass(frozen=True)
class OrthogonalFeatures:
    """PCA 正交化的结果（原文 Snippet 8.5）。

    `scores` 是正交特征（原文 `dfP`），`loadings` 是投影矩阵（原文 `eVec`：行=原始
    特征、列=主成分），两者都只含**保留下来**的主成分。载荷的正负号由 sklearn 的 SVD
    决定、本身没有含义（`pca_importance` 取绝对值）。
    """

    scores: pd.DataFrame
    loadings: pd.DataFrame
    eigenvalues: pd.Series
    explained: pd.Series


def orthogonal_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> OrthogonalFeatures:
    """对特征做 PCA 正交化，保留累计解释方差达到 `variance_threshold` 的主成分。

    阈值 1.0 表示**全保留**（原文口径允许，sklearn 的浮点模式只收开区间，这里自己映射）。
    标准化用样本标准差（`ddof=1`，与原文的 `dfX.std()` 一致）。常数特征无法定标、
    NaN/inf 会让分解失真，两者都直接报错而不是静默给出无意义的结果。
    """
    columns = list(features)
    if not columns:
        raise ValueError("至少需要一个特征")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("正交化需要有限的特征值（NaN/inf 会让分解结果无意义）")
    deviation = values.std(axis=0, ddof=1)
    if not (deviation > 0).all():
        raise ValueError("存在常数特征，无法标准化（它也没有可排序的方差）")
    centered = (values - values.mean(axis=0)) / deviation
    if variance_threshold <= 0:
        raise ValueError("variance_threshold 必须为正（累计解释方差的比例）")
    # sklearn 只收 (0, 1) 开区间的浮点；原文口径的 1.0 = 全保留，所以映射成 `None`。
    # 其余按比例交给 sklearn——它截断用的 `searchsorted(cumsum) + 1` 与原文逐字相同。
    keep: float | None = None if variance_threshold >= 1.0 else variance_threshold
    model = PCA(n_components=keep, svd_solver="full")
    scores = model.fit_transform(centered)
    # 桩把 components_ 标成「未拟合时为 None」，拟合后必为 ndarray，这里显式收窄
    # （仓库规则：不用 pyright: ignore 压掉）。
    loadings = cast(np.ndarray, model.components_).T
    names = pd.Index([f"PC_{index + 1}" for index in range(loadings.shape[1])])
    return OrthogonalFeatures(
        scores=pd.DataFrame(scores, index=frame.index, columns=names),
        loadings=pd.DataFrame(
            loadings, index=pd.Index(columns, name="feature"), columns=names
        ),
        eigenvalues=pd.Series(model.explained_variance_, index=names),
        explained=pd.Series(model.explained_variance_ratio_, index=names),
    )


def pca_importance(ortho: OrthogonalFeatures) -> pd.Series:
    """无监督重要性：保留子空间里按解释方差加权的**绝对载荷**之和。

    只看特征之间的协方差结构，完全不看标签。数值不对所有特征求和为 1（它不是份额，
    而是「主导方向压在这个特征上的总量」），只用来排序。
    """
    weights = (ortho.explained / ortho.explained.sum()).to_numpy(dtype=float)
    values = np.abs(ortho.loadings.to_numpy(dtype=float)) @ weights
    return pd.Series(values, index=ortho.loadings.index, name="pca_importance")


def unsupervised_rank(ortho: OrthogonalFeatures) -> pd.Series:
    """无监督排名（原文的 `pcRank`）：**1 = 重要性最高**，与原文默认值同向。"""
    score = pca_importance(ortho)
    order = np.argsort(-score.to_numpy(dtype=float), kind="stable")
    ranks = np.empty(score.size, dtype=int)
    ranks[order] = np.arange(1, score.size + 1)
    return pd.Series(ranks, index=score.index, name="pca_rank")


def rank_agreement(importance: pd.Series, rank: pd.Series) -> float:
    """原文 Snippet 8.6：有监督重要性对**逆** PCA 排名的加权 Kendall's tau。

    权重来自 scipy 默认的双曲 weigher（`1/(r+1)`，按重要性的位置算），所以「最重要的
    那几个特征排对了没有」占的权重更大。取值 −1 ~ +1，越大说明模型的重要性排序越贴近
    数据的方差结构。

    两个向量按标签对齐；特征少于 3 个时秩相关恒为 ±1、没有信息量，直接报错。
    """
    aligned = pd.concat({"importance": importance, "rank": rank}, axis=1)
    if bool(aligned.isna().to_numpy().any()):
        raise ValueError("重要性或排名里有 NaN（两边标签未对齐？），无法比较秩")
    if len(aligned) < MIN_FEATURES_FOR_RANK:
        raise ValueError(f"秩相关至少需要 {MIN_FEATURES_FOR_RANK} 个特征")
    # scipy 运行时返回具名元组 SignificanceResult(statistic, pvalue)，但本版本的桩把它推成
    # 未知类型，这里显式收窄（仓库规则：不用 pyright: ignore 压掉）。
    statistic = cast(
        float,
        weightedtau(
            aligned["importance"].to_numpy(dtype=float),
            1.0 / aligned["rank"].to_numpy(dtype=float),
        )[0],
    )
    return float(statistic)
