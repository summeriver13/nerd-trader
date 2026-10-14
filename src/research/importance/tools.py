"""Ch 8 的四种特征重要性工具，以及它们共用的「折上评估」。

作者在 Ch 8 开门见山的一句话是「回测不是研究工具，特征重要性才是」——他反对用
回测挑特征（那是过拟合的温床），主张先问「模型到底在用什么信息」。本章的几个工具在
本仓库已有的元标签模型上都能直接跑，而且它们的**分歧本身**就是信息：

1. **MDI**（`impurity_importance`）：森林自己的 `feature_importances_`，各折平均。
   它是**样本内**的，且天然偏向高基数特征——连纯噪声特征都会分到一点重要性。只当参考。
2. **MDA**（`mean_decrease_accuracy`）：**样本外**。拿一折已拟合的模型，在**测试折**上
   先算基准 `log_loss`，再把某一列打乱重算，**损失的增加就是重要性**。必须用
   purged/embargo 的折，否则泄漏会把重要性吹起来。
3. **聚类 MDA**：MDA 的致命伤是**替代效应**——两个高度相关的特征互为替身，只打乱其中一个
   时另一个还在，损失几乎不变，于是**两个都被报成不重要**。修法是把它们当成**一个簇**
   一起打乱（作者在《Machine Learning for Asset Managers》里叫 Clustered MDA），
   读到的就是整块信息的**联合**重要性。
4. **SFI**（`single_feature_importance`）：**每个特征单独建一个模型**，看它自己的样本外
   表现。按定义**没有替代效应**（每个特征都有自己的模型，不存在被替身顶掉），代价是
   **看不到联合效应**——只在组合里才有预测力的特征（如异或）在这里读数是 0。

三条口径（改代码前先读）：

1. **符号统一为「打乱后损失增加 = 正数 = 重要」。** 打乱一个有用的列会让模型变差、
   损失上升，所以重要性为正；不重要（可被替代或本就无用）的列接近 0，噪声列甚至可能为负。
2. **打乱发生在测试折内部**，且只打乱列、不动行——这是 MDA 的常规做法（Breiman 的
   permutation importance）。代价是它构造的是「把这一列重排」这个反事实，而不是一个
   现实场景；这是 MDA 的固有局限，不是实现细节。
3. **簇的划分来自特征相关性**，与标签无关（无监督），所以聚类本身不会带来前视。
   距离用作者的相关性距离 `sqrt((1 - corr) / 2)`，阈值 0.5 对应两两相关约 0.5。

参考：López de Prado (2018) AFML Ch 8；Clustered Feature Importance。被有意跳过的
工具：PCA 正交（无监督 PCA 排序与有监督 MDI 排序的 Weighted Kendall's Tau，是
确认性程序而非独立方法），原文口径已核到，排在 SFI 之后做。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from src.research.allocation import single_linkage
from src.research.meta import (
    LABEL_COLUMN,
    WEIGHT_COLUMN,
    FoldFit,
    MetaLabelOptions,
    fit_classifier,
)
from src.research.tuning import Classifier

DEFAULT_CLUSTER_DISTANCE = 0.5
IMPORTANCE_COLUMNS = ("mean", "std", "t", "folds", "members")


def feature_clusters(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    threshold: float = DEFAULT_CLUSTER_DISTANCE,
) -> dict[str, tuple[str, ...]]:
    """按相关性把特征聚成簇，返回 {簇名: (特征, ...)}。

    单链接层次聚类。`threshold` 是相关性距离的切分高度：0.5 表示「两两相关 ≥ 0.5 的
    特征算同一簇」。特征少于两个时不聚类，各自成簇。
    """
    columns = list(features)
    if len(columns) < 2:
        return {f"c{i + 1}": (name,) for i, name in enumerate(columns)}
    values = frame[columns].to_numpy(dtype=float)
    corr = np.nan_to_num(
        np.corrcoef(values, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0
    )
    np.fill_diagonal(corr, 1.0)
    # 距离与聚类口径与 HRP（第 16 章）共用一份实现，详见 allocation.py 的模块文档
    tree = single_linkage(corr)
    labels = fcluster(tree, t=threshold, criterion="distance")
    grouped: dict[str, list[str]] = {}
    for name, label in zip(columns, labels, strict=True):
        grouped.setdefault(f"c{int(label)}", []).append(name)
    return {key: tuple(value) for key, value in grouped.items()}


def block_log_loss(
    model: Classifier,
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> float:
    """模型在这批特征上的加权 log loss（越小越好）。

    参数类型是 `Classifier`（只需 `fit` + `predict_proba`）而不是随机森林：MDA/SFI 要的
    只是一个能给出概率的估计器，手写袋化集成（`research.bagging`）也要走这条路。
    """
    probability = np.asarray(model.predict_proba(values))
    return float(log_loss(target, probability, sample_weight=weight, labels=[0, 1]))


def _shuffle_block(
    values: np.ndarray,
    positions: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """把指定的若干列各自打乱（列内重排），其余列原样。"""
    shuffled = values.copy()
    for position in positions:
        shuffled[:, position] = rng.permutation(shuffled[:, position])
    return shuffled


def summarise_folds(
    importances: Mapping[str, list[float]],
    members: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """把逐折读数汇总成 mean / std / t / folds，按平均值降序。

    `members` 是每个名字包含哪些特征（逐特征时就是它自己，逐簇时是一整簇）。
    """
    names = list(importances)
    means = np.zeros(len(names))
    stds = np.zeros(len(names))
    ts = np.zeros(len(names))
    counts = np.zeros(len(names), dtype=int)
    for index, name in enumerate(names):
        array = np.asarray(importances[name], dtype=float)
        mean = float(array.mean()) if array.size else 0.0
        std = float(array.std(ddof=1)) if array.size > 1 else 0.0
        stderr = std / np.sqrt(array.size) if array.size else 0.0
        means[index] = mean
        stds[index] = std
        ts[index] = mean / stderr if stderr > 0 else 0.0
        counts[index] = int(array.size)
    order = [int(index) for index in np.argsort(-means, kind="stable")]
    return pd.DataFrame(
        {
            "mean": means[order],
            "std": stds[order],
            "t": ts[order],
            "folds": counts[order],
            "members": [",".join(members[names[index]]) for index in order],
        },
        index=pd.Index([names[index] for index in order], name="name"),
    )


def mean_decrease_accuracy(
    fits: Sequence[FoldFit],
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """MDA / 聚类 MDA：打乱一列（或一整个簇）后样本外 log loss 的增加。

    `fits` 来自 `meta.walk_forward_meta(..., collect_fits=True)`，即每折的模型与
    它在**测试折**上的行标签——重要性与元标签因此讲的是同一个模型、同一批折。

    `groups` 为空时逐特征打乱；传入 `feature_clusters(...)` 的结果则是逐簇打乱
    （Clustered MDA，抗替代效应）。返回的索引是特征名或簇名，`members` 列写出后者
    含哪些特征。

    **打乱的随机数在每个折内推进**，所以同一 `seed` 下结果可复现。
    """
    columns = tuple(features)
    use_groups: dict[str, tuple[str, ...]] = (
        {str(key): tuple(value) for key, value in groups.items()}
        if groups is not None
        else {name: (name,) for name in columns}
    )
    positions = {name: index for index, name in enumerate(columns)}
    for name, members in use_groups.items():
        unknown = [member for member in members if member not in positions]
        if unknown:
            raise ValueError(f"簇 {name} 含未知特征：{unknown}")
        if not members:
            raise ValueError(f"簇 {name} 是空的")
    rng = np.random.default_rng(seed)
    importances: dict[str, list[float]] = {name: [] for name in use_groups}
    for fit in fits:
        test = frame.loc[fit.rows]
        values = test[list(columns)].to_numpy(dtype=float)
        target = test[LABEL_COLUMN].to_numpy()
        weight = test[WEIGHT_COLUMN].to_numpy(dtype=float)
        baseline = block_log_loss(fit.model, values, target, weight)
        for name, members in use_groups.items():
            shuffled = _shuffle_block(
                values, [positions[member] for member in members], rng
            )
            importances[name].append(
                block_log_loss(fit.model, shuffled, target, weight) - baseline
            )
    return summarise_folds(importances, use_groups)


def _constant_log_loss(
    train: pd.DataFrame, target: np.ndarray, weight: np.ndarray
) -> float:
    """只报基准率的常数预测器在测试折上的 log loss（SFI 的参照线）。

    基准率取自**训练折**——常数预测器也是模型，它的参数同样只能看训练数据。
    """
    rate = float(
        np.average(
            train[LABEL_COLUMN].to_numpy(),
            weights=train[WEIGHT_COLUMN].to_numpy(dtype=float),
        )
    )
    probability = np.column_stack(
        [np.full(target.size, 1.0 - rate), np.full(target.size, rate)]
    )
    return float(log_loss(target, probability, sample_weight=weight, labels=[0, 1]))


def single_feature_importance(
    fits: Sequence[FoldFit],
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    options: MetaLabelOptions,
) -> pd.DataFrame:
    """SFI：每个特征**单独**建一个模型，测它的样本外表现。

    用元模型**同一个分类器、同一批折**（`meta.fit_classifier`），只把特征集换成单个
    特征，所以它与 MDA / MDI 的分歧可以归因到特征，而不是模型。

    - **优点：按定义没有替代效应。** 每个特征都有自己的模型，不存在「被替身顶掉」，
      一个特征再冗余也不会牵连另一个的读数。这是它比逐特征 MDA 强的地方。
    - **缺点：看不到联合效应。** 只在组合里才有预测力的特征（如异或）在这里是 0。

    `skill` 是相对「只报训练折基准率的常数预测器」的改进（正数 = 比什么都不看更好），
    `skill_t` 是逐折 skill 的均值除以标准误。按 skill 降序。
    """
    columns = list(features)
    losses: dict[str, list[float]] = {name: [] for name in columns}
    skills: dict[str, list[float]] = {name: [] for name in columns}
    for fit in fits:
        train = frame.loc[fit.train_rows]
        test = frame.loc[fit.rows]
        target = test[LABEL_COLUMN].to_numpy()
        weight = test[WEIGHT_COLUMN].to_numpy(dtype=float)
        baseline = _constant_log_loss(train, target, weight)
        for name in columns:
            model = fit_classifier(train, [name], options)
            logloss = block_log_loss(
                model, test[[name]].to_numpy(dtype=float), target, weight
            )
            losses[name].append(logloss)
            skills[name].append(baseline - logloss)
    summary = summarise_folds(skills, {name: (name,) for name in columns})
    table = summary.rename(columns={"mean": "skill", "t": "skill_t"})
    table["logloss"] = [float(np.mean(losses[str(name)])) for name in table.index]
    return cast(pd.DataFrame, table[["skill", "skill_t", "logloss", "std", "folds"]])


def impurity_importance(
    fits: Sequence[FoldFit],
    features: Sequence[str],
) -> pd.Series:
    """MDI：森林自己的 `feature_importances_`，各折平均。

    样本内、且偏向高基数特征，所以只能当参考——它与 MDA 的分歧才是要看的东西。
    """
    columns = list(features)
    if not fits:
        return pd.Series(np.zeros(len(columns)), index=columns, name="mdi")
    # 只有树模型有 MDI；本仓的流水线喂进来的都是森林（含手写袋化森林）
    values = np.asarray(
        [cast(RandomForestClassifier, fit.model).feature_importances_ for fit in fits],
        dtype=float,
    )
    return pd.Series(values.mean(axis=0), index=columns, name="mdi")
