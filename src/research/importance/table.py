"""把 Ch 8 几种工具的读数合成一张表。

四种读数单位不同、**不能直接比大小**：MDA 与聚类 MDA 是「打乱后的损失增加」，MDI 是
「样本内不纯度下降的份额」（各折加起来为 1），SFI 是「单特征模型相对常数预测器的改进」。
但把**排序并排**放在一行里，分歧就变成了信息：

- 一个特征 MDA 高、SFI 也高 → 它自己就带信息；
- MDA 低但 SFI 高 → 它在 MDA 里被替身顶掉了（簇的联合 MDA 会把它救回来）；
- MDI 高但 MDA 低 → 模型用了它，但它没带来泛化能力（MDI 是样本内的）。
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.research.importance.tools import (
    DEFAULT_CLUSTER_DISTANCE,
    feature_clusters,
    impurity_importance,
    mean_decrease_accuracy,
    single_feature_importance,
)
from src.research.meta import FoldFit, MetaLabelOptions


def importance_table(
    fits: Sequence[FoldFit],
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    options: MetaLabelOptions,
    threshold: float = DEFAULT_CLUSTER_DISTANCE,
) -> pd.DataFrame:
    """一张表读完：逐特征 MDA、它所在的簇、簇的联合 MDA、MDI、以及 SFI。

    四种工具放在一起，是因为**它们的分歧才是信息**：MDI 是样本内的，连噪声特征都会
    分到重要性；逐特征 MDA 会被替代效应稀释（有无替身差别可能上倍）；簇的联合 MDA
    才是「这一块信息到底值多少」；SFI 每个特征单独建模型，**天然免疫替代效应**但
    看不到联合效应（`sfi_skill` 是相对常数预测器的改进）。表按逐特征 MDA 降序。
    """
    clusters = feature_clusters(frame, features, threshold=threshold)
    per_feature = mean_decrease_accuracy(fits, frame, features)
    grouped = mean_decrease_accuracy(fits, frame, features, groups=clusters)
    mdi = impurity_importance(fits, features)
    single = single_feature_importance(fits, frame, features, options=options)
    cluster_of = {
        member: name for name, members in clusters.items() for member in members
    }
    names = [str(name) for name in per_feature.index]
    return pd.DataFrame(
        {
            "feature": names,
            "mda_logloss": [float(per_feature.loc[name, "mean"]) for name in names],
            "mda_t": [float(per_feature.loc[name, "t"]) for name in names],
            "cluster": [cluster_of[name] for name in names],
            "cluster_members": [",".join(clusters[cluster_of[name]]) for name in names],
            "cluster_logloss": [
                float(grouped.loc[cluster_of[name], "mean"]) for name in names
            ],
            "cluster_t": [float(grouped.loc[cluster_of[name], "t"]) for name in names],
            "mdi_gini": [float(mdi[name]) for name in names],
            "sfi_skill": [float(single.loc[name, "skill"]) for name in names],
            "sfi_t": [float(single.loc[name, "skill_t"]) for name in names],
            "sfi_logloss": [float(single.loc[name, "logloss"]) for name in names],
        }
    )
