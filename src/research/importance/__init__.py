"""特征重要性（de Prado Ch 8）：MDI 有偏、MDA 会被替代效应骗、聚类后的 MDA 才稳。

- `tools.py`：四种工具本身（MDI / MDA / 聚类 MDA / SFI）与共用的折上评估——**口径与
  各自的坑写在那里的模块文档串里，改代码前先读**。
- `table.py`：把几种读数合成**一张表**（单位不同不能比大小，但并排看分歧就是信息）。
- `pca.py`：Ch 8 的正交化与**确认性**秩相关（原文 Snippet 8.5 / 8.6）——无监督的方差
  排名与有监督的重要性排序一致到什么程度。

公开 API 从本包直接取：`importance.importance_table(...)`。
"""

from src.research.importance.pca import (
    OrthogonalFeatures,
    orthogonal_features,
    pca_importance,
    rank_agreement,
    unsupervised_rank,
)
from src.research.importance.table import importance_table
from src.research.importance.tools import (
    DEFAULT_CLUSTER_DISTANCE,
    IMPORTANCE_COLUMNS,
    block_log_loss,
    feature_clusters,
    impurity_importance,
    mean_decrease_accuracy,
    single_feature_importance,
    summarise_folds,
)
from src.research.labeling.meta import FoldFit

__all__ = [
    "DEFAULT_CLUSTER_DISTANCE",
    "IMPORTANCE_COLUMNS",
    "FoldFit",
    "OrthogonalFeatures",
    "orthogonal_features",
    "pca_importance",
    "rank_agreement",
    "unsupervised_rank",
    "block_log_loss",
    "feature_clusters",
    "importance_table",
    "impurity_importance",
    "mean_decrease_accuracy",
    "single_feature_importance",
    "summarise_folds",
]
