"""
袋化集成（de Prado Ch 6 §6.2–6.3）：把顺序 bootstrap 换进 bagging。

书的路线是两章接力：Ch 4 给采样器（标签重叠时均匀 bootstrap 反复抽到的是**重复信息**），
Ch 6 把采样器装进集成——随机森林的袋子默认按均匀 bootstrap 抽，作者在 random forest
一段里点明这一步可以换成顺序 bootstrap。本模块是那一步的落地，并配一个**同骨架、只换
袋子**的均匀对照臂：没有它，「顺序袋更好」与「手写的实现更好」分不开。

三条口径（改代码前先读）：

1. **袋子按标的分层抽。** 唯一性与并发必须在标的内部算（`t1` 是标的内的根序号，长表
   行号是另一套坐标，混用会静默错位——`observations` 里记过）。所以每只标的按自己的
   根坐标抽一批，再拼成一袋。
2. **袋长 = 该标的自己的标签数**（有放回），与 `scripts/bootstrap_study.py` 同口径。
3. **骨架与 sklearn 的随机森林同构**：每袋一棵树、按袋内行拟合（可带样本权重）、概率
   取平均、特征子抽样同样走 `max_features`。差别只有袋子的抽法。

`BaggedClassifier` 是不可变的，`fit` 返回新实例，所以 `meta.walk_forward_meta` 的
`fit` 扩展点（按折造模型）拿到的每折模型互不共享状态。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from src.research import bootstrap, meta
from src.research.tuning import Classifier

SAMPLER_UNIFORM: Final[str] = "uniform"
SAMPLER_SEQUENTIAL: Final[str] = "sequential"
SAMPLERS: Final[tuple[str, ...]] = (SAMPLER_UNIFORM, SAMPLER_SEQUENTIAL)

GROUP_COLUMN: Final[str] = "bag_group"
START_COLUMN: Final[str] = "bag_start"
END_COLUMN: Final[str] = "bag_end"
SIZE_COLUMN: Final[str] = "bag_size"
"""跨度坐标在帧上的列名：标的编号、起始根、结束根（含端点）、该标的的总根数。"""

SEED_UPPER_BOUND: Final[int] = 2**31 - 1
"""逐标的派生种子时 `Generator.integers` 的上界（必须 ≤ 2**32）。"""

FitFunction = Callable[[pd.DataFrame, list[str], meta.MetaLabelOptions], Classifier]
"""`meta.walk_forward_meta` 的拟合扩展点签名。"""


@dataclass(frozen=True)
class BaggingOptions:
    """集成超参。字段与 `meta.MetaLabelOptions` 的树参数一一对应——本研究里由它派生，
    保证各对照臂共用同一套超参。"""

    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 50
    max_features: float | str | None = "sqrt"
    seed: int = 0
    n_jobs: int = -1

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("至少要有一棵树")
        if self.max_depth < 1:
            raise ValueError("树深至少 1")
        if self.min_samples_leaf < 1:
            raise ValueError("叶节点样本数至少 1")


@dataclass(frozen=True)
class SpanCoordinates:
    """每行标签在自己标的根坐标里的位置——袋子按它抽，唯一性也按它算。

    `sizes[g]` 是标的 `g` 的**总根数**，不是它在训练帧里的行数：顺序 bootstrap 的并发
    要按标的的完整根轴算，训练帧只是其中被用到的那部分行。
    """

    groups: np.ndarray
    sizes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    def __post_init__(self) -> None:
        rows = int(self.groups.size)
        if self.starts.size != rows or self.ends.size != rows:
            raise ValueError("groups/starts/ends 必须逐行对齐")
        if rows and (
            int(self.groups.min()) < 0 or int(self.groups.max()) >= self.sizes.size
        ):
            raise ValueError("groups 超出 sizes 的标的编号范围")
        if rows and (self.starts.min() < 0 or self.ends.min() < 0):
            raise ValueError("根序号不能为负")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> SpanCoordinates:
        """从训练帧的四列读跨度坐标（列名见模块常量）。

        同一标的的所有行必须给出同一个 `SIZE_COLUMN`；这里取每个标的的第一行，
        所以上游写错会让并发被低估、顺序袋静默退化成均匀袋——`scripts/bagging_study.py`
        的构造函数是全仓唯一的写入点。
        """
        groups = frame[GROUP_COLUMN].to_numpy(dtype=np.int64)
        first = frame.drop_duplicates([GROUP_COLUMN])
        size_of = dict(
            zip(
                first[GROUP_COLUMN].to_numpy(dtype=np.int64),
                first[SIZE_COLUMN].to_numpy(dtype=np.int64),
                strict=True,
            )
        )
        width = max(size_of) + 1 if size_of else 0
        sizes = np.zeros(width, dtype=np.int64)
        for group, size in size_of.items():
            sizes[int(group)] = int(size)
        return cls(
            groups=groups,
            sizes=sizes,
            starts=frame[START_COLUMN].to_numpy(dtype=np.int64),
            ends=frame[END_COLUMN].to_numpy(dtype=np.int64),
        )

    def local_draws(
        self, sampler: str, *, seed: int
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """逐标的抽一袋：返回 (标的编号, 该标的的行下标, 袋内**局部**下标)。

        每只标的抽 `length = 它自己的行数` 次，与 sklearn 的 `bootstrap=True` 同量级；
        种子按标的派生，标的之间互不干扰、同一颗种子完全可复现。
        """
        if sampler not in SAMPLERS:
            raise ValueError(f"未知袋子抽法：{sampler}（可选 {SAMPLERS}）")
        rng = np.random.default_rng(seed)
        draws: list[tuple[int, np.ndarray, np.ndarray]] = []
        for group in range(int(self.sizes.size)):
            rows = np.flatnonzero(self.groups == group)
            if rows.size == 0:
                continue
            local_seed = int(rng.integers(0, SEED_UPPER_BOUND))
            if sampler == SAMPLER_SEQUENTIAL:
                local = bootstrap.sequential_bootstrap(
                    int(self.sizes[group]),
                    self.starts[rows],
                    self.ends[rows],
                    int(rows.size),
                    seed=local_seed,
                )
            else:
                local = bootstrap.uniform_bootstrap(
                    int(rows.size), int(rows.size), seed=local_seed
                )
            draws.append((int(group), rows, local))
        return draws

    def draw(self, sampler: str, *, seed: int) -> np.ndarray:
        """一袋的**全局行下标**（可重复，总长度等于训练行数）。"""
        pieces = [
            rows[local] for _, rows, local in self.local_draws(sampler, seed=seed)
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)

    def uniqueness_by_group(self, sampler: str, *, seed: int) -> pd.Series:
        """逐标的的袋平均唯一性（Ch 4 的量）——把采样器的读数接到集成读数上。"""
        values = {
            group: bootstrap.bag_uniqueness(
                int(self.sizes[group]), self.starts[rows], self.ends[rows], local
            )
            for group, rows, local in self.local_draws(sampler, seed=seed)
        }
        return pd.Series(values, dtype=float)


def tree_seed(index: int, seed: int) -> int:
    """第 `index` 棵树的种子（由 `SeedSequence` 派生，跨平台可复现）。

    公开是因为「逐袋重拟合」的参考实现要用同一批种子：等价性用例必须能复现它。
    """
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _fit_bagged_tree(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    spans: SpanCoordinates,
    sampler: str,
    index: int,
    options: BaggingOptions,
) -> DecisionTreeClassifier:
    """抽这一棵树的袋子并拟合它。

    抽袋子是本研究最贵的一步（逐标的、逐根顺序采样），所以它与拟合一起进并行任务：
    先串行把所有袋子抽完再并行拟合，等于把最贵的部分留在单线程上。
    """
    seed = tree_seed(index, options.seed)
    rows = spans.draw(sampler, seed=seed)
    tree = DecisionTreeClassifier(
        max_depth=options.max_depth,
        min_samples_leaf=options.min_samples_leaf,
        max_features=options.max_features,
        random_state=seed,
    )
    tree.fit(
        values[rows],
        target[rows],
        sample_weight=None if weight is None else weight[rows],
    )
    return tree


@dataclass(frozen=True)
class BaggedClassifier:
    """手写袋化集成：骨架与随机森林同构，只把袋子换成顺序（或均匀）bootstrap。"""

    options: BaggingOptions
    spans: SpanCoordinates
    sampler: str = SAMPLER_SEQUENTIAL
    trees: tuple[DecisionTreeClassifier, ...] = field(default=())

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> BaggedClassifier:
        """按袋拟合 `n_estimators` 棵树，返回新的（已拟合）实例。

        行数与跨度坐标不一致时**报错**：那意味着袋子会抽到别的观测上，而症状只是
        结果变差，不会报错（本仓踩过同类静默错位）。
        """
        self.options.validate()
        if self.sampler not in SAMPLERS:
            raise ValueError(f"未知袋子抽法：{self.sampler}（可选 {SAMPLERS}）")
        values = np.asarray(X, dtype=float)
        target = np.asarray(y)
        weight = (
            None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        if values.shape[0] != self.spans.groups.size:
            raise ValueError("X 的行数与跨度坐标不一致：袋子会抽到别的观测上")
        # 抽袋子是纯 Python 循环、被 GIL 串行化（实测线程并行只有 1.8 倍），
        # 所以并行粒度放在「一棵树 = 一个任务」上、用进程池吃满多核。
        trees = Parallel(n_jobs=self.options.n_jobs, prefer="processes")(
            delayed(_fit_bagged_tree)(
                values, target, weight, self.spans, self.sampler, index, self.options
            )
            for index in range(self.options.n_estimators)
        )
        return replace(self, trees=tuple(trees))

    def predict_proba(self, X: Any) -> np.ndarray:
        """各棵树的正类概率取平均（与随机森林同定义）。"""
        if not self.trees:
            raise ValueError("还没有拟合：先调用 fit")
        values = np.asarray(X, dtype=float)
        stacked = np.asarray(
            [np.asarray(tree.predict_proba(values))[:, 1] for tree in self.trees]
        )
        positive = stacked.mean(axis=0)
        return np.column_stack([1.0 - positive, positive])

    @property
    def feature_importances_(self) -> np.ndarray:
        """各棵树的 MDI 取平均（与随机森林同定义）。"""
        if not self.trees:
            raise ValueError("还没有拟合：先调用 fit")
        return np.asarray([tree.feature_importances_ for tree in self.trees]).mean(
            axis=0
        )


def bagged_factory(sampler: str) -> FitFunction:
    """造一个给 `meta.walk_forward_meta` 用的拟合函数（袋子按 `sampler` 抽）。

    超参从传入的 `MetaLabelOptions` 取，不在这里另立一份：否则「采样器的差异」会与
    「超参的差异」混在一起，而这正是本研究唯一要分开的东西。
    """

    def fit(
        train: pd.DataFrame, features: list[str], options: meta.MetaLabelOptions
    ) -> Classifier:
        bagging = BaggingOptions(
            n_estimators=options.n_estimators,
            max_depth=options.max_depth,
            min_samples_leaf=options.min_samples_leaf,
            seed=options.seed,
        )
        model = BaggedClassifier(bagging, SpanCoordinates.from_frame(train), sampler)
        return model.fit(
            train[features].to_numpy(dtype=float),
            train[meta.LABEL_COLUMN].to_numpy(),
            train[meta.WEIGHT_COLUMN].to_numpy(dtype=float),
        )

    return fit
