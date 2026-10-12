"""
第 9 章：超参调优（Hyper-Parameter Tuning）。

原文（Snippet 9.1~9.4）给了四件东西：拿 **purged K 折**做内层 CV 的网格搜索、一个能
透传 `sample_weight` 的增强 Pipeline、随机搜索、以及一个 **log-uniform** 分布。本模块
对应实现前三件；第四件直接用 `scipy.stats.loguniform`（原文的 `logUniform_gen` 是
scipy 1.4 之前手搓的，语义一样，见 `log_uniform`）。

## 三处必须说明的口径

1. **purge/embargo 的单位是「日期」而不是「观测」**。原文的单位是条数，因为它处理的是
   单条序列；本仓库的面板一行 = 一天的一只标的，同一天有上百行，按行算 purge 会把
   同一天的其它标的混进训练集——这正是 `meta.py` 里记过的泄漏（embargo 挡不住「同日」）。
   所以这里按**唯一日期**切折、按**日期数**给 purge 与 embargo，同一日期的所有标的
   永远同进同出。
2. **打分函数按标签集合选**：原文判 `set(lbl) == {0,1}` 就用 `f1`（元标签），否则用
   `neg_log_loss`（对各类别对称）。这里照办，把判据写成显式函数而不是隐式分支。
3. **搜索本身要计入试验账本**。在 K 个配置里挑最好的那个，与在 K 个策略里挑最好的那个
   没有区别。本模块返回 `candidates`（实际评估过的配置数），调用方必须记进
   `research.trials`，否则 DSR 的分母会少算。
4. **某折里没有正类时 f1 恒为 0，预测什么都一样**（实测：真值全 0 时，预测全 0 与预测
   全 1 都得 0）。这类「死折」会把折均值拉平，让调参看不见好坏——正类稀少的任务里，
   先看每折的正类占比，再决定要不要拿 f1 当搜索判据。

## 为什么不直接用 sklearn 的搜索器

`KFold` 与 `TimeSeriesSplit` 都不做 purge/embargo：相邻折之间标签重叠会把测试期信息漏
进训练集，而金融标签跨度往往就是几根到几十根。折循环自己写还有一个理由：要同时拿到
**逐折分数**（只给均分看不出稳定性）。抽样语义则与 `RandomizedSearchCV` 一致——每个
参数独立抽 `n_iter` 次，而不是取笛卡尔积。

估计器用**工厂 + Protocol**而不是传实例：`BaseEstimator` 不声明 `fit`/`predict_proba`
（那是 mixin 的事），`sklearn.base.clone` 的返回类型又是联合。与其在类型标注上绕圈子，
不如把「怎么造模型」显式交给调用方。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import check_random_state

DEFAULT_SPLITS: Final[int] = 5
META_LABEL_SCORING: Final[str] = "f1"
SYMMETRIC_SCORING: Final[str] = "neg_log_loss"
META_LABEL_VALUES: Final[frozenset[int]] = frozenset({0, 1})
MAX_GRID_CANDIDATES: Final[int] = 10_000


class Classifier(Protocol):
    """本模块对估计器的全部要求。"""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


class Distribution(Protocol):
    """能被抽样的分布（`scipy.stats` 的冻结分布）。"""

    def rvs(self, size: int = 1, random_state: Any = None) -> Any: ...


EstimatorFactory = Callable[[dict[str, Any]], Classifier]


def scoring_for(labels: Sequence[int] | pd.Series | np.ndarray) -> str:
    """按标签集合选打分函数名（供调用方交给 sklearn 的搜索器使用）。

    元标签（只有 0/1）用 `f1`——这个任务只看**被采纳的那一注**对不对，大量不交易的样本
    不该稀释分数；多分类或连续标签用 `neg_log_loss`，它对各类别对称、不偏向多数类。
    """
    values = frozenset(int(value) for value in np.unique(np.asarray(labels)))
    return META_LABEL_SCORING if values <= META_LABEL_VALUES else SYMMETRIC_SCORING


def positive_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    """正类 f1（即 `f1_score(average="binary", zero_division=0)`）。

    自己算而不用 sklearn：`zero_division` 在 sklearn 1.9 的标注里被写成 `str`，运行时却
    明确接受 0/1；默认值 `"warn"` 会在退化折上发警告，而本仓 `filterwarnings = ["error"]`
    会把它升级成异常。5 行公式比跟标注较劲干净。
    """
    truth = np.asarray(labels) == 1
    guess = np.asarray(predicted) == 1
    hits = float(np.sum(truth & guess))
    denominator = float(np.sum(truth) + np.sum(guess))
    return 0.0 if denominator == 0 else 2.0 * hits / denominator


def log_uniform(low: float, high: float) -> Distribution:
    """对数均匀分布，用于**乘法型**超参（学习率、正则强度、叶节点样本数）。

    原文 Snippet 9.4 的理由：这类参数在数量级上变化才有意义；均匀采样会把
    「0.001~0.01」与「100~1000」当成等宽的两段，搜索预算几乎全花在大数一端。取对数后
    均匀 = 每个数量级被抽到的概率相同。
    """
    if low <= 0 or high <= low:
        raise ValueError("对数均匀分布要求 0 < low < high")
    return cast(Distribution, stats.loguniform(low, high))


@dataclass(frozen=True)
class PurgedKFold:
    """按日期切折的 purge/embargo K 折，可直接当 sklearn 的 `cv` 用。

    `groups` 传已编码的日期序号（同一天同一个整数、按时间升序）；`purge` 与 `embargo`
    的单位是日期个数。
    """

    n_splits: int = DEFAULT_SPLITS
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("折数至少 2")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge 与 embargo 不能为负")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """产出 (训练行号, 测试行号)。`groups` 必传，否则退化成「同一天跨折」。"""
        if groups is None:
            raise ValueError("必须传 groups（日期序号），否则同一天会跨折泄漏")
        codes = np.asarray(groups)
        if codes.ndim != 1 or codes.size != len(np.asarray(X)):
            raise ValueError("groups 必须是与 X 行数相同的一维数组")
        unique = np.unique(codes)
        if unique.size < self.n_splits:
            raise ValueError("唯一日期数少于折数")
        bounds = np.linspace(0, unique.size, self.n_splits + 1, dtype=int)
        for fold in range(self.n_splits):
            test_dates = unique[bounds[fold] : bounds[fold + 1]]
            if test_dates.size == 0:
                continue
            yield (
                np.flatnonzero(self._train_mask(codes, test_dates, unique)),
                np.flatnonzero(np.isin(codes, test_dates)),
            )

    def _train_mask(
        self, codes: np.ndarray, test_dates: np.ndarray, unique: np.ndarray
    ) -> np.ndarray:
        """训练行：测试块之外，再挖掉紧邻的 purge 与 embargo 两段日期。"""
        low = int(np.searchsorted(unique, test_dates[0]))
        high = int(np.searchsorted(unique, test_dates[-1], side="right"))
        blocked = unique[
            max(low - self.purge, 0) : min(high + self.embargo, unique.size)
        ]
        return ~np.isin(codes, blocked)


@dataclass(frozen=True)
class TuningProblem:
    """一次调参所需的全部输入。

    `groups` 是日期序号（同一天同一个整数、按时间升序），`cv` 的内层折也由它切。
    """

    factory: EstimatorFactory
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    sample_weight: np.ndarray | None = None
    cv: PurgedKFold = field(default_factory=PurgedKFold)


@dataclass(frozen=True)
class SearchResult:
    """一次超参搜索的结果。`candidates` 是**实际评估过**的配置数（要记进账本）。"""

    best_params: dict[str, Any]
    best_score: float
    candidates: int
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def fold_scores(
    problem: TuningProblem, params: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """一个配置跑完所有折：返回（折均分, 各折分数）。

    只留分数、不留折外预测：调参用不到它，而「什么都返回」的接口迟早会被当缓存用。
    """
    scores: list[float] = []
    for train_index, test_index in problem.cv.split(
        problem.features, problem.labels, groups=problem.groups
    ):
        model = problem.factory(params)
        weights = (
            None
            if problem.sample_weight is None
            else problem.sample_weight[train_index]
        )
        model.fit(
            problem.features[train_index],
            problem.labels[train_index],
            sample_weight=weights,
        )
        probability = np.asarray(model.predict_proba(problem.features[test_index]))[
            :, 1
        ]
        scores.append(
            positive_f1(problem.labels[test_index], (probability >= 0.5).astype(int))
        )
    by_fold = {f"fold{index}": value for index, value in enumerate(scores)}
    return float(np.mean(scores)), by_fold


def search_hyperparameters(
    problem: TuningProblem,
    space: dict[str, Any],
    *,
    n_iter: int = 0,
    seed: int = 0,
    round_digits: int = 4,
) -> SearchResult:
    """随机（`n_iter > 0`）或网格搜索，内层 CV 是 purged K 折。

    `space` 的值可以是候选列表（枚举）或 `log_uniform` 之类的分布（抽样）；`n_iter > 0`
    时每个参数独立抽 `n_iter` 次，`n_iter <= 0` 时全枚举（所有键都必须是候选列表）。
    浮点参数按 `round_digits` 归整，保证 `best_params` 是可复现的字面量而不是
    `np.float64(0.30000000000000004)`——账本要按参数值做幂等判定。
    """
    evaluated: list[tuple[float, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for raw in candidate_list(space, n_iter=n_iter, seed=seed):
        params = clean_params(raw, round_digits)
        mean_score, by_fold = fold_scores(problem, params)
        evaluated.append((mean_score, params))
        rows.append({"score": mean_score, **by_fold, **params})
    best_score, best_params = max(evaluated, key=lambda item: item[0])
    frame = (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )
    return SearchResult(
        best_params=best_params,
        best_score=best_score,
        candidates=len(evaluated),
        fold_scores=frame,
    )


def clean_params(params: dict[str, Any], digits: int) -> dict[str, Any]:
    """把抽样出来的 numpy 标量变成可写进账本的普通 Python 标量。"""
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (np.floating, float)):
            number = float(value)
            cleaned[key] = int(round(number)) if digits <= 0 else round(number, digits)
        elif isinstance(value, (np.integer, int)):
            cleaned[key] = int(value)
        else:
            cleaned[key] = value
    return cleaned


def candidate_list(
    space: dict[str, Any], *, n_iter: int = 0, seed: int = 0
) -> list[dict[str, Any]]:
    """候选配置：给定 `n_iter` 就按参数独立抽样，否则枚举全网格。"""
    if not space:
        raise ValueError("搜索空间为空")
    rng = check_random_state(seed)
    if n_iter <= 0:
        return _grid(space)
    draws = {name: _draw(values, n_iter, rng) for name, values in space.items()}
    return [
        {name: values[position] for name, values in draws.items()}
        for position in range(n_iter)
    ]


def _grid(space: dict[str, Any]) -> list[dict[str, Any]]:
    """全枚举；候选值不是序列就说明调用方该给 `n_iter`。"""
    combos: list[dict[str, Any]] = [{}]
    for name, values in space.items():
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"n_iter<=0 时 {name} 必须是候选列表，分布式请给 n_iter")
        combos = [{**combo, name: value} for combo in combos for value in values]
        if len(combos) > MAX_GRID_CANDIDATES:
            raise ValueError("网格过大，请改用随机搜索（n_iter > 0）")
    return combos


def _draw(values: Any, n_iter: int, rng: np.random.RandomState) -> list[Any]:
    """一个参数抽 `n_iter` 个值：序列放回抽样，分布按分布抽。"""
    if isinstance(values, (list, tuple, np.ndarray)):
        return [rng.choice(np.asarray(values)) for _ in range(n_iter)]
    return list(values.rvs(size=n_iter, random_state=rng))
