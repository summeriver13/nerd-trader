"""
元标签（de Prado Ch 3.6）：把「下哪个方向」与「要不要下注」拆开。

仓库此前所有策略都是**一个**模型同时决定方向与仓位——这正是作者在
《The 10 Reasons Most Machine Learning Funds Fail》里列的 Pitfall #6
「同时学方向和仓位」。后果是模型无法诚实回答「我这个方向判断准不准」，
因为不准的方向也能靠仓位救回来；反过来，准的方向也会因为仓位没优化被埋没。

元标签把这件事拆成两步：

1. **主模型只定方向**——这里就是现有因子（`rev_3` 空高分位之类）。它的方向是
   二值的，不需要学。
2. **元模型只回答「这一注要不要下」**——标签是主模型那一注**赚没赚钱**（1/0），
   特征是当时的市场状态（波动、流动性、上市天数、SADF 爆炸性……）。

两条必须守住的口径（改代码前先读）：

1. **切分按日期，不按行。** 同一天有 200 个标的，按行切会让同一天的观测同时出现在
   训练与测试里——`embargo` 完全挡不住这种泄漏，因为泄漏不是「相邻」，是「同日」。
2. **滚动预测，不用样本内。** 分类器在样本内的精确率毫无意义（树模型能记住训练集），
   元标签的价值必须用**样本外**预测衡量。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 3.6。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATE_COLUMN = "date"
LABEL_COLUMN = "label"
WEIGHT_COLUMN = "weight"
PROBABILITY_COLUMN = "probability"


@dataclass(frozen=True)
class MetaLabelOptions:
    """滚动训练窗参数。`min_train` 是**观测数**（一行 = 一天的一只标的），
    `test_span` 与 `embargo` 是**天数**——窗口按唯一日期切，禁运也必须按天：本仓
    面板每天有上百只标的，按观测数算「200 个观测」只有约 1 天，3 日标签照样重叠。
    """

    min_train: int = 2000  # 首次训练所需的最少观测
    test_span: int = 250  # 每个测试窗的天数
    embargo: int = 8  # 训练与测试之间挖掉的天数（de Prado 的 0.01T 在本研究样本上约 8 天）
    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 50
    seed: int = 0
    collect_fits: bool = False  # 留下每折的模型，供特征重要性复用（免得重训一遍）

    def validate(self) -> None:
        if self.min_train < 50:
            raise ValueError("首次训练窗至少 50 个观测")
        if self.test_span < 1:
            raise ValueError("测试窗至少 1 个观测")
        if self.embargo < 0:
            raise ValueError("禁运不能为负")


@dataclass(frozen=True)
class FoldFit:
    """一折的拟合结果：模型 + 它在**测试折**上的行标签 + 它用过的**训练行**。

    特征重要性（`src/research/importance.py`）要用同一批折、同一批模型，
    否则「哪些特征有用」和「元标签有没有效」讲的是两个模型。

    `train_rows` 是给 SFI 用的：它必须在**同一批折**上重训（每折只用一个特征），
    没有训练行就没法重训。
    """

    model: RandomForestClassifier
    rows: pd.Index
    train_rows: pd.Index


@dataclass(frozen=True)
class MetaPrediction:
    """滚动预测的样本外结果。"""

    probability: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    folds: int = 0
    train_sizes: tuple[int, ...] = ()
    fits: tuple[FoldFit, ...] = ()

    @property
    def observations(self) -> int:
        return int(self.probability.notna().sum())


def meta_label(returns: pd.Series, side: pd.Series) -> pd.Series:
    """元标签：主模型那一注赚没赚钱（1/0）。

    `side` 已定向（做空为 −1），`returns` 是标的的原始收益。
    两者索引必须对齐、都无空值。
    """
    aligned = returns.reindex(side.index)
    if aligned.isna().any():
        raise ValueError("收益与方向未对齐，元标签会静默错位")
    signs = np.sign(aligned.to_numpy(dtype=float)) * side.to_numpy(dtype=float)
    return pd.Series((signs > 0).astype(int), index=side.index)


def build_classifier(options: MetaLabelOptions) -> RandomForestClassifier:
    """按 `options` 造一个**未拟合**的分类器。

    超参只应有一处定义：调参（`research.tuning`）要的是「按参数造模型」而不是
    「造好顺便训了」，所以把构造与拟合拆开，否则两边各写一份 n_estimators/max_depth，
    迟早会漂移成「模型的差异」冒充「特征的差异」。
    """
    return RandomForestClassifier(
        n_estimators=options.n_estimators,
        max_depth=options.max_depth,
        min_samples_leaf=options.min_samples_leaf,
        random_state=options.seed,
        n_jobs=-1,
    )


RF_PARAM_SPACE: Final[dict[str, list[Any]]] = {
    "n_estimators": [50, 200],
    "max_depth": [4, 6, 8, 12],
    "min_samples_leaf": [10, 50, 200],
}
"""调参的搜索空间（键必须是 `build_classifier` 能透给 `replace` 的字段）。"""

RF_DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 50,
}
"""默认配置，即流水线实际上线用的那一套（与 `MetaLabelOptions` 的默认值一致，
由用例钉住）。它必须在 `RF_PARAM_SPACE` 里——否则「调参 vs 默认」比的不是同一件事。"""


def fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    options: MetaLabelOptions,
) -> RandomForestClassifier:
    """按 `options` 训一个分类器。

    公开是因为 SFI（`importance.single_feature_importance`）要在**同一批折、同一组
    超参**上把特征集换成单个特征重训——超参只应有一处定义，否则「特征差异」会与
    「模型差异」混在一起。
    """
    model = build_classifier(options)
    model.fit(
        train[features].to_numpy(),
        train[LABEL_COLUMN].to_numpy(),
        sample_weight=train[WEIGHT_COLUMN].to_numpy(),
    )
    return model


def _positive_probability(
    model: RandomForestClassifier, test: pd.DataFrame, features: list[str]
) -> np.ndarray:
    proba = np.asarray(model.predict_proba(test[features].to_numpy()))
    return np.asarray(proba[:, 1], dtype=float)


def walk_forward_meta(
    frame: pd.DataFrame,
    features: list[str],
    options: MetaLabelOptions,
) -> MetaPrediction:
    """滚动训练 + 样本外预测。

    `frame` 必须含 `date`、`label`（元标签）、`weight`（唯一性权重）与特征列。
    每个测试窗用**它之前**的全部观测训练，末尾再挖掉 `embargo` **天**。

    切分按日期：同一日期的所有标的必须落在同一个窗里。

    返回的概率**按 `frame` 的行标签对齐**，不按位置——调用方很可能是按标的拼的
    frame（本仓库的研究脚本就是），按位置对齐会静默把概率配到别的观测上。
    """
    options.validate()
    required = (DATE_COLUMN, LABEL_COLUMN, WEIGHT_COLUMN, *features)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}")
    if frame.index.has_duplicates:
        raise ValueError("frame 的索引必须唯一：概率要按行标签回填")

    # kind="stable"：让输出与输入的行序无关（按日期排序时同一天的标的保持原次序）
    ordered = frame.sort_values(DATE_COLUMN, kind="stable")
    labels = ordered.index
    dates = ordered[DATE_COLUMN].to_numpy()
    unique_dates = np.unique(dates)
    # 每个唯一日期在 ordered 里的首行位置：按日期切分靠它，不靠 isin
    boundaries = np.searchsorted(dates, unique_dates, side="left")
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    train_sizes: list[int] = []
    fits: list[FoldFit] = []
    folds = 0
    cursor = 0
    while cursor < unique_dates.size:
        stop = min(cursor + options.test_span, unique_dates.size)
        test_start = int(boundaries[cursor])
        test_stop = int(boundaries[stop]) if stop < unique_dates.size else len(ordered)
        # 训练集是测试窗之前的全部观测，末尾再挖掉 embargo **天**。按日期切到整天边界：
        # 按观测数切会把不足一天的零头当禁运，3 日标签照样重叠（README 记过的残留漏洞）。
        train_stop = int(boundaries[max(cursor - options.embargo, 0)])
        if train_stop >= options.min_train:
            model = fit_classifier(ordered.iloc[:train_stop], features, options)
            fitted = _positive_probability(
                model, ordered.iloc[test_start:test_stop], features
            )
            rows = labels[test_start:test_stop]
            probability.loc[rows] = fitted
            train_sizes.append(train_stop)
            if options.collect_fits:
                fits.append(
                    FoldFit(
                        model=model,
                        rows=rows,
                        train_rows=labels[:train_stop],
                    )
                )
            folds += 1
        cursor = stop
    return MetaPrediction(
        probability=probability,
        folds=folds,
        train_sizes=tuple(train_sizes),
        fits=tuple(fits),
    )


def filter_summary(
    returns: pd.Series,
    probability: pd.Series,
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7),
) -> pd.DataFrame:
    """按元模型概率过滤后的每注收益与命中率。

    `thresholds` 里 0.0 表示「不过滤」（全部下注），作为对照基线。
    返回的 `lift` 是相对基线每注收益的增量。
    """
    aligned = returns.reindex(probability.index)
    usable = probability.notna() & aligned.notna()
    rows: list[dict[str, object]] = []
    baseline = float(aligned[usable].mean()) if usable.any() else 0.0
    for threshold in thresholds:
        taken = usable & (probability >= threshold)
        chosen = aligned[taken]
        per_bet = float(chosen.mean()) if chosen.size else 0.0
        rows.append(
            {
                "threshold": threshold,
                "bets": int(chosen.size),
                "share": float(chosen.size / usable.sum()) if usable.any() else 0.0,
                "per_bet_bp": per_bet * 1e4,
                "hit_rate": float((chosen > 0).mean()) if chosen.size else 0.0,
                "sharpe": float(chosen.mean() / chosen.std()) * np.sqrt(len(chosen))
                if chosen.size > 1 and chosen.std() > 0
                else 0.0,
                "lift_bp": (per_bet - baseline) * 1e4,
            }
        )
    return pd.DataFrame(rows)
