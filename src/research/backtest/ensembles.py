"""Ch 6 集成研究的基础设施：跨度接线与估计器族。

两件事：

1. `attach_spans`——把元标签长表接到「每只标的自己的根坐标」上。顺序 bootstrap 的并发与
   唯一性只能在这套坐标里算；用长表**行号**当根号不会报错，只会静默算错（观测行只是交易日
   的子集，rank 会把缺口压掉、并发算小，顺序采样就退化成均匀采样）。所以这里的键是
   `close.dropna()` 里的**位置**，与 `observations.build_meta_frame` 算 `barrier_t1` 用的是
   同一条序列。
2. `arm_fitters` / `forest_factory`——Ch 6 对照实验的估计器族：单棵、随机森林、ExtraTrees、
   手写均匀袋、手写顺序袋。超参一律从 `MetaLabelOptions` 派生，各臂之间**只差结构**。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.research.labeling import bagging, meta

DATE_COLUMN = "date"
SYMBOL_COLUMN = "symbol"
BARRIER_END_COLUMN = "barrier_t1"


def attach_spans(frame: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """补上跨度坐标四列：标的序号、起始根、结束根、该标的总根数。

    `close` 是面板的收盘价矩阵（索引为日期、列为标的）。返回**同一批行**（不删行）：
    标的或观测日对不上直接报错——那意味着袋子会抽到别的观测上；`barrier_t1` 缺失的
    留 NaN，丢不丢交给调用方。
    """
    unknown = set(frame[SYMBOL_COLUMN].astype(str)) - {str(s) for s in close.columns}
    if unknown:
        raise ValueError(f"这些标的在收盘价矩阵里没有：{sorted(unknown)}")
    positions: dict[str, dict[Any, int]] = {
        str(symbol): {
            pd.Timestamp(date): number
            for number, date in enumerate(close[symbol].dropna().index)
        }
        for symbol in close.columns
    }
    groups = {str(symbol): number for number, symbol in enumerate(close.columns)}
    sizes = {str(symbol): len(positions[str(symbol)]) for symbol in close.columns}
    starts: list[int] = []
    for date, symbol in zip(frame[DATE_COLUMN], frame[SYMBOL_COLUMN], strict=True):
        known = positions[str(symbol)]
        stamp = pd.Timestamp(date)
        if stamp not in known:
            raise ValueError(f"{symbol} 在 {stamp} 没有根：观测日与收盘价矩阵对不上")
        starts.append(known[stamp])
    symbols = [str(value) for value in frame[SYMBOL_COLUMN]]
    result = frame.copy()
    result[bagging.GROUP_COLUMN] = np.asarray(
        [groups[symbol] for symbol in symbols], dtype=np.int64
    )
    result[bagging.START_COLUMN] = np.asarray(starts, dtype=np.int64)
    result[bagging.END_COLUMN] = frame[BARRIER_END_COLUMN].to_numpy()
    result[bagging.SIZE_COLUMN] = np.asarray(
        [sizes[symbol] for symbol in symbols], dtype=np.int64
    )
    return result


def bag_uniqueness(frame: pd.DataFrame, *, rows: int, seed: int) -> pd.Series:
    """首窗训练行上的袋平均唯一性（逐标的再取均值）：均匀袋 vs 顺序袋各抽一次。

    只抽一次袋子（不按树数重复，也不按折重复）——抽袋子是本研究最贵的一步。
    """
    train = frame.sort_values(meta.DATE_COLUMN, kind="stable").iloc[:rows]
    spans = bagging.SpanCoordinates.from_frame(train)
    values = {
        sampler: float(spans.uniqueness_by_group(sampler, seed=seed).mean())
        for sampler in bagging.SAMPLERS
    }
    return pd.Series(values, name="bag_uniqueness")


def forest_factory(*, extra: bool, trees: int) -> meta.FitFunction:
    """sklearn 森林的拟合函数（`extra=True` 走 ExtraTrees：无 bootstrap、随机切分）。"""

    def fit(
        train: pd.DataFrame, features: list[str], options: meta.MetaLabelOptions
    ) -> meta.Classifier:
        chosen = ExtraTreesClassifier if extra else RandomForestClassifier
        model = chosen(
            n_estimators=trees,
            max_depth=options.max_depth,
            min_samples_leaf=options.min_samples_leaf,
            random_state=options.seed,
            n_jobs=-1,
        )
        model.fit(
            train[features].to_numpy(dtype=float),
            train[meta.LABEL_COLUMN].to_numpy(),
            sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(dtype=float),
        )
        return model

    return fit


def arm_fitters(trees: int) -> dict[str, meta.FitFunction]:
    """五个臂的拟合函数：`single`、`rf`、`et`、`uni`、`seq`（见研究脚本的对照表）。"""

    def single(
        train: pd.DataFrame, features: list[str], options: meta.MetaLabelOptions
    ) -> meta.Classifier:
        return meta.fit_classifier(train, features, replace(options, n_estimators=1))

    return {
        "single": single,
        "rf": forest_factory(extra=False, trees=trees),
        "et": forest_factory(extra=True, trees=trees),
        "uni": bagging.bagged_factory(bagging.SAMPLER_UNIFORM),
        "seq": bagging.bagged_factory(bagging.SAMPLER_SEQUENTIAL),
    }
