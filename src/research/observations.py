"""
从面板建元标签观测表（de Prado Ch 3 + Ch 4）。

一只标的、一天一行，含：主模型方向 `side`、固定 h 日前向收益 `forward`、两套标签
（`fixed_label` 与三重障碍的 `barrier_label`）、两套样本权重（`weight_uniform` 与
三重障碍唯一性 `weight_unique`）。三变体对照研究全部建在这张表上。

一条最容易写错、且错了不报错的口径（踩过）：三重障碍的 `t1` 是**标的内的根序号**，
而长表的行号是另一套坐标，两者混用会让唯一性权重**静默错位**（表象是打印出「中位持有
−83897 根」）。所以唯一性必须在标的内部算——`bar_uniqueness` 以该标的的根数为轴长，
`symbol_observations` 先按标的算完再拼成长表。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.research import labels, overlap
from src.research.factors.panel import Panel

BARRIER_FIELDS = ("label", "t1", "ret")
MIN_BARS_PER_VERTICAL = 4


def bar_uniqueness(touch: np.ndarray, bars: int) -> np.ndarray:
    """标的内的唯一性权重。`touch` 是该标的的根序号坐标，轴长就是 bars。"""
    weight = np.ones(touch.size, dtype=float)
    usable = np.isfinite(touch)
    if not usable.any():
        return weight
    starts = np.flatnonzero(usable)
    ends = np.minimum(touch[usable], bars).astype("int64")
    weight[starts] = overlap.uniqueness_from_ends(bars, starts, ends)
    return weight


def symbol_observations(
    panel: Panel, side_frame: pd.DataFrame, symbol: str, horizon: int
) -> pd.DataFrame:
    """单只标的：方向、前向收益、两套标签与唯一性权重。

    样本太短的标的直接返回空表（少于 `MIN_BARS_PER_VERTICAL` 个垂直障碍宽度）。
    """
    close = cast(pd.Series, panel.close[symbol]).dropna()
    options = labels.TripleBarrierOptions(vertical=horizon)
    if close.size < MIN_BARS_PER_VERTICAL * options.vertical:
        return pd.DataFrame()
    side = np.sign(side_frame[symbol].reindex(close.index).to_numpy(dtype=float))
    # 障碍方向依赖 side，两侧各算一遍再按观测挑
    bars = {
        s: labels.triple_barrier(close, options, side=s).reindex(close.index)
        for s in (1, -1)
    }
    is_long = side > 0
    columns: dict[str, object] = {}
    for field in BARRIER_FIELDS:
        columns[f"barrier_{field}"] = np.where(
            is_long, bars[1][field].to_numpy(), bars[-1][field].to_numpy()
        )
    touch = np.asarray(columns["barrier_t1"], dtype=float)
    forward = (close.shift(-horizon).div(close) - 1.0).to_numpy()
    return pd.DataFrame(
        {
            "date": close.index,
            "symbol": symbol,
            "side": side,
            "forward": forward,
            "fixed_label": (side * forward > 0).astype(float),
            "weight_uniform": 1.0,
            "weight_unique": bar_uniqueness(touch, close.size),
            **columns,
        }
    )


def build_observations(
    panel: Panel, side_frame: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """把全部标的的观测拼成长表（空表自动跳过）。"""
    rows = [
        symbol_observations(panel, side_frame, symbol, horizon)
        for symbol in panel.close.columns
    ]
    return pd.concat([row for row in rows if not row.empty], ignore_index=True)
