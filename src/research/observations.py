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

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.research import labels, overlap
from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.panel import Panel, listing_age_days, tradable_mask
from src.research.factors.portfolio import PortfolioOptions, portfolio_stats

BARRIER_FIELDS = ("label", "t1", "ret")
MIN_BARS_PER_VERTICAL = 4
AGE_COLUMN = "age_days"
TRADABLE_COLUMN = "tradable"
BARRIER_LABEL_COLUMN = "barrier_label"
"""三重障碍标签的列名（`symbol_observations` 由 `BARRIER_FIELDS` 拼出）。"""


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


def to_long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """宽表转长表（date, symbol, name）——面板与特征统一到这一个形态才能 join。

    `future_stack=True` 是新实现：旧实现 `dropna=False` 已弃用，而两者在当前 pandas 上
    逐值等价（暖机期的 NaN 都要留着，绝不能用会丢行的旧默认值）。
    """
    stacked = frame.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["date", "symbol"])
    return pd.DataFrame({name: stacked}).reset_index()


def build_meta_frame(
    panel: Panel,
    *,
    primary_key: str,
    feature_keys: Sequence[str],
    vertical: int,
) -> pd.DataFrame:
    """把面板拼成元标签研究用的长表：方向、标签、权重、特征一次到位。

    与 `build_observations` 的分工：那里只出标签与权重，这里再加上主模型方向、因子特征、
    上市天数（`age_days`——上市时长本身就是市场状态）与可交易掩码，并滤掉三类无意义行：
    不可交易、`side` 为 0（主模型没观点）、以及三个必需列有缺失。

    两条口径与 `build_observations` 一致：特征按 `(date, symbol)` **内连接**（缺一天就
    整行不要，不做填充），方向为 0 的行直接丢（元标签只回答「这一注要不要下」）。
    """
    side_frame = FACTOR_LIBRARY[primary_key].values(panel)
    table = build_observations(panel, side_frame, vertical)
    pieces = [to_long(FACTOR_LIBRARY[key].values(panel), key) for key in feature_keys]
    pieces.append(to_long(listing_age_days(panel), AGE_COLUMN))
    features = pieces[0]
    for piece in pieces[1:]:
        features = features.merge(piece, on=["date", "symbol"], how="inner")
    tradable = to_long(tradable_mask(panel), TRADABLE_COLUMN)
    frame = table.merge(features, on=["date", "symbol"], how="inner")
    frame = frame.merge(tradable, on=["date", "symbol"], how="inner")
    frame = frame.loc[frame[TRADABLE_COLUMN].fillna(False)]
    frame = frame.dropna(subset=["side", "forward", BARRIER_LABEL_COLUMN])
    return frame.loc[frame["side"] != 0].reset_index(drop=True)


def strategy_spreads(
    keys: Sequence[str],
    panel: Panel,
    mask: pd.DataFrame,
    *,
    horizon: int,
    min_series: int,
) -> pd.DataFrame:
    """逐策略跑分位组合，取逐期价差序列拼成宽表（列名 = 因子键）。

    `min_series` 以下的策略直接丢掉——拼宽表时它只会把公共区间截短。HRP 与 CPCV 两个
    研究脚本共用这一步，权重估计与切分口径由各自接。
    """
    series: dict[str, pd.Series] = {}
    for key in keys:
        stats = portfolio_stats(
            FACTOR_LIBRARY[key], panel, mask, PortfolioOptions(horizon=horizon)
        )
        spread = stats.spread_returns.dropna()
        if spread.size >= min_series:
            series[key] = spread
    return pd.DataFrame(series)
