"""
src/research/pairs/funding.py
两腿的资金费口径：符号约定、日费率对齐、覆盖率统计。

为什么单独一层：资金费是配对里唯一「两腿方向相反」的现金项——多头腿付正费率、
空头腿收正费率，所以净额远小于成交成本（实测只占成本的 4%）。它容易被写成
「和成本一样扣掉」，那样会把结论带偏，所以口径集中在这里，并且单独报覆盖率。

约定：``funding`` 是按日求和的费率宽表（正 = 多头付钱，见
``data_sources.funding.daily_rates``）。缺失日按 0 计，覆盖情况由
``funding_coverage`` 与 ``funding_symbols`` 两个数分别报告——标的面覆盖与
日覆盖是两件事，只报一个会被读成另一个。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.factors.panel import Panel
from src.research.pairs.selection import Pair


def funding_cost(
    pair: Pair, executed: pd.Series, funding: pd.DataFrame | None, weight: float
) -> np.ndarray:
    """两腿的资金费（正 = 支出）：多头付正费率、空头收正费率。

    每对占 1 单位资金时多头腿名义 ``weight = 1/(1+β)``、空头腿 ``β·weight``，
    所以 ``pos > 0``（多价差）付左腿、收右腿，``pos < 0`` 反之。
    """
    if funding is None or pair.left not in funding.columns:
        return np.zeros(executed.size, dtype=float)
    pos = executed.to_numpy(dtype=float)
    rate_left = _daily_rate(funding, pair.left, executed.index)
    rate_right = (
        _daily_rate(funding, pair.right, executed.index)
        if pair.right in funding.columns
        else np.zeros(executed.size, dtype=float)
    )
    return weight * pos * rate_left - weight * pair.beta * pos * rate_right


def funding_symbols(panel: Panel, funding: pd.DataFrame | None) -> int:
    """有费率表的池内标的数。"""
    if funding is None or funding.empty:
        return 0
    return sum(1 for column in funding.columns if column in set(panel.symbols))


def funding_coverage(panel: Panel, funding: pd.DataFrame | None) -> float:
    """资金费日覆盖：**已有费率表的那些标的**里，有价格的格子有多大比例有费率。"""
    if funding is None or funding.empty:
        return 0.0
    columns = [column for column in funding.columns if column in set(panel.symbols)]
    if not columns:
        return 0.0
    prices = panel.close[columns].notna().to_numpy()
    if not bool(prices.any()):
        return 0.0
    rates = funding[columns].reindex(panel.index).notna().to_numpy()
    return float(rates[prices].mean())


def _daily_rate(funding: pd.DataFrame, symbol: str, index: pd.Index) -> np.ndarray:
    """取某标的的日费率并对齐到交易期（缺失日 0）。"""
    days = pd.DatetimeIndex(pd.DatetimeIndex(index).to_series().dt.normalize())
    series = funding[symbol].reindex(days)
    return np.nan_to_num(series.to_numpy(dtype=float), nan=0.0)
