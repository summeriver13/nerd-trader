"""
src/analytics/metrics.py
绩效指标的唯一实现：研究引擎、稳健性检验、模拟盘都从这里取，禁止各算一套。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 年化口径：crypto 7×24，按 365 天折算
ANNUAL_DAYS = 365
# 至少两个观测点才算得出一条收益序列的均值/标准差
MIN_OBSERVATIONS = 2


def total_return(net_ret: pd.Series) -> float:
    """一段区间的复利收益。"""
    return float((1 + net_ret).prod() - 1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """资金曲线逐点回撤（绘图与统计共用）；曲线归零处记 -1。"""
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    values = np.where(peak > 0, (equity - peak) / peak, -1.0)
    return pd.Series(values, index=equity.index)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """资金曲线的最大回撤（负数）。"""
    if equity.empty:
        return 0.0
    return float(drawdown_series(equity).min())


def max_drawdown(net_ret: pd.Series) -> float:
    """由收益序列还原资金曲线后的最大回撤（负数）。"""
    return max_drawdown_from_equity((1 + net_ret).cumprod())


def annualize(
    total_ret: float, periods: int, periods_per_year: int = ANNUAL_DAYS
) -> float:
    """把区间收益按 periods_per_year 复利折算成年化；亏光返回 -1。"""
    if total_ret <= -1 or periods <= 0:
        return -1.0
    return (1 + total_ret) ** (periods_per_year / periods) - 1


def bars_per_year(index: pd.Index, periods_per_year: int = ANNUAL_DAYS) -> float:
    """每年多少根 K 线：按区间实际跨度算，7×24 与有缺口的品种都能对齐。"""
    span_days = (index[-1] - index[0]).days if len(index) > 1 else 0
    if span_days <= 0:
        span_days = 1
    return len(index) / (span_days / periods_per_year)


def sharpe_ratio(returns: pd.Series | np.ndarray) -> float:
    """逐根夏普（未年化）：DSR / PBO 用这个口径。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_OBSERVATIONS:
        return 0.0
    sd = float(arr.std(ddof=1))
    return float(arr.mean() / sd) if sd > 0 else 0.0


def annualized_sharpe(returns: pd.Series | np.ndarray, periods: float) -> float:
    """年化夏普：展示口径（= 逐根夏普 × √每年根数）。"""
    return sharpe_ratio(returns) * float(np.sqrt(periods))


def sortino_ratio(returns: pd.Series | np.ndarray, periods: float) -> float:
    """索提诺：下行偏差用 MAR=0 的标准定义（全样本取 min(r, 0) 的均方根）。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_OBSERVATIONS:
        return 0.0
    downside = float(np.sqrt((np.minimum(arr, 0.0) ** 2).mean()))
    if downside == 0:
        return 0.0
    return float(arr.mean() / downside * np.sqrt(periods))


def calmar_ratio(annual_ret: float, max_dd: float) -> float:
    """年化收益 / |最大回撤|；无回撤时记 0。"""
    if not max_dd:
        return 0.0
    return annual_ret / abs(max_dd)
