"""
src/research/factors/operators.py
Alpha101（Kakushadze 2016, arXiv:1601.00991）里的三个通用算子：不含因子语义，只吃历史。

为什么只搬算子、不搬那 101 条公式：见 docs/factors/README.md 第二十八节。实测结论是
「信号真、成本不允许」——纯 OHLCV 的条目在永续日线 h=1 的平衡成本只有 0.0~1.1bp，
而本库单边成本 10bp；缺的不是公式，是能把换手压下来的变换。

约定与 library.py 一致：
- 满窗才出值（min_periods=window），故派生因子的预热下标 = 基因子预热 + window − 1；
- 常数窗（方差为 0）返回 NaN 而不是 0——那条序列上根本没有相关信息。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

# 两个平滑算子的扫描窗口（根）。5/10/20 覆盖「几乎不平滑 → 慢到改变信号周期」
DECAY_WINDOWS: tuple[int, ...] = (5, 10, 20)
TS_RANK_WINDOWS: tuple[int, ...] = (10, 20)
# 量价交互因子的相关窗口
CORRELATION_WINDOWS: tuple[int, ...] = (10, 20)

MIN_OPERATOR_WINDOW: int = 2


def decay_linear(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权平均：窗口内权重 window, window−1, …, 1（越近权重越大）。

    等价 Alpha101 的 decay_linear。它是**换手平滑器**：同一信号在新旧读数之间
    按线性权重过渡，逐期仓位变化变小，成本随之下降——但毛边际也会被一起磨平，
    净效果只能实测（见 docs/factors/README.md 第二十八节）。
    """
    if window < MIN_OPERATOR_WINDOW:
        raise ValueError(f"decay_linear 的窗口至少 {MIN_OPERATOR_WINDOW} 根：{window}")
    weights = np.arange(window, 0, -1, dtype=float)
    lagged = [frame.shift(lag) * float(weight) for lag, weight in enumerate(weights)]
    total = lagged[0]
    for part in lagged[1:]:
        total = total + part
    counted = cast(pd.DataFrame, frame.notna().rolling(window, min_periods=window).sum())
    return cast(pd.DataFrame, total.div(float(weights.sum())).where(counted.eq(window)))


def ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """当根在过去 window 根里的百分位排名（1 = 自身近期最高）。

    等价 Alpha101 的 ts_rank：把「绝对值」换成「相对自身近期历史的位置」，天然
    抗尺度漂移（币价涨 10 倍不影响读数），代价是历史被反复重算、换手未必更低。
    """
    if window < MIN_OPERATOR_WINDOW:
        raise ValueError(f"ts_rank 的窗口至少 {MIN_OPERATOR_WINDOW} 根：{window}")
    rolling = frame.rolling(window, min_periods=window)
    return cast(pd.DataFrame, rolling.rank(pct=True))


def correlation(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    """两条面板序列的滚动 Pearson 相关（逐标的，满窗才出值）。

    等价 Alpha101 的 correlation。任一序列在窗口内为常数时方差为 0，结果记 NaN：
    成交额恒定的标的（测试床里常见）本来就没有「量价关系」可谈，填 0 会变成
    「完全不相关」这种假读数。
    """
    if window < MIN_OPERATOR_WINDOW:
        raise ValueError(f"correlation 的窗口至少 {MIN_OPERATOR_WINDOW} 根：{window}")
    rolling = left.rolling(window, min_periods=window)
    return cast(pd.DataFrame, rolling.corr(right))
