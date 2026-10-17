"""
信息驱动采样：订单流失衡棒与 run 棒（de Prado Ch 2.3）。

等量棒（美元/量/tick 棒）按**成交量**采样，回答「交易发生了多少」；失衡棒按**买卖失衡**
采样，回答「谁在吃单」——同样一笔成交，主动买与主动卖的信息含量不同。做法是逐笔累积
有符号的不平衡量 θ，当 |θ| 超过「期望每根棒的不平衡量」时收一根棒，而期望值随每根棒
用 EWMA 更新：

- 失衡棒：`|Σ b_t| ≥ E[T]·E[|b_t|]·|2P[b_t=+1]−1|`。b_t 是有符号量：tick 口径取
  `±1`（此时 `E[|b_t|]=1`，退化为书里的 `E[T]·|2P−1|`），量棒与美元棒取有符号成交量/
  成交额——多出的 `E[|b_t|]` 是把阈值换算回同一量纲，是本仓对原式量纲不齐处的补充。
- run 棒：`max(本根的买 run 长度, 卖 run 长度) ≥ E[run 长度]`，看的是**同向连续**本身，
  与净额无关：交替行情里 run 恒为 1、几乎不收棒，趋势行情里 run 一拉长就收。

  这一族在公开实现间有出入（原书代码、mlfinlab、MQL5 各不同）。本仓在真实数据（BTC/ETH/SOL
  的 1 小时源）上比过三种：① θ 累加「当前最长 run」——θ 按 T²/2 涨而阈值只按 T 涨，棒长
  收敛到 1 根源棒（BTC 上 58,220 根 ≈ 源棒数）；② 「买卖分路累积、取大者」——实测与失衡棒
  同根数，区分不开；③ 「最长 run 对期望 run」——既不退化又与失衡棒不同（6.4~7.0 小时一根、
  跨度 p5-p95 = 3~16 小时）。故取 ③。

失衡棒用三个尺度：`E[T]`（每根棒的源棒数期望）、`P[b=+1]`（主动买概率）、`E[|b_t|]`
（有符号量的量级），前两个在每根棒收盘时更新、第三个逐笔更新，都用同一个 EWMA(α)；
run 棒只用「期望 run 长度」，收盘时用本根实现的 run 长度更新。`expected_bars` 是这些尺度的
**初值**（失衡棒＝目标每根装几根源棒，run 棒＝初始期望 run 长度），不是择优参数。

**粒度降级（已知代价）**：de Prado 的算法跑在逐笔数据上，本仓的源是 **1 小时 K 线**
（Binance 现货带 `trades` 与 taker 分列，见 `scripts/fetch_binance_klines.py`），所以
「一笔」= 1 小时棒，小时内的成交顺序丢失，run 长度以小时计。读数只能读作「同粒度下
采样方式的差异」，不能当逐笔结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.research.sampling.bars import BAR_COLUMNS, aggregate_groups, group_bounds

# 有符号量口径：只取方向（±1）、有符号成交量、有符号成交额
SIGN_TICK = "tick"
SIGN_VOLUME = "volume"
SIGN_NOTIONAL = "notional"


@dataclass(frozen=True)
class FlowBarOptions:
    """失衡/run 棒的采样口径。"""

    # EWMA 平滑系数：每根棒收盘时更新 E[T] 与 P[b=+1]（E[|b|] 逐笔更新）
    alpha: float = 0.05
    # 尺度初值：失衡棒是 E[T]（每根棒期望装几根源棒），run 棒是期望的最长 run 长度。
    # 两者随后都由 EWMA 自适应，这里只是起点，不是择优旋钮
    expected_bars: float = 24.0
    # 丢掉末尾那根未达标的棒：尾部不完整，信息量不足
    drop_partial: bool = True


def signed_metric(frame: pd.DataFrame, kind: str) -> pd.Series:
    """有符号量 b_t：主动买减主动卖。

    成交额口径 = `2×taker_buy_quote − quote_volume`；成交量口径同式换成 base 列；
    tick 口径只取成交额净额的符号，零失衡按「延续」记为 +1（与常见实现一致）。
    """
    net = frame["taker_buy_quote"] * 2.0 - frame["quote_volume"]
    if kind == SIGN_NOTIONAL:
        return cast(pd.Series, net)
    if kind == SIGN_VOLUME:
        return cast(pd.Series, frame["taker_buy_base"] * 2.0 - frame["volume"])
    sign = np.sign(net.to_numpy(dtype=float))
    sign[sign == 0.0] = 1.0
    return pd.Series(sign, index=frame.index)


def flow_bars(
    frame: pd.DataFrame,
    values: pd.Series,
    options: FlowBarOptions,
    *,
    runs: bool = False,
) -> pd.DataFrame:
    """顺序累积采样：失衡棒（`runs=False`）或 run 棒（`runs=True`）。

    顺序算法无法向量化——阈值依赖前一根棒的收盘状态，只能逐根源棒推。
    """
    weights = values.to_numpy(dtype=float)
    total = weights.size
    if total == 0 or options.expected_bars <= 0:
        return _empty()
    alpha = options.alpha
    expected = options.expected_bars
    prob_buy = 0.5
    abs_metric = abs(float(weights[0])) or 1.0
    theta = 0.0
    up_run = down_run = 0
    group = np.zeros(total, dtype=np.int64)
    group_id = 0
    start = 0
    for index in range(total):
        value = weights[index]
        if runs:
            if value > 0.0:
                up_run += 1
                down_run = 0
            elif value < 0.0:
                down_run += 1
                up_run = 0
            theta = float(max(up_run, down_run))
            threshold = expected
        else:
            theta += value
            abs_metric = (1.0 - alpha) * abs_metric + alpha * abs(value)
            threshold = expected * abs_metric * abs(2.0 * prob_buy - 1.0)
        group[index] = group_id
        if abs(theta) >= threshold:
            realized = theta if runs else float(index - start + 1)
            expected = (1.0 - alpha) * expected + alpha * realized
            if value > 0.0:
                prob_buy = (1.0 - alpha) * prob_buy + alpha
            elif value < 0.0:
                prob_buy = (1.0 - alpha) * prob_buy
            theta = 0.0
            up_run = down_run = 0
            start = index + 1
            group_id += 1
    starts, ends = group_bounds(group)
    if options.drop_partial and start < total:
        starts, ends = starts[:-1], ends[:-1]
    if starts.size == 0:
        return _empty()
    return aggregate_groups(frame, starts, ends)


def imbalance_bars(
    frame: pd.DataFrame, options: FlowBarOptions, kind: str = SIGN_NOTIONAL
) -> pd.DataFrame:
    """失衡棒：累积 |有符号量| 超出期望即收一根（美元失衡默认口径）。"""
    return flow_bars(frame, signed_metric(frame, kind), options)


def run_bars(
    frame: pd.DataFrame, options: FlowBarOptions, kind: str = SIGN_NOTIONAL
) -> pd.DataFrame:
    """run 棒：本根内最长同向 run 超过期望长度即收一根。

    `options.expected_bars` 在这里是**初始期望 run 长度**，随后由实现的 run 长度用同一
    EWMA 自我更新，所以它跟着市场自己的连续性强弱走，不需要外部标定。
    """
    return flow_bars(frame, signed_metric(frame, kind), options, runs=True)


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="datetime")
    return pd.DataFrame(columns=pd.Index(list(BAR_COLUMNS)), index=index)
