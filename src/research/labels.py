"""
三重障碍标签（de Prado Ch 3）。

固定期限标签（「h 天后的收益符号」）有两个问题：一是它假设你**只**在 h 天后平仓，
而真实交易会在止盈/止损/超时三条路径上先到先出；二是它的标签长度恒为 h，
使唯一性权重退化成 1/h（与抽稀数学等价），Ch 4 因此无法独立实现。

三重障碍把标签长度变成**可变**的：每条路径在自己的触及时刻结束。这带来两个后果——
标签更接近可实现的交易，以及唯一性权重终于有内容（见 overlap.py）。

三条口径（改代码前先读）：

1. **障碍宽度按波动率标度，不是固定百分比。** 同一个 1% 对 BTC 和 PEPE 是完全不同的
   事件；固定百分比会让高波动标的几乎总是先触上障碍。
2. **首次触及决定标签，标签是触及时刻的收益符号**（不是「哪个障碍被触及」）。
   这是作者 getBins 的口径：bin = sign(ret)，所以碰到垂直障碍时可能是 0。
   同一根上两条障碍都被穿越时取上障碍——与作者 min(axis=1) 的任意取舍一致
   （同一时刻两个时间戳相等，取谁都对）。
3. **t1 返回整数位置而非时间戳。** 唯一性权重与 purge 都按根序号算，
   返回时间戳还要再折回来一次。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 3。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_VOL_SPAN = 100  # 作者 getDailyVol 的默认 span
MIN_VOL_OBSERVATIONS = 20  # 波动率可用前的最少根数

BARRIER_COLUMNS = ("label", "t1", "ret", "upper", "lower", "vol")


@dataclass(frozen=True)
class TripleBarrierOptions:
    """三重障碍参数。side 在调用处给，见 triple_barrier。"""

    profit_take: float = 1.0  # 上障碍 = +profit_take × 波动率
    stop_loss: float = 1.0  # 下障碍 = −stop_loss × 波动率
    vertical: int = 20  # 垂直障碍：最多持有多少根
    vol_span: float = DEFAULT_VOL_SPAN

    def validate(self) -> None:
        if self.profit_take <= 0 or self.stop_loss <= 0:
            raise ValueError("障碍宽度必须为正")
        if self.vertical < 1:
            raise ValueError("垂直障碍至少 1 根")
        if self.vol_span <= 1:
            raise ValueError("波动率 span 至少 2")


def ewma_volatility(close: pd.Series, span: float = DEFAULT_VOL_SPAN) -> pd.Series:
    """EWMA 波动率（作者 getDailyVol：收益率的标准差，非年化）。"""
    return close.pct_change().ewm(span=span).std()


def triple_barrier(
    close: pd.Series,
    options: TripleBarrierOptions,
    side: int = 1,
) -> pd.DataFrame:
    """逐根出发的三重障碍标签，返回索引与 close 相同的表。

    side=1 看多、side=-1 看空：路径收益统一乘 side 后再比障碍，
    所以做空时「上障碍」就是价格下跌那一侧（作者的 sides 口径）。

    列：label（触及时刻收益的符号）、t1（触及位置，整数根序号）、
    ret（触及时刻相对起点的收益，已按 side 定向）、upper/lower（障碍价格）、
    vol（起点处的波动率）。

    末尾 vertical 根不出事件——它们凑不满垂直窗口。
    """
    options.validate()
    if side not in (1, -1):
        raise ValueError("side 只能是 1 或 -1")
    size = len(close)
    empty = pd.DataFrame(index=close.index, columns=pd.Index(BARRIER_COLUMNS))
    if size <= options.vertical + MIN_VOL_OBSERVATIONS:
        return empty

    values = close.to_numpy(dtype=float)
    vol = ewma_volatility(close, options.vol_span).to_numpy(dtype=float)
    starts = np.arange(size - options.vertical)
    widths = vol[starts]
    usable = np.isfinite(widths) & (widths > 0)
    starts, widths = starts[usable], widths[usable]
    if starts.size == 0:
        return empty

    # 路径矩阵：第 i 行是第 i 个事件的 vertical 根前向收益（已按 side 定向）
    steps = np.arange(1, options.vertical + 1)
    window = starts[:, None] + steps[None, :]
    path = (values[window] / values[starts][:, None] - 1.0) * side
    # 首次触及：argmax 在全 False 时返回 0，必须用哨兵换掉
    hit_up = path > (options.profit_take * widths)[:, None]
    hit_down = path < (-options.stop_loss * widths)[:, None]
    last = options.vertical - 1
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), last)
    first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), last)
    column = np.minimum(first_up, first_down)
    rows = np.arange(starts.size)

    upper = values[starts] * (1.0 + options.profit_take * widths)
    lower = values[starts] * (1.0 - options.stop_loss * widths)
    if side == -1:  # 做空时价格方向反过来
        upper, lower = lower, upper

    return pd.DataFrame(
        {
            "label": np.sign(path[rows, column]).astype(int),
            "t1": starts + column + 1,
            "ret": path[rows, column],
            "upper": upper,
            "lower": lower,
            "vol": widths,
        },
        index=close.index.take(starts),
    )
