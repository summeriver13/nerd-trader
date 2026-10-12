"""
信息驱动采样：美元棒（de Prado Ch 2）。

时间棒假设信息按恒定熵率到达，实际不是——行情冷清时一根 1m 棒里几乎没发生事，
行情剧烈时同一根里发生了一整天的事，两者却被当成同等权重的观测。
美元棒按**成交金额**累积采样：每根棒包含同样多的美元成交额，于是每根棒的信息含量
大致相等，采样频率自动随活跃度变化。

对本仓库的意义：现有全部结论都建立在时间棒上，「3 根反转」是**日历尺度**。换成
美元棒后同一个窗口号变成**信息尺度**，两者的 IC 差异直接回答「信号到底活在哪个
尺度上」——这是时间棒无法回答的问题。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 2。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")
SECONDS_PER_DAY = 86_400
# 每根棒至少要装到门槛的这个比例。不要求恰好 1.0：分组按累积额取整，跨过门槛的
# 那根 1m 棒的溢出会被计入下一组，所以偶有略低于门槛的棒（实测 BTC p1=0.976、
# p0=0.902，全在 0.9 以上）。低于一半就是分组算错了。
MIN_BAR_FILL_RATIO = 0.5


@dataclass(frozen=True)
class DollarBarOptions:
    """美元棒的采样口径。"""

    # 每根棒累积的美元成交额门槛；累积额跨过它即收一根
    threshold: float
    # 丢掉末尾不足门槛的那根：样本尾部必然不完整，留着会得到一根信息量不足的棒
    drop_partial: bool = True


def threshold_for_bars(frame: pd.DataFrame, target_bars: int) -> float:
    """按目标根数反推门槛：总名义额 / 目标根数。

    门槛是**每标的各自算**的——不同币的成交额差几个数量级，用统一门槛会让
    大币每根棒只含几分钟、小币一根棒跨几个月。
    """
    notional = float((frame["close"] * frame["volume"]).sum())
    return notional / max(target_bars, 1)


def threshold_for_daily_bars(frame: pd.DataFrame) -> float:
    """按「每天一根」折算门槛，便于与日线时间棒直接对比根数。

    天数向上取整：不足一天的那截也是日历上的一天。截断会让跳 1.99 天的样本
    只按 1 天折算，门槛翻倍、根数减半。
    """
    span = (frame.index[-1] - frame.index[0]).total_seconds()
    days = max(math.ceil(span / SECONDS_PER_DAY), 1)
    return threshold_for_bars(frame, days)


def dollar_bars(frame: pd.DataFrame, options: DollarBarOptions) -> pd.DataFrame:
    """把时间棒聚合成美元棒，索引取每根的**收盘时刻**（信息此时才可得）。

    累积成交额除以门槛取整就是分组号，因此整段聚合可以向量化——30 只 × 300 万根
    用 Python 循环跑不动。open 取组内第一根的开盘，close 取最后一根的收盘，
    high/low 取组内极值，volume/notional 求和。

    分组号取的是**累积额在每根末端的取整值**，所以跨过门槛的那根 1m 棒会归入它
    完成的那一组（而不是下一组），溢出量极小时每根都恰好装到门槛；单根 1m 棒
    本身很大时溢出会抬高下一组起点，使某根略低于门槛，下限见 `MIN_BAR_FILL_RATIO`。
    """
    notional = (frame["close"] * frame["volume"]).to_numpy(dtype=float)
    if notional.size == 0 or options.threshold <= 0:
        return _empty()
    group = np.floor(np.cumsum(notional) / options.threshold).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:] - 1, group.size - 1]
    bars = pd.DataFrame(
        {
            "open": frame["open"].to_numpy(dtype=float)[starts],
            "high": np.maximum.reduceat(frame["high"].to_numpy(dtype=float), starts),
            "low": np.minimum.reduceat(frame["low"].to_numpy(dtype=float), starts),
            "close": frame["close"].to_numpy(dtype=float)[ends],
            "volume": np.add.reduceat(frame["volume"].to_numpy(dtype=float), starts),
            "notional": np.add.reduceat(notional, starts),
        },
        index=pd.DatetimeIndex(frame.index[ends], name="datetime"),
    )
    if options.drop_partial and bars["notional"].iloc[-1] < options.threshold:
        bars = bars.iloc[:-1]
    return bars


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="datetime")
    return pd.DataFrame(columns=pd.Index([*BAR_COLUMNS, "notional"]), index=index)
