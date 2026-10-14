"""
信息驱动采样：按累积变量收棒的「等量棒」家族（de Prado Ch 2）。

时间棒假设信息按恒定熵率到达，实际不是——行情冷清时一根 1m 棒里几乎没发生事，
行情剧烈时同一根里发生了一整天的事，两者却被当成同等权重的观测。
本模块按**累积变量**采样：每根棒装到同样多的成交额（美元棒）、成交量（量棒）或
成交笔数（tick 棒）才收一根，于是每根棒的信息含量大致相等，采样频率自动随活跃度变化。

三者共用同一套分组与聚合逻辑（`accumulation_bars`），只换累积变量——这样才能回答
「换累积变量会不会改变结论」，而不是把差异混进两套实现里。按**买卖失衡**采样
（imbalance / run 棒）是另一套顺序算法，见 `src/research/imbalance_bars.py`。

对本仓库的意义：现有全部结论都建立在时间棒上，「3 根反转」是**日历尺度**。换成
美元棒后同一个窗口号变成**信息尺度**，两者的 IC 差异直接回答「信号到底活在哪个
尺度上」——这是时间棒无法回答的问题。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 2。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")
SECONDS_PER_DAY = 86_400
# 累积变量：成交额（美元棒，de Prado 主推口径）、成交量（量棒）、成交笔数（tick 棒）
ACCUMULATION_NOTIONAL = "notional"
ACCUMULATION_VOLUME = "volume"
ACCUMULATION_TRADES = "trades"
# 每根棒至少要装到门槛的这个比例。不要求恰好 1.0：分组按累积额取整，跨过门槛的
# 那根 1m 棒的溢出会被计入下一组，所以偶有略低于门槛的棒（实测 BTC p1=0.976、
# p0=0.902，全在 0.9 以上）。低于一半就是分组算错了。
MIN_BAR_FILL_RATIO = 0.5


@dataclass(frozen=True)
class BarOptions:
    """等量棒的采样口径：累积变量装到多少就收一根。"""

    # 每根棒累积的量（成交额/成交量/笔数，由调用方选定）门槛；累积量跨过它即收一根
    threshold: float
    # 丢掉末尾不足门槛的那根：样本尾部必然不完整，留着会得到一根信息量不足的棒
    drop_partial: bool = True


def accumulation_weights(frame: pd.DataFrame, variable: str) -> pd.Series:
    """累积变量序列：成交额（收盘价×成交量）、成交量或成交笔数。

    成交额用计价货币计，门槛就是「每根棒装多少美元」；成交量用基础货币计；笔数就是笔数。
    需要 `volume` 或 `trades` 列——OKX 的 K 线只有 OHLCV，所以 tick 棒只能用带笔数的源。
    """
    if variable == ACCUMULATION_NOTIONAL:
        return cast(pd.Series, frame["close"] * frame["volume"])
    return cast(pd.Series, frame[variable])


def threshold_for_bars(
    frame: pd.DataFrame, target_bars: int, variable: str = ACCUMULATION_NOTIONAL
) -> float:
    """按目标根数反推门槛：总累积量 / 目标根数。

    门槛是**每标的各自算**的——不同币的成交额差几个数量级，用统一门槛会让
    大币每根棒只含几分钟、小币一根棒跨几个月。
    """
    total = float(accumulation_weights(frame, variable).sum())
    return total / max(target_bars, 1)


def threshold_for_daily_bars(
    frame: pd.DataFrame, variable: str = ACCUMULATION_NOTIONAL
) -> float:
    """按「每天一根」折算门槛，便于与日线时间棒直接对比根数。

    天数向上取整：不足一天的那截也是日历上的一天。截断会让跳 1.99 天的样本
    只按 1 天折算，门槛翻倍、根数减半。
    """
    span = (frame.index[-1] - frame.index[0]).total_seconds()
    days = max(math.ceil(span / SECONDS_PER_DAY), 1)
    return threshold_for_bars(frame, days, variable)


def group_bounds(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分组号 → 每组的首末下标，供聚合与顺序算法（失衡/run 棒）共用。"""
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:] - 1, group.size - 1]
    return starts, ends


def aggregate_groups(
    frame: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    extra_name: str = "",
    extra: np.ndarray | None = None,
) -> pd.DataFrame:
    """按分组下标把源棒聚成新棒：首开尾收、极值取组内、量求和，索引取**收盘时刻**。

    索引必须是最后一根的收盘时刻：那一刻这根棒的信息才齐全，用开盘时刻等于用了未来信息。
    """
    values: dict[str, np.ndarray] = {
        "open": frame["open"].to_numpy(dtype=float)[starts],
        "high": np.maximum.reduceat(frame["high"].to_numpy(dtype=float), starts),
        "low": np.minimum.reduceat(frame["low"].to_numpy(dtype=float), starts),
        "close": frame["close"].to_numpy(dtype=float)[ends],
        "volume": np.add.reduceat(frame["volume"].to_numpy(dtype=float), starts),
    }
    if extra is not None and extra_name:
        values[extra_name] = np.add.reduceat(extra, starts)
    return pd.DataFrame(
        values, index=pd.DatetimeIndex(frame.index[ends], name="datetime")
    )


def accumulation_bars(
    frame: pd.DataFrame, options: BarOptions, variable: str = ACCUMULATION_NOTIONAL
) -> pd.DataFrame:
    """把时间棒聚合成「每根装同样多累积变量」的棒，索引取每根的**收盘时刻**。

    累积量除以门槛取整就是分组号，因此整段聚合可以向量化——30 只 × 300 万根
    用 Python 循环跑不动。

    分组号取的是**累积量在每根末端的取整值**，所以跨过门槛的那根源棒会归入它
    完成的那一组（而不是下一组），溢出量极小时每根都恰好装到门槛；单根源棒
    本身很大时溢出会抬高下一组起点，使某根略低于门槛，下限见 `MIN_BAR_FILL_RATIO`。
    """
    weights = accumulation_weights(frame, variable).to_numpy(dtype=float)
    if weights.size == 0 or options.threshold <= 0:
        return _empty(variable)
    group = np.floor(np.cumsum(weights) / options.threshold).astype(np.int64)
    starts, ends = group_bounds(group)
    bars = aggregate_groups(frame, starts, ends, extra_name=variable, extra=weights)
    if options.drop_partial and float(bars[variable].iloc[-1]) < options.threshold:
        bars = bars.iloc[:-1]
    return bars


def dollar_bars(frame: pd.DataFrame, options: BarOptions) -> pd.DataFrame:
    """美元棒：每根装同样多的成交额（de Prado 的主推口径）。"""
    return accumulation_bars(frame, options, ACCUMULATION_NOTIONAL)


def volume_bars(frame: pd.DataFrame, options: BarOptions) -> pd.DataFrame:
    """量棒：每根装同样多的成交量（基础货币计，不折算价格）。"""
    return accumulation_bars(frame, options, ACCUMULATION_VOLUME)


def tick_bars(frame: pd.DataFrame, options: BarOptions) -> pd.DataFrame:
    """tick 棒：每根装同样多的成交笔数（要求源数据带 `trades` 列）。"""
    return accumulation_bars(frame, options, ACCUMULATION_TRADES)


def signed_volume(frame: pd.DataFrame) -> pd.Series:
    """订单流净额（计价货币）：2×taker_buy_quote − quote_volume，正 = 主动买占优。

    taker 主动买是被对手方挂单吃掉的成交，这是「谁在吃单」唯一可得的代理；
    OKX 的 K 线不给这个拆分，只有 Binance 的 K 线带 `taker_buy_base/quote`。
    """
    return cast(pd.Series, 2.0 * frame["taker_buy_quote"] - frame["quote_volume"])


def _empty(variable: str = ACCUMULATION_NOTIONAL) -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="datetime")
    extra = [] if variable in BAR_COLUMNS else [variable]
    return pd.DataFrame(columns=pd.Index([*BAR_COLUMNS, *extra]), index=index)
