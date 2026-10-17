"""
tests/research/test_bars.py
美元棒：按成交金额累积采样，索引必须取每根的收盘时刻。

钉的是三条会静默算错的地方：分组边界（累积额取整）、OHLC 归属（首开尾收极值）、
时间戳（收盘时刻而非开盘时刻——取错就等于用了未来信息）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.sampling import bars

CLOSES = [10.0] * 10
OPENS = [10.0, 10.1, 9.9, 10.2, 9.8, 10.3, 9.7, 10.4, 9.6, 10.5]
HIGHS = [11.0, 12.0, 10.5, 11.5, 13.0, 10.2, 11.1, 10.9, 12.5, 10.3]
LOWS = [9.0, 8.0, 9.5, 8.5, 7.0, 9.8, 8.9, 9.1, 7.5, 9.7]
THRESHOLD = 25.0


def _tile(values: list[float], periods: int) -> list[float]:
    """把 10 根的图案循环铺到任意长度（长样本用）。"""
    return [values[i % len(values)] for i in range(periods)]


def _frame(periods: int = 10) -> pd.DataFrame:
    """每根 1m 名义额恒为 10（close 10 × volume 1），累积额便于手算。"""
    index = pd.date_range("2022-01-01", periods=periods, freq="1min", tz="UTC")
    index.name = "datetime"
    return pd.DataFrame(
        {
            "open": _tile(OPENS, periods),
            "high": _tile(HIGHS, periods),
            "low": _tile(LOWS, periods),
            "close": _tile(CLOSES, periods),
            "volume": 1.0,
        },
        index=index,
    )


def _options(threshold: float = THRESHOLD, drop_partial: bool = True):
    return bars.BarOptions(threshold=threshold, drop_partial=drop_partial)


def test_cumulative_notional_defines_the_groups() -> None:
    """累积额除以门槛取整即分组号：25 的门槛下每 2~3 根收一根。"""
    result = bars.dollar_bars(_frame(), _options())
    assert len(result) == 4
    assert result["notional"].tolist() == [20.0, 20.0, 30.0, 20.0]


def test_ohlc_comes_from_the_right_bars() -> None:
    """开盘取组内第一根、收盘取最后一根、高低取组内极值。"""
    result = bars.dollar_bars(_frame(), _options())
    assert result["open"].tolist() == [10.0, 9.9, 9.8, 10.4]
    assert result["close"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert result["high"].tolist() == [12.0, 11.5, 13.0, 12.5]
    assert result["low"].tolist() == [8.0, 8.5, 7.0, 7.5]
    assert result["volume"].tolist() == [2.0, 2.0, 3.0, 2.0]


def test_timestamp_is_the_close_time_of_the_last_bar() -> None:
    """时间戳必须是最后一根的收盘时刻——取开盘时刻就等于用了未来信息。"""
    frame = _frame()
    result = bars.dollar_bars(frame, _options())
    assert result.index[0] == frame.index[1]
    assert result.index[2] == frame.index[6]
    assert result.index[3] == frame.index[8]
    assert result.index.name == "datetime"


def test_drop_partial_removes_the_incomplete_tail() -> None:
    """末尾那根累积额不足门槛（这里只有 10 < 25），默认丢掉。"""
    kept = bars.dollar_bars(_frame(), _options(drop_partial=False))
    dropped = bars.dollar_bars(_frame(), _options(drop_partial=True))
    assert len(kept) == 5
    assert len(dropped) == 4
    assert kept.index[4] == _frame().index[9]


def test_notional_is_conserved_except_for_the_dropped_tail() -> None:
    """聚合不能凭空多算或少算成交额。"""
    frame = _frame()
    total = float((frame["close"] * frame["volume"]).sum())
    kept = bars.dollar_bars(frame, _options(drop_partial=False))
    assert float(kept["notional"].sum()) == pytest.approx(total)
    dropped = bars.dollar_bars(frame, _options())
    assert float(dropped["notional"].sum()) == pytest.approx(total - 10.0)


def test_threshold_scales_with_target_bar_count() -> None:
    """门槛 = 总名义额 / 目标根数；目标根数越多，每根越细。"""
    frame = _frame()
    assert bars.threshold_for_bars(frame, 4) == pytest.approx(25.0)
    assert bars.threshold_for_bars(frame, 20) == pytest.approx(5.0)


def test_daily_threshold_targets_one_bar_per_calendar_day() -> None:
    """按「每天一根」折算：两天数据应得到约 2 根棒。"""
    frame = _frame(periods=2880)
    threshold = bars.threshold_for_daily_bars(frame)
    result = bars.dollar_bars(frame, bars.BarOptions(threshold=threshold))
    assert 2 <= len(result) <= 3


def test_empty_or_non_positive_threshold_yields_no_bars() -> None:
    """门槛非正时不能返回一根「全量」的假棒。"""
    assert bars.dollar_bars(_frame(), _options(threshold=0.0)).empty
    assert bars.dollar_bars(_frame(), _options(threshold=0.0)).index.name == "datetime"


def test_volume_bars_group_on_volume() -> None:
    """量棒按成交量分组：量恒为 1 时门槛 2.5 得到每根 2~3 的量。"""
    result = bars.volume_bars(_frame(), _options(threshold=2.5))
    assert result["volume"].tolist() == [2.0, 2.0, 3.0, 2.0]


def test_tick_bars_need_a_trades_column() -> None:
    """tick 棒按笔数分组；源数据没有 trades 列时必须显式报错，不能拿别的列顶替。"""
    with pytest.raises(KeyError):
        bars.tick_bars(_frame(), _options(threshold=2.5))
    frame = _frame()
    frame["trades"] = 2.0
    result = bars.tick_bars(frame, _options(threshold=4.0))
    assert result["trades"].tolist() == [2.0, 4.0, 4.0, 4.0, 4.0]


def test_accumulation_kinds_agree_when_proportional() -> None:
    """成交量恒为 1 时成交额与成交量成正比，两种分组的切口必须逐根相同。"""
    dollars = bars.dollar_bars(_frame(), _options(threshold=25.0))
    volumes = bars.volume_bars(_frame(), _options(threshold=2.5))
    assert dollars.index.equals(volumes.index)
    assert dollars["close"].tolist() == volumes["close"].tolist()


def test_signed_volume_is_taker_buy_minus_sell() -> None:
    """订单流净额 = 2×主动买 − 总量（等价于主动买 − 主动卖）。"""
    frame = _frame()
    frame["quote_volume"] = 100.0
    frame["taker_buy_quote"] = 60.0
    assert bars.signed_volume(frame).tolist() == [20.0] * 10
