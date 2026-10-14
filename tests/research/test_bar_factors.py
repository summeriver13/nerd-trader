"""
tests/research/test_bar_factors.py
棒尺度因子的两个静默陷阱：逐标的各算各的、按日抽样不看未来。

先拼成 DataFrame 再 `shift(w)` 会把不同标的的棒时戳错位对齐（实测 IC 处处恰好 0）；
抽样目标日不加一天（日线索引是 K 线**起始**时刻）会用到当日收盘之后才收的棒。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.bar_factors import bar_reversal, calendar_reversal, rescaled_reversal


def _bars(prices: list[float], start: str = "2024-01-01") -> pd.Series:
    """每天一根、时刻取当日 23:00（收盘时刻）的棒序列。"""
    index = pd.date_range(
        start, periods=len(prices), freq="D", tz="UTC"
    ) + np.timedelta64(23, "h")
    index.name = "datetime"
    return pd.Series(prices, index=index, dtype=float)


def _daily(periods: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """日线索引是 K 线的起始时刻（每天 00:00）。"""
    return pd.DatetimeIndex(
        pd.date_range(start, periods=periods, freq="D", tz="UTC"), name="datetime"
    )


def test_rescaled_reversal_uses_each_symbols_own_bars() -> None:
    """两个标的走势相反时，结果必须按各自序列算——先拼后 shift 会把符号搅乱。"""
    closes = {
        "AAA": _bars([10.0, 11.0, 12.0, 13.0]),
        "BBB": _bars([20.0, 19.0, 18.0, 17.0]),
    }
    result = rescaled_reversal(closes, _daily(4), 1)
    assert (result["AAA"].iloc[1:] < 0).all()
    assert (result["BBB"].iloc[1:] > 0).all()


def test_rescaled_reversal_does_not_use_bars_after_the_close() -> None:
    """第 2 天只能用 1/2 23:00 那根，不能用 1/3 的——目标时刻加一天才对。"""
    closes = {"AAA": _bars([10.0, 11.0, 12.0])}
    result = rescaled_reversal(closes, _daily(2), 1)
    assert result["AAA"].iloc[1] == pytest.approx(-(11.0 / 10.0 - 1.0))


def test_reversal_points_against_the_move() -> None:
    """方向约定：涨过（10→12）给负值、跌过（12→9）给正值。"""
    close = _bars([10.0, 12.0, 9.0])
    assert bar_reversal(close, 1).iloc[1] < 0
    assert bar_reversal(close, 1).iloc[2] > 0
    frame = pd.DataFrame({"AAA": close})
    assert float(calendar_reversal(frame, 1).iloc[2, 0]) > 0


def test_rescaled_reversal_rejects_misaligned_timestamps() -> None:
    """棒全部晚于面板索引时要显式报错，不能悄悄返回全 NaN 的因子。

    反过来（棒早于面板）不报错也不该报错：抽样用 ffill，早的棒本来就是有效历史。
    """
    closes = {"AAA": _bars([10.0, 11.0], start="2025-01-01")}
    with pytest.raises(ValueError, match="全为 NaN"):
        rescaled_reversal(closes, _daily(3, start="2024-01-01"), 1)
