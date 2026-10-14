"""
tests/research/test_tsmom_sign.py
「过去 N 根收益的符号」信号（经典 TSMOM 口径）的不变量。

与突破臂共用 TsmomOptions、波动率缩放与池子，只差 `kind`；这里锁住符号信号自己的性质：
暖机期为 0、单调行情恒 ±1、多空由 allow_short 控制、多速度取平均、无前视、缩放照乘，
以及默认 kind 仍是突破（不回归）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.tsmom import (
    SIGNAL_BREAKOUT,
    SIGNAL_SIGN,
    TsmomOptions,
    multi_speed_signal,
    sign_signal,
    tsmom_signal,
)

BARS = 400


def make_bars(closes: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """构造日线：high/low 由 close 派生，volume 固定为 1。"""
    index = pd.date_range(start, periods=closes.size, freq="D", tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1.0,
        }
    )


def trending(periods: int = BARS, drift: float = 0.01) -> pd.DataFrame:
    """单调上行序列。"""
    return make_bars(100.0 * np.cumprod(1.0 + np.full(periods, drift)))


def test_sign_signal_is_one_when_trending_up() -> None:
    """三个速度都在涨 → 信号恒 +1（只多与多空相同）；暖机期给 0。"""
    bars = trending()
    long_only = tsmom_signal(bars, TsmomOptions(kind=SIGNAL_SIGN, vol_target=None))
    long_short = tsmom_signal(
        bars, TsmomOptions(kind=SIGNAL_SIGN, vol_target=None, allow_short=True)
    )
    assert long_only.iloc[:20].eq(0.0).all()  # 三档都还没暖机完
    assert long_only.iloc[100:].eq(1.0).all()
    assert long_short.iloc[100:].eq(1.0).all()


def test_sign_signal_is_negative_only_when_shorts_allowed() -> None:
    """单调下行 → 多空 −1、只多 0：同一信号，截不截负号就差在这里。"""
    bars = make_bars(100.0 * np.cumprod(np.full(BARS, 0.99)))
    long_only = tsmom_signal(bars, TsmomOptions(kind=SIGNAL_SIGN, vol_target=None))
    long_short = tsmom_signal(
        bars, TsmomOptions(kind=SIGNAL_SIGN, vol_target=None, allow_short=True)
    )
    assert long_only.iloc[100:].eq(0.0).all()
    assert long_short.iloc[100:].eq(-1.0).all()


def test_sign_signal_averages_speeds() -> None:
    """短速度说涨、长速度说跌 → 两档平均归零（多空口径）。"""
    down = 100.0 * np.cumprod(np.full(40, 0.98))
    up = down[-1] * np.cumprod(np.full(20, 1.01))
    bars = make_bars(np.concatenate([down, up]))
    fast = tsmom_signal(
        bars,
        TsmomOptions(
            kind=SIGNAL_SIGN, entry_periods=(20,), vol_target=None, allow_short=True
        ),
    )
    slow = tsmom_signal(
        bars,
        TsmomOptions(
            kind=SIGNAL_SIGN, entry_periods=(55,), vol_target=None, allow_short=True
        ),
    )
    both = tsmom_signal(
        bars,
        TsmomOptions(
            kind=SIGNAL_SIGN, entry_periods=(20, 55), vol_target=None, allow_short=True
        ),
    )
    assert fast.iloc[-1] == pytest.approx(1.0)
    assert slow.iloc[-1] == pytest.approx(-1.0)
    assert both.iloc[-1] == pytest.approx(0.0)


def test_sign_signal_does_not_look_ahead() -> None:
    """改掉最后 100 根之后的行情，不该改变更早的信号（最长速度 100）。"""
    options = TsmomOptions(kind=SIGNAL_SIGN, vol_target=None, allow_short=True)
    base = trending()
    before = tsmom_signal(base, options)
    changed = base.copy()
    changed.loc[changed.index[300] :, "close"] *= 0.5
    after = tsmom_signal(changed, options)
    assert before.iloc[:300].equals(after.iloc[:300])


def test_sign_signal_respects_vol_target() -> None:
    """波动率缩放照乘：关掉缩放时等于原始符号，高波动时被缩小。"""
    index = np.arange(BARS)
    bars = make_bars(100.0 * (1.005**index) * np.where(index % 2 == 0, 1.0, 1.06))
    raw = sign_signal(
        bars, TsmomOptions(kind=SIGNAL_SIGN, entry_periods=(20,), vol_target=None)
    )
    plain = tsmom_signal(
        bars, TsmomOptions(kind=SIGNAL_SIGN, entry_periods=(20,), vol_target=None)
    )
    scaled = tsmom_signal(
        bars, TsmomOptions(kind=SIGNAL_SIGN, entry_periods=(20,), vol_target=0.05)
    )
    assert plain.equals(raw)
    assert abs(scaled.iloc[-1]) < abs(raw.iloc[-1])


def test_breakout_remains_the_default_kind() -> None:
    """默认 kind 仍是突破：加了符号分支不影响已有口径。"""
    bars = trending()
    assert TsmomOptions().kind == SIGNAL_BREAKOUT
    default = tsmom_signal(bars, TsmomOptions(vol_target=None))
    breakout = multi_speed_signal(bars, TsmomOptions(vol_target=None))
    assert default.equals(breakout)
