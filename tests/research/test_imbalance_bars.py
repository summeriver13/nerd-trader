"""
tests/research/test_imbalance_bars.py
失衡棒与 run 棒：顺序累积、由 flow 驱动、无前视。

钉的是四件事：① 收棒由**失衡**驱动而不是时间（单边行情里持续收棒、零失衡时不收）；
② run 棒按同向连续长度累积，与失衡棒不是同一条曲线；③ 阈值只看过去——改未来不能
移动已经收掉的棒；④ 末尾未达标的棒要把丢掉，否则会多出一根信息量不足的棒。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research import imbalance_bars as ib

BARS = 80


def make_flow_frame(net: np.ndarray) -> pd.DataFrame:
    """构造带 taker 分列的 1 小时源棒：净额给定，主动买卖对其余部分对半拆。

    净额 = 2×主动买 − 总量，所以反过来给定净额即可精确控制有符号量。
    """
    index = pd.date_range("2024-01-01", periods=net.size, freq="h", tz="UTC")
    index.name = "datetime"
    quote = np.abs(net) + 100.0
    taker_buy = (quote + net) / 2.0
    close = 100.0 + np.arange(net.size) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
            "quote_volume": quote,
            "trades": 10.0,
            "taker_buy_base": taker_buy / close,
            "taker_buy_quote": taker_buy,
        },
        index=index,
    )


def one_sided_then_quiet() -> pd.DataFrame:
    """前 40 根单边买（每根 +1），后 40 根零失衡。"""
    return make_flow_frame(np.r_[np.full(40, 1.0), np.zeros(40)])


def test_signed_metric_is_buy_minus_sell() -> None:
    """有符号量 = 2×主动买 − 总量：净额给 3 就是 3，给 −5 就是 −5。"""
    frame = make_flow_frame(np.array([3.0, -5.0, 0.0]))
    assert ib.signed_metric(frame, ib.SIGN_NOTIONAL).tolist() == [3.0, -5.0, 0.0]
    assert ib.signed_metric(frame, ib.SIGN_TICK).tolist() == [1.0, -1.0, 1.0]


def test_bars_are_driven_by_flow_not_by_time() -> None:
    """单边段收棒密集、零失衡段几乎不收——这正是时间棒做不到的事。"""
    frame = one_sided_then_quiet()
    bars = ib.imbalance_bars(
        frame, ib.FlowBarOptions(expected_bars=8.0), ib.SIGN_NOTIONAL
    )
    cutoff = frame.index[39] + np.timedelta64(1, "h")
    early = int((bars.index < cutoff).sum())
    late = int((bars.index >= cutoff).sum())
    assert early >= 4  # 单边段 40 根源棒里连收 4 根以上
    # 零失衡段 40 根里最多再收 1 根：|b| 的 EWMA 衰减会把阈值拉低，残留的 θ 偶尔越线
    assert late <= 1


def test_run_bars_trigger_on_the_longest_run() -> None:
    """run 棒按**同向连续长度**收棒：交替段不收（run 恒为 1），连续段一拉长就收。

    这是它区别于失衡棒的地方——失衡棒只看净额，交替行情里净额在 0 附近，
    而 run 棒看的是「连续了多久」，所以能认出趋势段。
    """
    frame = make_flow_frame(np.r_[np.tile([1.0, -1.0], 30), np.full(20, 1.0)])
    bars = ib.run_bars(frame, ib.FlowBarOptions(expected_bars=5.0), ib.SIGN_NOTIONAL)
    cutoff = frame.index[60]
    early = int((bars.index < cutoff).sum())
    late = int((bars.index >= cutoff).sum())
    assert late >= 2  # 连续段（run 冲到 5 就收）至少收两根
    assert early <= 1  # 交替段 run 恒为 1、达不到期望，几乎不收


def test_does_not_look_ahead() -> None:
    """改掉最后 20 根源棒，之前的收棒时刻必须一模一样。"""
    frame = one_sided_then_quiet()
    options = ib.FlowBarOptions(expected_bars=8.0)
    before = ib.imbalance_bars(frame, options, ib.SIGN_NOTIONAL)
    changed = make_flow_frame(np.r_[np.full(40, 1.0), np.zeros(40)])
    changed.loc[changed.index[-20] :, "quote_volume"] = 1.0e6
    changed.loc[changed.index[-20] :, "taker_buy_quote"] = 0.0
    after = ib.imbalance_bars(changed, options, ib.SIGN_NOTIONAL)
    cutoff = frame.index[-20]
    assert before[before.index < cutoff].index.equals(after[after.index < cutoff].index)


def test_index_is_the_close_time_and_strictly_increasing() -> None:
    """索引取每根最后一根源棒的收盘时刻（即源时戳本身），且严格递增。"""
    frame = one_sided_then_quiet()
    bars = ib.imbalance_bars(
        frame, ib.FlowBarOptions(expected_bars=8.0), ib.SIGN_NOTIONAL
    )
    assert bars.index.name == "datetime"
    assert bool(bars.index.is_monotonic_increasing)
    assert bool(bars.index.isin(frame.index).all())


def test_drop_partial_removes_the_unfinished_tail() -> None:
    """末尾那根没达标的棒：默认丢掉，关掉则多出一根。"""
    frame = one_sided_then_quiet()
    kept = ib.imbalance_bars(
        frame,
        ib.FlowBarOptions(expected_bars=8.0, drop_partial=False),
        ib.SIGN_NOTIONAL,
    )
    dropped = ib.imbalance_bars(
        frame, ib.FlowBarOptions(expected_bars=8.0, drop_partial=True), ib.SIGN_NOTIONAL
    )
    assert len(dropped) + 1 == len(kept)


def test_empty_and_non_positive_expected_bars_yield_no_bars() -> None:
    """空输入与非法 E[T] 都不能返回假棒。"""
    empty = make_flow_frame(np.array([]))
    options = ib.FlowBarOptions(expected_bars=8.0)
    assert ib.imbalance_bars(empty, options, ib.SIGN_NOTIONAL).empty
    assert ib.imbalance_bars(empty, options, ib.SIGN_NOTIONAL).index.name == "datetime"
    frame = one_sided_then_quiet()
    zero = ib.FlowBarOptions(expected_bars=0.0)
    assert ib.imbalance_bars(frame, zero, ib.SIGN_NOTIONAL).empty
