"""多标的时序动量（TSMOM）的不变量：无前视、1/N_t、多速度平均、只缩不放。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.tsmom import (
    TsmomOptions,
    active_counts,
    cross_section_mean,
    donchian_signal,
    multi_speed_signal,
    portfolio_returns,
    run_symbol,
    tsmom_signal,
    volatility_scale,
)

BARS = 400
ENTRY = 20


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
    """单调上行序列：短窗口必定先突破。"""
    return make_bars(100.0 * np.cumprod(1.0 + np.full(periods, drift)))


def test_cross_section_mean_ignores_missing_symbols() -> None:
    """等权 1/N_t：缺失的标的退出分母，不是补零。"""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    a = pd.Series([1.0, 2.0, 3.0], index=index)
    b = pd.Series([3.0, np.nan, 5.0], index=index)
    mean = cross_section_mean(pd.DataFrame({"a": a, "b": b}))
    assert mean.iloc[0] == pytest.approx(2.0)  # (1 + 3) / 2
    assert mean.iloc[1] == pytest.approx(2.0)  # 只剩 a
    assert mean.iloc[2] == pytest.approx(4.0)  # (3 + 5) / 2
    counts = active_counts({"a": a, "b": b})
    assert counts.tolist() == [2, 1, 2]


def test_portfolio_returns_equals_manual_mean() -> None:
    """组合收益就是逐行截面均值。"""
    index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    streams = {
        "x": pd.Series([0.10, -0.02], index=index),
        "y": pd.Series([0.00, 0.04], index=index),
    }
    out = portfolio_returns(streams)
    assert out.tolist() == pytest.approx([0.05, 0.01])


def test_multi_speed_signal_is_the_average_of_speeds() -> None:
    """多速度 = 各速度 0/1 信号取平均，取值只能是 0、1/n、…、1。"""
    bars = trending()
    options = TsmomOptions(entry_periods=(10, 20, 40), vol_target=None)
    signal = multi_speed_signal(bars, options)
    manual = np.mean(
        [
            donchian_signal(bars, entry).to_numpy()
            for entry in options.entry_periods
        ],
        axis=0,
    )
    assert np.allclose(signal.to_numpy(), manual)
    values = np.unique(signal.to_numpy())
    allowed = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    for value in values:
        assert np.isclose(value, allowed).any(), value
    assert np.isclose(values.max(), 1.0)  # 单调上行序列最终三个窗口全开


def test_signal_does_not_look_ahead() -> None:
    """改动未来行情不得改变历史信号（引擎再滞后一根，故 t+1 也不受影响）。"""
    closes = 100.0 * np.cumprod(1.0 + np.sin(np.arange(BARS) / 7.0) * 0.03)
    bars = make_bars(closes)
    options = TsmomOptions(vol_target=None)
    before = tsmom_signal(bars, options)

    tampered = bars.copy()
    tampered.iloc[200:, tampered.columns.get_loc("close")] *= 5.0
    tampered.iloc[200:, tampered.columns.get_loc("high")] *= 5.0
    after = tsmom_signal(tampered, options)
    assert np.allclose(before.to_numpy()[:200], after.to_numpy()[:200])


def test_volatility_scale_is_capped_and_zero_in_warmup() -> None:
    """只缩不放：缩放系数 ≤ 1，暖机期为 0（不持仓）。"""
    bars = trending(drift=0.02)
    options = TsmomOptions(vol_lookback=30)
    scale = volatility_scale(bars, options)
    assert float(scale.max()) <= 1.0
    assert float(scale.iloc[0]) == 0.0
    assert float(scale.iloc[options.vol_lookback]) > 0.0


def test_volatility_scale_shrinks_high_vol_positions() -> None:
    """同样的目标波动下，实现波动越大、仓位越小。"""
    rng = np.random.default_rng(20261014)
    quiet = make_bars(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, BARS)))
    wild = make_bars(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.05, BARS)))
    options = TsmomOptions(vol_lookback=30)
    assert float(volatility_scale(quiet, options).iloc[-1]) > float(
        volatility_scale(wild, options).iloc[-1]
    )


def test_tsmom_signal_without_vol_target_equals_multi_speed() -> None:
    """关掉波动率缩放时，给引擎的信号就是多速度信号本身。"""
    bars = trending()
    options = TsmomOptions(vol_target=None)
    assert tsmom_signal(bars, options).equals(multi_speed_signal(bars, options))


def test_run_symbol_matches_buy_and_hold_returns() -> None:
    """买入持有：首根无仓位、第二根吃一次 10bp 成本，其后逐根等于涨跌幅。"""
    closes = 100.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.01, BARS))
    bars = make_bars(closes)
    run = run_symbol(bars, pd.Series(1.0, index=bars.index))
    pct = bars["close"].pct_change().fillna(0.0)
    assert float(run.pos.iloc[0]) == 0.0
    assert float(run.pos.iloc[1]) == 1.0
    assert float(run.net_ret.iloc[0]) == 0.0
    assert float(run.net_ret.iloc[1]) == pytest.approx(float(pct.iloc[1]) - 0.001)
    assert np.allclose(
        run.net_ret.to_numpy()[2:], pct.to_numpy()[2:]
    )


def test_run_symbol_charges_more_for_churning_signal() -> None:
    """换手越高成本越高：逐根翻转的信号比常开信号贵一个量级。"""
    bars = trending(drift=0.001)
    always = run_symbol(bars, pd.Series(1.0, index=bars.index))
    flip = pd.Series(
        [1.0 if i % 2 == 0 else 0.0 for i in range(BARS)], index=bars.index
    )
    churn = run_symbol(bars, flip)
    assert float(churn.cost.sum()) > 10 * float(always.cost.sum())
