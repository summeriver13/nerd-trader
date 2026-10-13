"""多标的时序动量（TSMOM）的不变量：无前视、1/N_t、多速度平均、只缩不放。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.tsmom import (
    TsmomOptions,
    active_counts,
    book_scale,
    cross_section_mean,
    donchian_short_signal,
    donchian_signal,
    multi_speed_signal,
    portfolio_returns,
    run_symbol,
    split_portfolio,
    trailing_volatility,
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
        [donchian_signal(bars, entry).to_numpy() for entry in options.entry_periods],
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
    assert np.allclose(run.net_ret.to_numpy()[2:], pct.to_numpy()[2:])


def test_trailing_volatility_matches_manual_std() -> None:
    """滚动实现波动 = 收益率滚动标准差 × √365。"""
    rng = np.random.default_rng(20261014)
    bars = make_bars(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, BARS)))
    lookback = 30
    got = trailing_volatility(bars, lookback)
    manual = bars["close"].pct_change().rolling(lookback).std() * np.sqrt(365)
    assert np.allclose(
        got.to_numpy()[lookback:], manual.to_numpy()[lookback:], equal_nan=True
    )


def test_book_scale_hits_cap_when_book_is_quiet() -> None:
    """账本波动低于目标时放大（顶到上限），暖机期为 0，任何时刻不超过上限。"""
    index = pd.date_range("2024-01-01", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    quiet = pd.Series(rng.normal(0.0, 0.0002, 600), index=index)
    wild = pd.Series(rng.normal(0.0, 0.02, 600), index=index)
    scale_quiet = book_scale(quiet, target=0.40, cap=3.0)
    scale_wild = book_scale(wild, target=0.40, cap=3.0)
    assert float(scale_quiet.iloc[-1]) == pytest.approx(3.0)
    assert float(scale_wild.iloc[-1]) == pytest.approx(1.0, abs=0.25)
    assert float(scale_quiet.max()) <= 3.0
    assert float(scale_quiet.iloc[0]) == 0.0


def test_split_portfolio_picks_the_right_half() -> None:
    """按波动分位切两半：低波动那一半只含低波动标的。"""
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    streams = {
        "quiet": pd.Series(0.01, index=index),
        "mid": pd.Series(0.02, index=index),
        "loud": pd.Series(0.03, index=index),
        "wild": pd.Series(0.04, index=index),
    }
    vols = {
        "quiet": pd.Series(0.2, index=index),
        "mid": pd.Series(0.3, index=index),
        "loud": pd.Series(0.4, index=index),
        "wild": pd.Series(0.5, index=index),
    }
    low = split_portfolio(streams, vols, low=True)
    high = split_portfolio(streams, vols, low=False)
    assert low.tolist() == pytest.approx([0.015] * 4)  # (0.01 + 0.02) / 2
    assert high.tolist() == pytest.approx([0.035] * 4)  # (0.03 + 0.04) / 2


def test_split_portfolio_skips_missing_volatility_and_thin_halves() -> None:
    """没有波动读数的标的退出分母；半池成员少于门槛时该日为空（避免单标的爆掉读成 −100%）。"""
    index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    streams = {
        "a": pd.Series([0.02, 0.02], index=index),
        "b": pd.Series([0.04, 0.04], index=index),
    }
    vols = {
        "a": pd.Series([0.2, np.nan], index=index),
        "b": pd.Series([0.4, 0.4], index=index),
    }
    low = split_portfolio(streams, vols, low=True)
    assert float(low.iloc[0]) == pytest.approx(0.02)
    assert np.isnan(float(low.iloc[1]))
    thin = split_portfolio(streams, vols, low=True, min_members=2)
    assert np.isnan(float(thin.iloc[0]))


def test_donchian_short_signal_mirrors_long_breakout() -> None:
    """下跌序列开空后一直持有（−1），上涨序列从不做空（取值只有 0/—1）。

    用几何下跌（每日 −1.5%）：`make_bars` 的 high/low 是 ±1% 带，日跌幅不够就永远
    破不了前 entry 根的最低 low，空头根本不会开仓。
    """
    down = make_bars(100.0 * (0.985 ** np.arange(BARS)))
    up = make_bars(40.0 * (1.015 ** np.arange(BARS)))
    short_down = donchian_short_signal(down, ENTRY)
    short_up = donchian_short_signal(up, ENTRY)
    assert set(np.unique(short_down.to_numpy()).tolist()) <= {0.0, -1.0}
    assert float(short_down.iloc[-1]) == -1.0
    assert float(short_up.abs().sum()) == 0.0


def test_short_signal_has_no_lookahead() -> None:
    """改动 200 根之后的行情不改变之前的空头信号。"""
    closes = 100.0 + np.sin(np.arange(BARS) / 7.0) * 10.0
    tweaked = closes.copy()
    tweaked[200:] *= 0.5
    base = donchian_short_signal(make_bars(closes), ENTRY)
    later = donchian_short_signal(make_bars(tweaked), ENTRY)
    pd.testing.assert_series_equal(base.iloc[:200], later.iloc[:200])


def test_multi_speed_long_short_signal_is_signed() -> None:
    """多空口径：每档先合成 ±1 再跟速度取平均，取值落在 {−1,−2/3,−1/3,0,1/3,2/3,1}。"""
    closes = 100.0 + np.sin(np.arange(1500) / 11.0) * 18.0
    signal = tsmom_signal(
        make_bars(closes), TsmomOptions(vol_target=None, allow_short=True)
    )
    values = np.unique(np.round(signal.to_numpy(), 6)).tolist()
    allowed = np.round([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0], 6).tolist()
    assert set(values) <= set(allowed)
    assert float(signal.min()) < 0.0 < float(signal.max())


def test_allow_short_defaults_to_long_only() -> None:
    """默认口径不变：只多时信号非负，且与显式 allow_short=False 逐位相等。"""
    closes = 100.0 + np.sin(np.arange(BARS) / 9.0) * 12.0
    bars = make_bars(closes)
    default = tsmom_signal(bars, TsmomOptions())
    explicit = tsmom_signal(bars, TsmomOptions(allow_short=False))
    pd.testing.assert_series_equal(default, explicit)
    assert float(default.min()) >= 0.0


def test_funding_is_charged_on_position() -> None:
    """资金费按仓位逐根计：净收益差恰为 −pos × 日费率（正费率 = 多头付钱）。"""
    closes = 100.0 + np.sin(np.arange(BARS) / 9.0) * 12.0
    bars = make_bars(closes)
    signal = tsmom_signal(bars, TsmomOptions(vol_target=None))
    rate = 0.0003
    bare = run_symbol(bars, signal)
    charged = run_symbol(bars, signal, funding=pd.Series(rate, index=bars.index))
    diff = charged.net_ret.to_numpy(dtype=float) - bare.net_ret.to_numpy(dtype=float)
    np.testing.assert_allclose(diff, -bare.pos.to_numpy(dtype=float) * rate)


def test_zero_funding_matches_no_funding_argument() -> None:
    """零费率与不传资金费完全等价（不能因为传参多出任何成本）。"""
    closes = 100.0 + np.sin(np.arange(BARS) / 9.0) * 12.0
    bars = make_bars(closes)
    signal = tsmom_signal(bars, TsmomOptions(vol_target=None))
    zeros = pd.Series(0.0, index=bars.index)
    pd.testing.assert_series_equal(
        run_symbol(bars, signal).net_ret,
        run_symbol(bars, signal, funding=zeros).net_ret,
    )


def test_short_leg_collects_positive_funding() -> None:
    """正费率下空头收钱：单调下跌且开多空的信号，带资金费比不带赚更多。"""
    bars = make_bars(100.0 * (0.985 ** np.arange(BARS)))
    signal = tsmom_signal(bars, TsmomOptions(vol_target=None, allow_short=True))
    assert float(signal.iloc[-1]) < 0.0
    funding = pd.Series(0.0005, index=bars.index)
    bare = run_symbol(bars, signal)
    charged = run_symbol(bars, signal, funding=funding)
    assert float(charged.net_ret.sum()) > float(bare.net_ret.sum())


def test_run_symbol_charges_more_for_churning_signal() -> None:
    """换手越高成本越高：逐根翻转的信号比常开信号贵一个量级。"""
    bars = trending(drift=0.001)
    always = run_symbol(bars, pd.Series(1.0, index=bars.index))
    flip = pd.Series(
        [1.0 if i % 2 == 0 else 0.0 for i in range(BARS)], index=bars.index
    )
    churn = run_symbol(bars, flip)
    assert float(churn.cost.sum()) > 10 * float(always.cost.sum())
