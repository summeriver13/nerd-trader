"""微观结构特征的估计量不变量：能由已知答案反推、且不偷偷看未来。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.liquidity.microstructure import (
    MicroOptions,
    amihud_bp,
    bulk_volume_split,
    corwin_schultz_spread_bp,
    daily_microstructure,
    kyle_lambda_bp,
    roll_spread_bp,
    vpin,
)

MINUTES_PER_DAY: int = 1440


def _bounce_returns(spread_bp: float, size: int) -> np.ndarray:
    """买/卖交替成交的价格序列（中间价不动），返回 bp 收益。"""
    signs = np.where(np.arange(size) % 2 == 0, 1.0, -1.0)
    return signs * spread_bp / 2.0


def _minute_frame(
    days: int,
    *,
    price: float = 100.0,
    minutes: int = MINUTES_PER_DAY,
    seed: int = 7,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """造 d 天 1 分钟棒：价格随机游走 + 固定买卖价差。

    `start` 必须逐日错开：不同 day 用同一批时间戳时，concat 后会被按索引去重，
    测试就变成在比「同一天的两种造法」而不是「两天」。
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * minutes, freq="1min", tz="UTC")
    ret = rng.normal(0.0, 1e-4, size=index.size)
    mid = price * np.exp(np.cumsum(ret))
    half = price * 2.5e-4  # 5bp 往返价差的一半
    high = mid + half
    low = mid - half
    close = mid + np.where(np.arange(index.size) % 2 == 0, half, -half)
    volume = np.full(index.size, 100.0)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def test_roll_recovers_the_spread_of_a_pure_bid_ask_bounce() -> None:
    """只有买卖价差、没有中间价波动时，Roll 必须原样估出那个价差。"""
    # 容差 1e-4：交替序列在 499 个滞后对上买/卖个数不等，样本均值不严格为 0，
    # 估计量本身正确，偏差是 O(1/n²) 量级的小样本项
    for spread_bp in (2.0, 5.0, 20.0):
        got = roll_spread_bp(_bounce_returns(spread_bp, 500))
        assert got == pytest.approx(spread_bp, rel=1e-4)


def test_roll_is_nan_when_the_autocovariance_is_positive() -> None:
    """动量序列估不出价差（cov ≥ 0）→ nan，绝不能截成 0——那会被读成「零成本」。"""
    trending = np.linspace(0.0, 100.0, 200)
    assert np.isnan(roll_spread_bp(trending))


def test_roll_needs_enough_observations() -> None:
    assert np.isnan(roll_spread_bp(np.array([1.0, -1.0])))


def test_corwin_schultz_recovers_a_constant_spread_when_volatility_is_none() -> None:
    """两天高低幅完全一样（无波动）时，全部高低幅都是价差，S 必须等于给定价差。"""
    spread = 0.002
    high = np.full(10, 100.0 * (1.0 + spread / 2.0))
    low = np.full(10, 100.0 * (1.0 - spread / 2.0))
    got = corwin_schultz_spread_bp(high, low)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], spread * 1e4, rtol=1e-6)


def test_corwin_schultz_overstates_the_spread_when_volatility_is_large() -> None:
    """原文承认的偏差方向：有波动时把波动当价差，估出来只会偏高。"""
    rng = np.random.default_rng(3)
    spread = 0.0002
    high = 100.0 * (1.0 + spread / 2.0) * np.exp(rng.normal(0.0, 0.01, 50))
    low = 100.0 * (1.0 - spread / 2.0) * np.exp(rng.normal(0.0, 0.01, 50))
    got = corwin_schultz_spread_bp(high, low)[1:]
    assert np.nanmedian(got) > spread * 1e4


def test_corwin_schultz_clamps_free_alpha_to_zero_not_nan() -> None:
    """α<0 时按原文取 0（价差不为负）：这是口径选择，不是把失败样本算成 0 成本。"""
    flat = np.full(5, 100.0)
    got = corwin_schultz_spread_bp(flat, flat)
    assert np.all(got[1:] == 0.0)


def test_kyle_lambda_recovers_a_known_price_impact() -> None:
    rng = np.random.default_rng(11)
    lam = 1.5
    flow = rng.normal(0.0, 5.0, 400)
    returns = lam * flow + rng.normal(0.0, 0.2, 400)
    assert abs(kyle_lambda_bp(returns, flow) - lam) < 0.05


def test_kyle_lambda_is_nan_without_flow_variation() -> None:
    assert np.isnan(kyle_lambda_bp(np.arange(5.0), np.full(5, 3.0)))


def test_amihud_is_inverse_in_notional() -> None:
    returns = np.array([1.0, -2.0, 3.0, -1.0])
    assert amihud_bp(returns, 1.0) == pytest.approx(2.0 * amihud_bp(returns, 2.0))
    assert np.isnan(amihud_bp(returns, 0.0))


def test_bvc_splits_volume_by_return_direction() -> None:
    up = np.array([50.0, 60.0])
    sigma = np.array([10.0, 10.0])
    volume = np.array([10.0, 20.0])
    buy, sell = bulk_volume_split(up, volume, sigma)
    assert buy.sum() > 0.99 * volume.sum()
    down, _ = bulk_volume_split(-up, volume, sigma)
    assert down.sum() < 0.01 * volume.sum()
    flat_buy, flat_sell = bulk_volume_split(up, volume, np.zeros(2))
    np.testing.assert_allclose(flat_buy, flat_sell)


def test_vpin_is_one_for_one_sided_flow_even_when_volumes_dwarf_bar_counts() -> None:
    """回归用例：桶量按**成交量**加权。

    早先的实现在这里恒返回 nan——它用桶内地**根数**去比桶量（体积单位），量纲不符，
    「满桶」判据永远为假，于是 VPIN 全表缺失。成交量 1e6/根、只有 100 根时必现。
    """
    volume = np.full(100, 1e6)
    buy = volume.copy()
    sell = np.zeros(100)
    assert abs(vpin(buy, sell, volume) - 1.0) < 1e-9
    balanced = vpin(volume / 2.0, volume / 2.0, volume)
    assert abs(balanced) < 1e-9


def test_vpin_is_nan_without_volume() -> None:
    assert np.isnan(vpin(np.zeros(5), np.zeros(5), np.zeros(5)))


def test_daily_microstructure_groups_by_utc_day_and_drops_short_days() -> None:
    frame = _minute_frame(2, start="2024-01-01", seed=5)
    short = _minute_frame(1, start="2024-01-03", seed=6).iloc[:100]
    frame_and_short = pd.concat([frame, short])
    daily = daily_microstructure(frame_and_short, MicroOptions(min_bars=60))
    assert daily.shape[0] == 3  # 两天整天 + 100 根的半天（门槛 60 允许）
    assert isinstance(daily.index, pd.DatetimeIndex)
    # 桩里 DatetimeIndex.normalize 不存在，走 .dt 访问器（同 microstructure.py）
    stamps = pd.Series(daily.index)
    assert (stamps == stamps.dt.normalize()).all()
    strict = daily_microstructure(frame_and_short, MicroOptions(min_bars=200))
    assert strict.shape[0] == 2


def test_daily_microstructure_does_not_look_ahead() -> None:
    """后一天的数据变了，前一天的读数必须一字不变（σ 与方向都只能用前向信息）。"""
    first = _minute_frame(1, start="2024-01-01", seed=1)
    second_a = _minute_frame(1, start="2024-01-02", seed=2)
    second_b = _minute_frame(1, start="2024-01-02", seed=3)
    second_b["high"] *= 1.5
    second_b["low"] *= 0.7
    second_b["volume"] *= 9.0
    options = MicroOptions(min_bars=60)
    base = daily_microstructure(pd.concat([first, second_a]), options)
    other = daily_microstructure(pd.concat([first, second_b]), options)
    pd.testing.assert_series_equal(
        base.iloc[0], other.iloc[0], check_names=False, rtol=1e-12
    )
