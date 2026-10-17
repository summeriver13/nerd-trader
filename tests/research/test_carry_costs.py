"""carry 口径不变量：成本按持有期摊薄、缺日不静默补 0、窗口截断。

历史教训（修复前）：一次性进出费按年扣、再平衡按双腿全名义换手（2.4%/年），
把 BTC 全样本净年化从 ~8.0% 压到 5.52%；--carry-days 也完全不生效。
"""

import numpy as np
import pandas as pd
import pytest

from src.research.funding.carry import (
    ANNUAL_DAYS,
    CAPITAL_MULTIPLE,
    ROUND_TRIP_COST,
    TURNOVER_SHARE,
    build_carry_report,
    cost_drag,
    daily_funding,
    rolling_annual,
)


def _funding(days: int, rate: float = 0.0001) -> pd.Series:
    """每天 3 次结算的合成资金费率。"""
    index = pd.date_range("2024-01-01", periods=days * 3, freq="8h", tz="UTC")
    return pd.Series(rate, index=index)


def test_cost_drag_amortizes_round_trip() -> None:
    """一次性进出按持有年数摊薄：7 年持有期下 0.3%（双腿 taker）只剩 0.029%/年。"""
    amortized, rebalance = cost_drag(holding_years=7.0, turnover=TURNOVER_SHARE)
    assert amortized == pytest.approx(ROUND_TRIP_COST / CAPITAL_MULTIPLE / 7.0)
    assert amortized == pytest.approx(0.000286, abs=1e-5)
    # 再平衡按换手率：12 × 0.3% × 5% / 1.5 = 0.12%/年，而不是全名义的 3.6%
    assert rebalance == pytest.approx(12 * ROUND_TRIP_COST * 0.05 / CAPITAL_MULTIPLE)
    assert rebalance == pytest.approx(0.0012, abs=1e-5)


def test_cost_drag_short_holding_does_not_explode() -> None:
    """持有期趋零时不能除零：按最短一天摊，成本退化为年化后的单次全额。"""
    amortized, _ = cost_drag(holding_years=0.0)
    assert amortized == pytest.approx(ROUND_TRIP_COST / CAPITAL_MULTIPLE * ANNUAL_DAYS)


def test_daily_funding_counts_gaps_instead_of_filling_zero() -> None:
    """缺日要能被数出来（修复前 resample 默认补 0，缺日会静默稀释年化）。"""
    funding = _funding(10)
    gap = pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True)
    keep = ~funding.index.to_series().dt.normalize().isin(gap)
    daily, missing = daily_funding(funding.loc[keep])
    assert missing == 2
    assert (daily.loc["2024-01-03":"2024-01-04"] == 0.0).all()


def test_daily_funding_drops_incomplete_last_day() -> None:
    """最后一天只结算了 1 次（常态 3 次）视为不完整，剔除。"""
    funding = _funding(5)
    funding = funding.loc[funding.index < pd.Timestamp("2024-01-06", tz="UTC")]
    daily, _ = daily_funding(funding)
    assert len(daily) == 5  # 第 6 天被剔除
    assert str(daily.index[-1])[:10] == "2024-01-05"


def test_rolling_annual_counts_dropped_windows() -> None:
    """累计跌破 -100% 的窗口不能再取对数，但要计数而不是静默丢弃。"""
    index = pd.date_range("2024-01-01", periods=40, tz="UTC")
    daily = pd.Series([0.001] * 40, index=index)
    daily.iloc[5] = -1.5  # 亏光
    rolling, dropped = rolling_annual(daily, window=30)
    assert dropped == 6  # 覆盖该点的窗口有 6 个（起点 -29 到 0），全部剔除但计数
    assert rolling.notna().all()


def test_report_net_is_capital_basis_and_window_is_respected() -> None:
    """净年化 = 资本口径毛收益 − 摊薄进出 − 再平衡；窗口截断后年化随之改变。"""
    full = build_carry_report("TEST/USDT", _funding(365, rate=0.0002))
    assert full.days == 365
    expected = full.capital_adjusted - full.amortized_round_trip - full.rebalance_cost
    assert full.net_annual == pytest.approx(expected)
    assert full.net_annual_stress < full.net_annual  # 漂移越大成本越高

    short = build_carry_report("TEST/USDT", _funding(30, rate=0.0002))
    assert short.days == 30
    assert short.amortized_round_trip > full.amortized_round_trip  # 摊薄年数变短
    assert np.isclose(
        short.gross_annual, full.gross_annual, atol=1e-6
    )  # 费率相同则毛年化相同
