"""资金费率进入向量化回测的口径：多头付正费率、空头收正费率，按持仓名义×杠杆。"""

import numpy as np
import pandas as pd
import pytest

from src.research.backtest.engine import BacktestEngine
from src.research.domains.crypto import CryptoDomain


def _frame(signal: list[float]) -> pd.DataFrame:
    """无涨跌的价格序列 + 给定信号：隔离出资金费率这一项。"""
    index = pd.date_range("2026-01-01", periods=len(signal), freq="1D", tz="UTC")
    return pd.DataFrame(
        {"close": 100.0, "high": 100.0, "low": 100.0, "signal": signal}, index=index
    )


def _funding(days: int, rate: float) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=days, freq="1D", tz="UTC")
    return pd.Series(rate, index=index)


def test_long_pays_and_short_receives_funding() -> None:
    """同一份正费率：多头付钱（成本为正）、空头收钱（成本为负），信号次日生效。"""
    domain = CryptoDomain(fee_rate=0.0, slippage=0.0)
    long_only = BacktestEngine(
        domain, initial_capital=10_000.0, funding=_funding(4, 0.001)
    ).run(_frame([1.0, 1.0, 1.0, 1.0]))
    short_only = BacktestEngine(
        domain, initial_capital=10_000.0, funding=_funding(4, 0.001)
    ).run(_frame([-1.0, -1.0, -1.0, -1.0]))

    # 仓位在信号次日生效：4 根里只有后 3 根持仓，各付 0.1%
    assert long_only["total_funding"] == pytest.approx(0.003)
    assert short_only["total_funding"] == pytest.approx(-0.003)


def test_funding_scales_with_leverage_and_aligns_intraday() -> None:
    """资金费率成本按持仓名义×杠杆计；日内 K 线分摊当天的日费率。"""
    index = pd.date_range("2026-01-01", periods=8, freq="12h", tz="UTC")
    frame = pd.DataFrame(
        {"close": 100.0, "high": 100.0, "low": 100.0, "signal": 1.0}, index=index
    )
    funding = _funding(4, 0.001)
    result = BacktestEngine(
        CryptoDomain(fee_rate=0.0, slippage=0.0),
        initial_capital=10_000.0,
        leverage=2.0,
        funding=funding,
    ).run(frame)

    # 8 根 12h = 4 天，每天 0.1% 由当天 2 根分摊（每根 0.05%）；
    # 8 根里只有 7 根持仓（信号次日生效）× 0.05% × 2 倍杠杆 = 0.7%
    assert result["total_funding"] == pytest.approx(0.007)
    assert np.isclose(
        result["total_funding"], float(result["df"]["funding_cost"].sum())
    )


def test_missing_funding_days_count_as_zero_not_error() -> None:
    """缓存没覆盖到的日子按 0 计（覆盖情况由取数侧打印），不抛异常、不当成缺失。"""
    result = BacktestEngine(
        CryptoDomain(fee_rate=0.0, slippage=0.0),
        initial_capital=10_000.0,
        funding=_funding(2, 0.001),
    ).run(_frame([1.0, 1.0, 1.0, 1.0]))

    assert result["total_funding"] == pytest.approx(0.001)


def test_no_funding_reports_none_instead_of_zero() -> None:
    """未注入资金费率时返回 None：报 0 会被读成「已计入且为零」。"""
    result = BacktestEngine(
        CryptoDomain(fee_rate=0.0, slippage=0.0), initial_capital=10_000.0
    ).run(_frame([1.0, 1.0]))

    assert result["total_funding"] is None
    assert result["funding_annual"] is None
