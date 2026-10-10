"""上新币效应：测试只覆盖会静默出错的地方。

四个高风险点（错了都不报错，只会让结论变好看）：

1. **生命周期分段必须是独立的**——写成累计会把「每段都在跌」读成「越跌越少」，
   两者的数字完全不同但都长得像收益曲线。
2. **首日假象的判定**——`open == low` 是「当天没有成交落在 open 附近」的判据，
   判定反了就会把参考价当成成交价，凭空造出中位 +11% 的首日收益。
3. **组合回测的成本按换手收**——误按每日固定扣一笔会把 +85% 的累计收益算成 −93%
   （实测踩过，差 90 倍）。
4. **NaN 不能当 0% 收益**——面板是各标的索引的并集，末尾未收盘那一根与上市前的空档
   都是 NaN，当成 0% 会静默污染组合日收益（实测让某个参数整列变 NaN）。
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.research.listing import (
    SEGMENT_EDGES,
    SEGMENT_LABELS,
    ListingEvent,
    _open_at_low,
    artifact_summary,
    holding_return,
    horizon_table,
    lifecycle_table,
)
from src.research.listing_portfolio import (
    _daily_turnover,
    holding_windows,
    short_backtest,
)

DAYS = 400
# stub 把 Timestamp() 的返回标成 Timestamp | NaTType，字面量运行时必为 Timestamp
LIST_TIME = cast(pd.Timestamp, pd.Timestamp("2022-06-01 06:00", tz="UTC"))
FLAT_PRICE = 100.0


def _event(
    symbol: str, closes: list[float], *, open_at_low: bool = False
) -> ListingEvent:
    """合成上市事件。"""
    return ListingEvent(
        symbol=symbol,
        list_time=LIST_TIME,
        closes=np.array(closes, dtype=float),
        open_at_low=open_at_low,
    )


@pytest.fixture
def linear_event() -> ListingEvent:
    """收盘价 = 100 + 下标：每段收益都能手算。"""
    return _event("LIN-USDT", [100.0 + index for index in range(DAYS)])


@pytest.fixture
def flat_panel() -> pd.DataFrame:
    """10 天、2 只标的的恒定价格面板：收益为 0，把成本单独隔离出来看。"""
    index = pd.date_range("2022-06-01", periods=10, freq="1D", tz="UTC")
    return pd.DataFrame({"A-USDT": FLAT_PRICE, "B-USDT": FLAT_PRICE}, index=index)


def test_holding_return_needs_enough_bars() -> None:
    """数据不够时必须返回 nan，而不是拿最后一根凑数。"""
    event = _event("S-USDT", [100.0, 110.0, 121.0])
    assert holding_return(event, 0, 1) == pytest.approx(0.10)
    assert holding_return(event, 0, 2) == pytest.approx(0.21)
    assert np.isnan(holding_return(event, 0, 3))
    assert np.isnan(holding_return(event, 1, 2))


def test_lifecycle_segments_are_independent(linear_event: ListingEvent) -> None:
    """每段都从该段起点重算，不是累计——累计口径下最后一段会等于全期总收益。"""
    table = lifecycle_table([linear_event])
    closes = linear_event.closes
    for start, stop, label in zip(
        SEGMENT_EDGES[:-1], SEGMENT_EDGES[1:], SEGMENT_LABELS, strict=True
    ):
        expected = closes[stop] / closes[start] - 1.0
        assert float(table.loc[label, "median"]) == pytest.approx(expected)
    # 首段（持有 1 天）只涨 1%，若误写成累计，末段会接近 366%
    assert float(table.iloc[-1]["median"]) < 1.0


def test_horizon_table_reports_median_and_positive_share() -> None:
    """中位与均值必须分开算：分布极度右偏时两者讲的故事相反。"""
    events = [
        _event("A-USDT", [100.0, 200.0]),
        _event("B-USDT", [100.0, 110.0]),
        _event("C-USDT", [100.0, 120.0]),
    ]
    table = horizon_table(events, horizons=(1,))
    assert float(table.loc[1, "median"]) == pytest.approx(0.20)
    assert float(table.loc[1, "mean"]) == pytest.approx((1.0 + 0.10 + 0.20) / 3)
    assert float(table.loc[1, "positive"]) == pytest.approx(1.0)


def test_open_at_low_detects_reference_price() -> None:
    """open 等于当日最低价说明没有成交落在 open，首日收益不可用。"""
    index = pd.date_range("2022-06-01", periods=2, freq="1D", tz="UTC")
    reference = pd.DataFrame(
        {
            "open": [0.10, 0.11],
            "high": [2.0, 2.1],
            "low": [0.10, 0.11],
            "close": [1.4, 1.5],
        },
        index=index,
    )
    traded = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [2.0, 2.1],
            "low": [0.9, 1.0],
            "close": [1.4, 1.5],
        },
        index=index,
    )
    assert _open_at_low(reference) is True
    assert _open_at_low(traded) is False


def test_artifact_summary_counts_share() -> None:
    """假象占比是后续判断「首日收益能不能用」的唯一依据。"""
    events = [
        _event("A-USDT", [1.0], open_at_low=True),
        _event("B-USDT", [1.0], open_at_low=False),
        _event("C-USDT", [1.0], open_at_low=True),
    ]
    summary = artifact_summary(events)
    assert summary["count"] == 3
    assert summary["open_at_low_share"] == pytest.approx(2 / 3)
    assert artifact_summary([])["open_at_low_share"] == 0.0


def test_holding_windows_locates_listing_day(flat_panel: pd.DataFrame) -> None:
    """上市时刻（06:00）不等于 bar 时间戳（00:00），仍必须定位到上市当日那一根。"""
    event = _event("A-USDT", [FLAT_PRICE] * 10)
    windows = holding_windows(
        [event], cast(pd.DatetimeIndex, flat_panel.index), entry=1, hold=3
    )
    assert windows["A-USDT"] == {2, 3, 4}


def test_turnover_counts_entries_and_exits() -> None:
    """换手 = 进出仓位占当时持仓数的比例；全部清仓那天也要收费。"""
    held = [[], ["A"], ["A", "B"], ["A", "B"], ["B"], []]
    positions = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    turnover = _daily_turnover(held, positions)
    assert turnover[1] == pytest.approx(1.0)  # 建仓 A，1 个仓位里换 1 个
    assert turnover[2] == pytest.approx(0.5)  # B 进场，2 个仓位里换 1 个
    assert turnover[3] == pytest.approx(0.0)  # 持仓不动
    assert turnover[4] == pytest.approx(1.0)  # A 离场，1 个仓位里换 1 个
    assert turnover[5] == pytest.approx(1.0)  # 清仓，分母回退到前一日


def test_cost_follows_turnover_not_calendar_days(flat_panel: pd.DataFrame) -> None:
    """建仓与清仓各收一次，持仓期间不再收——按天收会把成本放大到 90 倍。"""
    windows = {"A-USDT": {1, 2, 3}, "B-USDT": {1, 2, 3}}
    result = short_backtest(flat_panel, windows, entry=1, hold=3, fee_one_way=0.01)
    # 只发生两次换手（第 1 天建仓、第 4 天清仓），各 100%
    assert result.total_return == pytest.approx(0.99**2 - 1.0, rel=1e-9)
    assert result.days == 4


def test_nan_returns_are_skipped() -> None:
    """价格缺失的那一天不能进组合：否则日收益变 NaN，整条曲线跟着废掉。"""
    index = pd.date_range("2022-06-01", periods=7, freq="1D", tz="UTC")
    prices = pd.DataFrame(
        {"A-USDT": [100.0, 101.0, 102.0, np.nan, 104.0, 105.0, 106.0]}, index=index
    )
    event = _event("A-USDT", [100.0] * 7)
    windows = holding_windows(
        [event], cast(pd.DatetimeIndex, prices.index), entry=0, hold=5
    )
    result = short_backtest(prices, windows, entry=0, hold=5, fee_one_way=0.0)
    assert not np.isnan(result.total_return)
    assert not np.isnan(result.annual_compounded)
    # 第 3 根是 NaN、第 4 根也连带 NaN，两天都不进组合
    assert result.days < len(prices)


def test_flat_prices_with_no_fee_earn_nothing(flat_panel: pd.DataFrame) -> None:
    """零收益零成本必须严格是 0：任何非零都说明口径里混进了别的东西。"""
    windows = {"A-USDT": set(range(1, 6))}
    result = short_backtest(flat_panel, windows, entry=1, hold=5, fee_one_way=0.0)
    assert result.total_return == pytest.approx(0.0, abs=1e-12)
    assert result.sharpe == pytest.approx(0.0, abs=1e-12)
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_short_pnl_is_negative_of_long_return() -> None:
    """做空日收益 = −多头日收益：符号错了整个结论会反过来。"""
    index = pd.date_range("2022-06-01", periods=4, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A-USDT": [100.0, 90.0, 99.0, 99.0]}, index=index)
    event = _event("A-USDT", [100.0] * 4)
    windows = holding_windows(
        [event], cast(pd.DatetimeIndex, prices.index), entry=0, hold=3
    )
    result = short_backtest(prices, windows, entry=0, hold=3, fee_one_way=0.0)
    # 下跌 10% 后再涨 10%：空头净赚约 1%，回撤来自反弹那一段
    assert result.total_return == pytest.approx(0.9 * 1.1 - 1.0, rel=1e-9)
    assert result.max_drawdown < 0.0


def test_price_panel_keeps_union_index() -> None:
    """并集索引是 NaN 的来源，这里钉住它的形状以便上面的 NaN 测试有意义。"""
    index = pd.date_range("2022-06-01", periods=3, freq="1D", tz="UTC")
    panel = pd.DataFrame({"A-USDT": [1.0, 2.0, 3.0]}, index=index)
    shifted = pd.DataFrame({"B-USDT": [4.0, 5.0]}, index=index[1:])
    merged = pd.concat([panel, shifted], axis=1)
    assert cast(pd.DataFrame, merged).shape == (3, 2)
    assert bool(pd.isna(merged.loc[index[0], "B-USDT"]))
