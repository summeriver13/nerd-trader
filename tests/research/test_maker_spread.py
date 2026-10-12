"""
tests/research/test_maker_spread.py
两腿被动挂单的口径测试：只测会「静默算错」的地方——限价锚点、两腿方向、成交与否
如何进收益、以及单边敞口的识别。

数值断言全部用手算得出来的合成 1m 缓存（monkeypatch 掉缓存路径），不读真实数据。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.research import maker_conditional as mc
from src.research import maker_spread as ms

START = "2022-01-01"
DAY = pd.Timestamp(START, tz="UTC")
LONG_SYMBOL = "AAA/USDT:USDT"
SHORT_SYMBOL = "BBB/USDT:USDT"
MID_SYMBOL = "CCC/USDT:USDT"
ENTRY = 100.0
EXIT = 90.0
HORIZON = 1440
ENVELOPE = 0.5
FEES = ms.MAKER_FEE_BP + ms.TAKER_FEE_BP


def _frame(prices: list[float], spread: float = ENVELOPE) -> pd.DataFrame:
    """合成 1m 行情：每根收盘价给定，高低价按 spread 包络（保证 low ≤ close ≤ high）。"""
    index = pd.date_range(START, periods=len(prices), freq="1min", tz="UTC")
    index.name = "datetime"
    close = pd.Series(prices, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1.0,
        }
    )


def _two_day_frame() -> pd.DataFrame:
    """两天：第一天 100（入场），第二天 90（出场）。第二天不放单（horizon 越界）。"""
    return _frame([ENTRY] * HORIZON + [EXIT] * HORIZON)


def _write_cache(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    venue = root / "okx" / "swap" / "1m"
    venue.mkdir(parents=True, exist_ok=True)
    for symbol, frame in frames.items():
        frame.to_parquet(venue / f"{symbol.replace('/', '-').split(':')[0]}.parquet")


def _patch_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """maker_spread 自己导入了 crypto_bar_path，要 patch 它那个模块里的全局名。"""
    venue = root / "okx" / "swap" / "1m"
    monkeypatch.setattr(
        ms,
        "crypto_bar_path",
        lambda exchange, symbol, timeframe: (
            venue / f"{symbol.replace('/', '-').split(':')[0]}.parquet"
        ),
    )


def _context(groups: dict[str, str]) -> mc.SignalContext:
    """全可交易、截面够宽的单日上下文；分组按传入值，缺省是中间组（不参与）。"""
    index = pd.DatetimeIndex([DAY])
    columns = pd.Index([LONG_SYMBOL, SHORT_SYMBOL, MID_SYMBOL])
    return mc.SignalContext(
        groups=pd.DataFrame(groups, index=index, columns=columns),
        tradable=pd.DataFrame(True, index=index, columns=columns),
        wide=pd.Series(True, index=index),
    )


def _options(**overrides: object) -> ms.SpreadOptions:
    base = ms.SpreadOptions(
        offsets_bp=(1.0,),
        horizons_bars=(HORIZON,),
        min_cross_section=3,
        market="swap",
        timeframe="1m",
    )
    return dataclasses.replace(base, **overrides)


def _legs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pd.DataFrame:
    """输家 AAA、赢家 BBB、中间组 CCC，行情两天，只应产出一行腿明细（每天一条）。"""
    _write_cache(
        tmp_path,
        {
            LONG_SYMBOL: _two_day_frame(),
            SHORT_SYMBOL: _two_day_frame(),
            MID_SYMBOL: _two_day_frame(),
        },
    )
    _patch_paths(monkeypatch, tmp_path)
    context = _context({LONG_SYMBOL: mc.GROUP_LOSER, SHORT_SYMBOL: mc.GROUP_WINNER})
    options = _options()
    frames = [
        ms.symbol_legs(symbol, context, options)
        for symbol in (LONG_SYMBOL, SHORT_SYMBOL, MID_SYMBOL)
    ]
    return pd.concat(frames, ignore_index=True)


def test_limit_is_anchored_on_the_entry_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """限价必须锚在入场那根的收盘价上。

    曾经用全长度数组算限价、再拿筛后的下标去取，直接广播失败——多天缓存本身就是
    这条的回归用例。
    """
    legs = _legs(monkeypatch, tmp_path)
    assert len(legs) == 2
    assert set(legs["side"]) == {ms.SIDE_LONG, ms.SIDE_SHORT}


def test_loser_buys_below_and_winner_sells_above(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输家走买腿（限价在收盘下方）、赢家走卖腿（上方），中间组不参与。

    入场 100、出场 90：买腿限价 100×(1−1bp) 成交后亏，卖腿限价 100×(1+1bp)
    成交后赚——这正是反转价差要的方向。收益已扣进场 maker + 出场 taker。
    """
    legs = _legs(monkeypatch, tmp_path)
    rows = {row["side"]: row for row in legs.to_dict(orient="records")}
    long_ret = (EXIT / (ENTRY * (1 - 1e-4)) - 1.0) * ms.BP - FEES
    short_ret = -(EXIT / (ENTRY * (1 + 1e-4)) - 1.0) * ms.BP - FEES
    assert bool(rows[ms.SIDE_LONG]["hit"]) is True
    assert bool(rows[ms.SIDE_SHORT]["hit"]) is True
    assert rows[ms.SIDE_LONG]["ret_bp"] == pytest.approx(long_ret, rel=1e-9)
    assert rows[ms.SIDE_SHORT]["ret_bp"] == pytest.approx(short_ret, rel=1e-9)
    assert long_ret < 0 < short_ret


def test_unfilled_legs_count_as_zero_only_when_realized() -> None:
    """未成交按 0 计（资金空转）；「假设一定成交」是另一个口径，两者之差就是拖累。"""
    side = pd.DataFrame({"ret_bp": [100.0, 200.0], "hit": [True, False]})
    assert ms._leg_bp(side, realized=True) == pytest.approx(50.0)
    assert ms._leg_bp(side, realized=False) == pytest.approx(150.0)
    assert ms._leg_bp(side.iloc[:0], realized=True) == 0.0


def test_daily_frame_splits_legs_and_flags_one_sided_days() -> None:
    """两腿各自等权后相减；只有一腿成交的日子必须被标出来——那不是中性组合。"""
    legs = pd.DataFrame(
        [
            {
                "day": DAY,
                "offset_bp": 1.0,
                "horizon_bars": HORIZON,
                "side": ms.SIDE_LONG,
                "hit": True,
                "ret_bp": 10.0,
            },
            {
                "day": DAY,
                "offset_bp": 1.0,
                "horizon_bars": HORIZON,
                "side": ms.SIDE_LONG,
                "hit": False,
                "ret_bp": 30.0,
            },
            {
                "day": DAY,
                "offset_bp": 1.0,
                "horizon_bars": HORIZON,
                "side": ms.SIDE_SHORT,
                "hit": False,
                "ret_bp": 4.0,
            },
        ]
    )
    daily = ms.daily_frame(legs, _options())
    row = daily.iloc[0].to_dict()
    assert row["long_bp"] == pytest.approx(5.0)
    assert row["long_all_bp"] == pytest.approx(20.0)
    assert row["long_fill"] == pytest.approx(0.5)
    assert row["short_fill"] == pytest.approx(0.0)
    assert row["spread_bp"] == pytest.approx(5.0)
    assert row["spread_all_bp"] == pytest.approx(16.0)
    assert bool(row["both_fill"]) is False
    assert row["imbalance"] == pytest.approx(0.5)


def test_summary_reports_mean_and_median() -> None:
    """重尾分布下均值会被单个名字主导，中位数与胜率必须一起给。"""
    days = pd.date_range(START, periods=4, freq="1D", tz="UTC")
    daily = pd.DataFrame(
        {
            "day": days,
            "offset_bp": 1.0,
            "horizon_bars": HORIZON,
            "spread_bp": [10.0, 20.0, 30.0, -10000.0],
            "spread_all_bp": [10.0, 20.0, 30.0, -10000.0],
            "long_fill": 1.0,
            "short_fill": 1.0,
            "both_fill": True,
            "imbalance": 0.0,
        }
    )
    summary = ms.summary_frame(daily, _options())
    row = summary.iloc[0].to_dict()
    assert row["days"] == 4
    assert row["spread_bp"] == pytest.approx(-2485.0)
    assert row["median_bp"] == pytest.approx(15.0)
    assert row["win_rate"] == pytest.approx(0.75)
    assert row["worst_bp"] == pytest.approx(-10000.0)


def test_thin_cross_section_produces_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """截面不够宽的日子整日丢弃：单标的的价差不是组合。"""
    _write_cache(tmp_path, {LONG_SYMBOL: _two_day_frame()})
    _patch_paths(monkeypatch, tmp_path)
    context = _context({LONG_SYMBOL: mc.GROUP_LOSER})
    context = mc.SignalContext(
        groups=context.groups,
        tradable=context.tradable,
        wide=pd.Series(False, index=context.wide.index),
    )
    assert ms.symbol_legs(LONG_SYMBOL, context, _options()).empty
