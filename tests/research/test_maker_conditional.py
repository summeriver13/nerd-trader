"""
tests/research/test_maker_conditional.py
条件化挂单的口径测试：只测会「静默算错」的地方——成交判定、挂单收益与条件漂移的
恒等式（一阶）、截面宽度门槛、以及三个派生指标的定义。

数值断言全部用手算得出来的合成 1m 缓存（monkeypatch 掉缓存路径），不读真实数据。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.liquidity import maker_conditional as mc

START = "2022-01-01"
DAY = pd.Timestamp(START, tz="UTC")


def _bars(minutes: int, price: float = 100.0, low: float = 99.5) -> pd.DataFrame:
    """合成 1m 行情：收盘价恒定、low 恒定，便于手算「挂单价有没有被触及」。

    索引名必须叫 datetime：缓存契约里时戳就在这个索引上，模块靠 reset_index 取它。
    """
    index = pd.date_range(START, periods=minutes, freq="1min", tz="UTC")
    index.name = "datetime"
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": low,
            "close": price,
            "volume": 1.0,
        },
        index=index,
    )


def _write_cache(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    """按 {venue}/{market}/{timeframe}/{symbol}.parquet 写最小缓存。"""
    venue = root / "okx" / "swap" / "1m"
    venue.mkdir(parents=True, exist_ok=True)
    for symbol, frame in frames.items():
        frame.to_parquet(venue / f"{symbol.replace('/', '-').split(':')[0]}.parquet")


def _patch_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """把模块里的缓存路径指到 tmp_path（模块内按全局名调用，patch 全局名即可）。"""
    venue = root / "okx" / "swap" / "1m"
    monkeypatch.setattr(mc, "crypto_dataset_dir", lambda *args, **kwargs: venue)
    monkeypatch.setattr(
        mc,
        "crypto_bar_path",
        lambda exchange, symbol, timeframe: (
            venue / f"{symbol.replace('/', '-').split(':')[0]}.parquet"
        ),
    )


def _context(symbols: list[str], group: str = mc.GROUP_LOSER) -> mc.SignalContext:
    """全可交易、截面够宽、全部落在同一分组的上下文（单日）。"""
    index = pd.DatetimeIndex([DAY])
    columns = pd.Index(symbols)
    return mc.SignalContext(
        groups=pd.DataFrame(group, index=index, columns=columns),
        tradable=pd.DataFrame(True, index=index, columns=columns),
        wide=pd.Series(True, index=index),
    )


# === 时戳归一：每日首根 ===


def test_first_of_day_flags_one_bar_per_calendar_day() -> None:
    """跨两天的分钟序列里，每天恰好标出一根首根（含跨月与跨年边界）。

    序列首根也算「当日首根」：它是该标的的第一根，之后每天才是真正的 00:00。
    """
    index = pd.date_range("2022-12-31 23:58", periods=5, freq="1min", tz="UTC")
    flags = mc._first_of_day(index)
    assert flags.tolist() == [True, False, True, False, False]
    stamps = mc._day_stamps(index)
    assert stamps[0] == pd.Timestamp("2022-12-31", tz="UTC")
    assert stamps[-1] == pd.Timestamp("2023-01-01", tz="UTC")


# === 成交判定与恒等式 ===


def test_fill_requires_touching_the_limit_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """买单在收盘价下方 offset：下一根 low ≤ 挂单价才算成交。

    收盘 100、下一根 low 99.5 → 1bp 的挂单价 99.99 高于 99.5（成交）、
    60bp 的挂单价 99.40 低于 99.5（不成交）。
    """
    _write_cache(tmp_path, {"AAA/USDT:USDT": _bars(3)})
    _patch_paths(monkeypatch, tmp_path)
    options = mc.MakerOptions(offsets_bp=(1.0, 60.0), horizons_bars=(1,))
    events = mc.symbol_events("AAA-USDT", _context(["AAA-USDT"]), options)
    hit = dict(zip(events["offset_bp"], events["hit"], strict=True))
    assert bool(hit[1.0]) is True
    assert bool(hit[60.0]) is False


def test_maker_return_equals_cond_plus_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """恒等式（一阶）：挂单收益 = 条件漂移 + 挂单偏移，见 docs/execution 的机制分解。

    收盘恒定 100、下一根 low 99.5：2bp 挂单价 99.98 必然成交，且到期价 = 100，
    所以条件漂移恰为 0、挂单收益恰为 100/99.98 − 1 ≈ 2.0004bp。
    """
    _write_cache(tmp_path, {"AAA/USDT:USDT": _bars(3)})
    _patch_paths(monkeypatch, tmp_path)
    options = mc.MakerOptions(offsets_bp=(2.0,), horizons_bars=(1,))
    events = mc.symbol_events("AAA-USDT", _context(["AAA-USDT"]), options)
    assert bool(events["hit"].iloc[0]) is True
    assert events["cond_bp"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert events["maker_bp"].iloc[0] == pytest.approx(2.0004, abs=1e-3)


def test_thin_cross_section_days_are_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """截面宽度不够的交易日整日丢弃：那种日子里的「分位」不是截面。"""
    _write_cache(tmp_path, {"AAA/USDT:USDT": _bars(3)})
    _patch_paths(monkeypatch, tmp_path)
    context = _context(["AAA-USDT"])
    narrow = mc.SignalContext(
        groups=context.groups,
        tradable=context.tradable,
        wide=pd.Series(False, index=context.wide.index),
    )
    options = mc.MakerOptions(offsets_bp=(1.0,), horizons_bars=(1,))
    assert mc.symbol_events("AAA-USDT", narrow, options).empty
    assert not mc.symbol_events("AAA-USDT", context, options).empty


def test_horizon_beyond_the_cache_produces_no_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """持有期超出缓存末端的事件必须整条丢弃：拿不到到期价就没有收益可算。"""
    _write_cache(tmp_path, {"AAA/USDT:USDT": _bars(3)})
    _patch_paths(monkeypatch, tmp_path)
    options = mc.MakerOptions(offsets_bp=(1.0,), horizons_bars=(4320,))
    assert mc.symbol_events("AAA-USDT", _context(["AAA-USDT"]), options).empty


# === 派生指标的定义 ===


def test_cell_metrics_follow_the_definitions() -> None:
    """三个派生指标：逆向选择、每笔优势、每次期望，全部能手算。"""
    cell = mc.MakerCell(
        group=mc.GROUP_LOSER,
        offset_bp=1.0,
        horizon_bars=1440,
        events=100,
        fill_rate=0.8,
        maker_bp=10.0,
        cond_bp=7.0,
        base_bp=9.0,
    )
    assert cell.adverse_bp == pytest.approx(-2.0)
    # (挂单 10 − maker 2) − (市价 9 − taker 10) = 8 − (−1) = 9
    assert cell.per_trade_bp == pytest.approx(9.0)
    # 成交率 0.8 × (10 − 2) = 6.4
    assert cell.expected_bp == pytest.approx(6.4)


def test_aggregate_covers_every_cell() -> None:
    """聚合必须覆盖每个 (组, 偏移, 持有) 组合，且市价基准用全样本。"""
    events = pd.DataFrame(
        {
            "group": [mc.GROUP_LOSER] * 4,
            "offset_bp": [0.5, 0.5, 1.0, 1.0],
            "horizon_bars": [1440] * 4,
            "hit": [True, False, True, True],
            "maker_bp": [3.0, 0.0, 4.0, 5.0],
            "cond_bp": [2.0, -1.0, 3.0, 4.0],
        }
    )
    cells = mc.aggregate(events, mc.MakerOptions())
    assert len(cells) == 2
    first = cells[cells["offset_bp"] == 0.5].iloc[0]
    assert int(first["events"]) == 2
    assert first["fill_rate"] == pytest.approx(0.5)
    assert first["maker_bp"] == pytest.approx(3.0)
    assert first["cond_bp"] == pytest.approx(2.0)
    # 市价基准用全样本（含未成交的那一笔：−1bp）
    assert first["base_bp"] == pytest.approx(0.5)


def test_annual_frame_splits_by_calendar_year() -> None:
    """年度分解：结论随 regime 翻转，必须能按自然年切开。"""
    events = pd.DataFrame(
        {
            "day": pd.to_datetime(["2022-06-01", "2022-06-02", "2023-06-01"], utc=True),
            "group": [mc.GROUP_LOSER] * 3,
            "hit": [True, True, True],
            "maker_bp": [1.0, 3.0, 5.0],
            "cond_bp": [0.0, 2.0, 4.0],
        }
    )
    annual = mc.annual_frame(events)
    assert annual["year"].tolist() == [2022, 2023]
    assert annual["maker_bp"].tolist() == pytest.approx([2.0, 5.0])
    assert annual["fill_rate"].tolist() == pytest.approx([1.0, 1.0])
