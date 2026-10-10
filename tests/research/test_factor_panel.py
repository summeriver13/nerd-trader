"""
面板层的池子口径：非加密仪器剔除、上市天数、点内时间的流动性掩码。

历史教训：截面因子结论的第一杀手是池子口径。若掩码用全样本（或未来）成交额筛选，
低流动性标的会带着「后来才有的成交额」进入早期截面，小市值/波动率类因子的收益会被
系统性抬高，而代码不会报任何错。
"""

from pathlib import Path

import pandas as pd
import pytest

from src.research.factors.panel import (
    Panel,
    UniverseRule,
    listing_age_days,
    load_panel,
    tradable_counts,
    tradable_mask,
)

TIMEFRAME = "1d"


def _write_cache(
    root: Path, symbol: str, start: str, days: int, notional: float
) -> None:
    """在临时目录写一份最小缓存：只有面板需要的 close/volume 两列。"""
    index = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    frame = pd.DataFrame(
        {"close": 100.0, "volume": notional / 100.0},
        index=index,
    )
    target = root / "okx" / "spot" / TIMEFRAME
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{symbol}.parquet")


def test_load_panel_requires_cache(tmp_path: Path) -> None:
    """缓存目录不存在时直接报错，不静默返回空面板。"""
    with pytest.raises(FileNotFoundError):
        load_panel(TIMEFRAME, root=tmp_path)


def test_non_crypto_instruments_are_excluded(tmp_path: Path) -> None:
    """代币化股票（X 前缀）与测试仪器被剔除，X 前缀的真加密资产保留。"""
    for symbol in ("BTC-USDT", "XRP-USDT", "XTSLA-USDT", "XTESTA-USDT"):
        _write_cache(tmp_path, symbol, "2020-01-01", 200, 5e6)

    panel = load_panel(TIMEFRAME, root=tmp_path)
    assert list(panel.symbols) == ["BTC-USDT", "XRP-USDT"]
    assert panel.excluded == ("XTESTA-USDT", "XTSLA-USDT")

    kept = load_panel(TIMEFRAME, rule=UniverseRule(keep_non_crypto=True), root=tmp_path)
    assert set(kept.symbols) == {"BTC-USDT", "XRP-USDT", "XTSLA-USDT", "XTESTA-USDT"}


def test_listing_age_days_counts_from_first_bar(tmp_path: Path) -> None:
    """上市天数按各标的自己的首根 K 线算；上市之前是 NaN，不是负数。"""
    _write_cache(tmp_path, "OLD-USDT", "2020-01-01", 100, 5e6)
    _write_cache(tmp_path, "NEW-USDT", "2020-01-21", 80, 5e6)

    age = listing_age_days(load_panel(TIMEFRAME, root=tmp_path))
    assert age["OLD-USDT"].iloc[0] == 0
    assert age["OLD-USDT"].iloc[50] == 50
    assert pd.isna(age["NEW-USDT"].iloc[0])  # 尚未上市
    assert age["NEW-USDT"].iloc[20] == 0


def test_tradable_mask_needs_full_liquidity_window(tmp_path: Path) -> None:
    """流动性窗口不满时不可交易：不用半截窗口的名义额放行。"""
    _write_cache(tmp_path, "AAA-USDT", "2020-01-01", 60, 5e6)
    rule = UniverseRule(min_listed_days=0, liquidity_window=30)

    mask = tradable_mask(load_panel(TIMEFRAME, root=tmp_path), rule)
    assert not mask["AAA-USDT"].iloc[:29].any()  # 前 29 根窗口没满
    assert mask["AAA-USDT"].iloc[29:].all()


def test_tradable_mask_is_point_in_time(tmp_path: Path) -> None:
    """成交额突然放大不会立刻翻转掩码：滚动中位数要过半才会。"""
    index = pd.date_range("2020-01-01", periods=140, freq="1D", tz="UTC")
    volume = pd.Series(1.0, index=index)  # 名义额 = 100，远低于门槛
    volume.iloc[100:] = 100_000.0  # 第 100 根起名义额 = 1e7
    target = tmp_path / "okx" / "spot" / TIMEFRAME
    target.mkdir(parents=True)
    pd.DataFrame({"close": 100.0, "volume": volume}).to_parquet(
        target / "AAA-USDT.parquet"
    )

    rule = UniverseRule(min_listed_days=0, liquidity_window=30)
    mask = tradable_mask(load_panel(TIMEFRAME, root=tmp_path), rule)["AAA-USDT"]
    assert not mask.loc[index[100:113]].any()  # 窗口内低价根仍占多数
    assert mask.loc[index[115:]].all()  # 过半之后才翻转
    assert mask.loc[index[:99]].sum() == 0  # 早期不因未来的成交额而放行


def test_tradable_mask_requires_listing_age(tmp_path: Path) -> None:
    """成交额再高也要等上市满期数：避开上币初期的行情。"""
    _write_cache(tmp_path, "AAA-USDT", "2020-01-01", 400, 5e7)
    panel = load_panel(TIMEFRAME, root=tmp_path)

    mask = tradable_mask(panel, UniverseRule(min_listed_days=180, liquidity_window=30))
    assert not mask["AAA-USDT"].iloc[:180].any()  # 首根起算，第 180 天年龄才到 180
    assert mask["AAA-USDT"].iloc[180:].all()
    assert tradable_counts(mask).max() == 1


def _panel_with(notional_usd: float, days: int) -> Panel:
    """构造一个单标的合成面板（掩码逻辑的单元测试用）。"""
    index = pd.date_range("2020-01-01", periods=days, freq="1D", tz="UTC")
    close = pd.DataFrame({"AAA-USDT": 100.0}, index=index)
    volume = pd.DataFrame({"AAA-USDT": notional_usd / 100.0}, index=index)
    return Panel(
        timeframe=TIMEFRAME,
        close=close,
        volume=volume,
        notional=close * volume,
    )


def test_mask_respects_notional_threshold() -> None:
    """名义额门槛是硬条件：刚好低于门槛不可交易。"""
    rule = UniverseRule(min_notional_usd=1e6, min_listed_days=0, liquidity_window=30)
    assert not tradable_mask(_panel_with(999_999.0, 60), rule)["AAA-USDT"].iloc[-1]
    assert tradable_mask(_panel_with(1_000_000.0, 60), rule)["AAA-USDT"].iloc[-1]
