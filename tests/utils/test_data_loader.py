"""数据层不变量：未收盘 K 线不进回测、脏数据不进缓存、取数失败不静默。

历史教训（修复前）：取数失败会静默回退到几周前的缓存、最后一根未收盘的 Bar
会被当成完整数据参与指标，两者都让「同一条命令给出不同结果」。
"""

from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.utils.data_loader import MarketDataHandler, _align_to_grid
from src.utils.data_sources.base import BaseDataSource, DataSourceError


class _FakeSource(BaseDataSource):
    """假数据源：可指定返回的帧，或强制抛异常。"""

    @property
    def label(self) -> str:
        return "fake"

    def __init__(self, frame: pd.DataFrame | None = None, error: bool = False) -> None:
        self.frame = frame if frame is not None else pd.DataFrame()
        self.error = error

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if self.error:
            raise RuntimeError("proxy down")
        return self.frame


def _ts(text: str) -> pd.Timestamp:
    """pd.Timestamp(...) 静态类型是 Timestamp | NaTType，测试里断言合法输入"""
    return cast(pd.Timestamp, pd.Timestamp(text, tz="UTC"))


def _frame(periods: int, freq: str = "1D") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index
    )


def test_align_to_grid_uses_monday_for_weekly() -> None:
    """周线网格锚点是周一（epoch 是周四，直接 ceil 会偏 3 天）。"""
    stamp = _ts("2026-10-09 09:00")  # 周五
    aligned = _align_to_grid(stamp, 604_800)
    assert aligned.weekday() == 0
    assert aligned >= stamp

    daily = _align_to_grid(_ts("2026-10-09 09:00"), 86_400)
    assert daily == pd.Timestamp("2026-10-10 00:00", tz="UTC")


def test_unclosed_last_bar_is_filtered(tmp_path: Path) -> None:
    """最后一根还没走完的 Bar 不进回测：否则盘中与收盘后跑出两个答案。"""
    source = _FakeSource(_frame(5))
    handler = MarketDataHandler(source, data_root=tmp_path)
    out = handler.prepare_data("TEST/USDT", "1d", days_back=10, force_update=True)
    assert out.index[-1] < pd.Timestamp.now(tz="UTC").ceil("1D")


def test_value_validation_drops_dirty_rows(tmp_path: Path) -> None:
    """非正价 / 高低倒挂 / 负成交量的行不进缓存（容器 SOP：在采集侧过滤）。"""
    frame = _frame(3)
    frame.loc[frame.index[1], "close"] = 0.0
    frame.loc[frame.index[2], "high"] = 0.5  # low=1.0 → 倒挂
    handler = MarketDataHandler(_FakeSource(frame), data_root=tmp_path)
    out = handler.prepare_data("TEST/USDT", "1d", days_back=10, force_update=True)
    assert len(out) == 1


def test_fetch_failure_raises_instead_of_returning_stale_cache(tmp_path: Path) -> None:
    """取数失败必须上抛：静默回退到旧缓存会让人拿过期数据做决策。"""
    handler = MarketDataHandler(_FakeSource(error=True), data_root=tmp_path)
    with pytest.raises(DataSourceError):
        handler.prepare_data("TEST/USDT", "1d", days_back=10, force_update=True)
