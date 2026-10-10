"""数据层不变量：未收盘 K 线不进回测、脏数据不进缓存、取数失败不静默。

历史教训（修复前）：取数失败会静默回退到几周前的缓存、最后一根未收盘的 Bar
会被当成完整数据参与指标，两者都让「同一条命令给出不同结果」。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import cast, override

import pandas as pd
import pytest

from src.utils.data_loader import MarketDataHandler, _align_to_grid
from src.utils.data_sources.base import BaseDataSource, DataSourceError


class _FakeSource(BaseDataSource):
    """假数据源：可指定返回的帧，或强制抛异常。"""

    @property
    @override
    def label(self) -> str:
        return "fake"

    def __init__(self, frame: pd.DataFrame | None = None, error: bool = False) -> None:
        self.frame = frame if frame is not None else pd.DataFrame()
        self.error = error
        self.calls: list[tuple[datetime, datetime]] = []

    @override
    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        self.calls.append((start_dt, end_dt))
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


def _recent_frame(periods: int, freq: str = "1D") -> pd.DataFrame:
    """以昨天结尾的帧：测试不能随日历过期，也天然满足「滞后 > 24h」。"""
    end = cast(pd.Timestamp, pd.Timestamp.now(tz="UTC").floor("1D")) - pd.Timedelta(
        1, unit="D"
    )
    index = pd.date_range(end=end, periods=periods, freq=freq)
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


def test_ohlc_envelope_violation_is_dropped(tmp_path: Path) -> None:
    """包络违反（收盘高于最高价）只可能是数据源混了字段。

    实测：ccxt 的 OKX 路径把别处的值塞进 high/close，给过 H=3616 万、C=13.3 万。
    """
    frame = _frame(3)
    frame.loc[frame.index[1], ["open", "high", "low", "close"]] = [
        30500.0,
        31000.0,
        29400.0,
        133653.0,  # 收盘高于最高价
    ]
    handler = MarketDataHandler(_FakeSource(frame), data_root=tmp_path)
    out = handler.prepare_data("TEST/USDT", "1d", days_back=10, force_update=True)
    assert len(out) == 2


def test_suspicious_spike_warns_but_is_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """孤立尖峰只告警不丢弃：真实插针与数据源污染本地分不清，删真数据比留脏数据更糟。"""
    frame = _frame(13)
    frame.loc[frame.index[6], "high"] = 1_000_000.0  # 邻域中位数 1.0 → 百万倍
    handler = MarketDataHandler(_FakeSource(frame), data_root=tmp_path)

    with caplog.at_level(logging.WARNING):
        out = handler.prepare_data("TEST/USDT", "1d", days_back=20, force_update=True)

    assert len(out) == 13
    messages = [record.getMessage() for record in caplog.records]
    assert any("偏离邻域中位数" in message for message in messages)


def test_fetch_failure_raises_instead_of_returning_stale_cache(tmp_path: Path) -> None:
    """取数失败必须上抛：静默回退到旧缓存会让人拿过期数据做决策。"""
    handler = MarketDataHandler(_FakeSource(error=True), data_root=tmp_path)
    with pytest.raises(DataSourceError):
        handler.prepare_data("TEST/USDT", "1d", days_back=10, force_update=True)


def test_young_symbol_does_not_resync_full_history_every_run(tmp_path: Path) -> None:
    """年轻标的的请求起点永远早于上市起点：第二次运行只能做增量，不能全量重下。"""
    frame = _recent_frame(5)  # 只有 5 天历史，晚于 days_back=30 的请求起点
    source = _FakeSource(frame)
    handler = MarketDataHandler(source, data_root=tmp_path)

    handler.prepare_data("TEST/USDT", "1d", days_back=30)
    first_start = source.calls[0][0]
    source.calls.clear()
    handler.prepare_data("TEST/USDT", "1d", days_back=30)

    assert len(source.calls) == 1
    assert source.calls[0][0] == cast(pd.Timestamp, frame.index[-1]).to_pydatetime()
    assert source.calls[0][0] > first_start


def test_force_update_keeps_local_history(tmp_path: Path) -> None:
    """force_update 只该重问数据源，不能丢掉本地已有的历史。

    历史教训：force_update 走「替换」分支，把 8 年缓存（3194 根）截断成请求窗口的 20 根。
    """
    MarketDataHandler(_FakeSource(_recent_frame(60)), data_root=tmp_path).prepare_data(
        "TEST/USDT", "1d", days_back=60
    )
    # 第二次强制更新：源侧这次只返回 3 根，本地 60 根不能被替换掉
    MarketDataHandler(_FakeSource(_recent_frame(3)), data_root=tmp_path).prepare_data(
        "TEST/USDT", "1d", days_back=3, force_update=True
    )

    cached = pd.read_parquet(next(tmp_path.rglob("*.parquet")))
    assert len(cached) == 60


def test_short_frame_still_reports_staleness(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """源侧返回「更短但非空」的帧也要报滞后：空帧分支覆盖不到这种情况。"""
    source = _FakeSource(_recent_frame(3))
    handler = MarketDataHandler(source, data_root=tmp_path)

    with caplog.at_level(logging.WARNING):
        handler.prepare_data("TEST/USDT", "1d", days_back=30)

    messages = [record.getMessage() for record in caplog.records]
    assert any("仍滞后" in message for message in messages)
    assert any("覆盖仍不足" in message for message in messages)
