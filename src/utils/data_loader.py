"""
src/utils/data_loader.py
核心数据调度器：缓存管理、预热缓冲计算与数据源抓取。

职责分工：本模块只负责「拿到一段可用历史」，窗口裁剪由调用方（runners.load_window）决定。
已知陷阱：合并只覆盖 incoming 出现的键，删不掉旧坏行——坏行修正只能删 parquet 重建。
已知陷阱：数据源会给出数值上不可能的 K 线（详见 bar_quality），落盘前丢弃 + 告警。
旁挂 .meta.json 记「已问过的最早起点」：年轻标的的请求起点永远早于上市起点，
不记就会每次运行全量重下（源侧给不出更早历史）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal, cast

import ccxt
import pandas as pd

from src.config.paths import STORAGE_ROOT

from .bar_quality import (
    SPIKE_RATIO,
    impossible_bars,
    sample_times,
    suspicious_spikes,
)
from .data_sources.base import BaseDataSource, DataSourceError

logger = logging.getLogger(__name__)

TimeFrameStr = Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

UTC_TZ: Final[str] = "UTC"
STALE_HOURS: Final[int] = 24  # 本地数据滞后超过此值就尝试增量更新
WEEK_SECONDS: Final[int] = 604_800
# 周线网格锚点在 epoch（1970-01-01 是周四），先平移 3 天再对齐等价于锚在周一
WEEK_GRID_SHIFT: Final[timedelta] = timedelta(days=3)


def _align_to_grid(stamp: pd.Timestamp, bar_seconds: int) -> pd.Timestamp:
    """对齐到 K 线网格：数据源只返回网格点上的 Bar。

    不对齐会把「同日稍晚」误判成历史深度不足，导致每次运行都全量重下。
    """
    shift = WEEK_GRID_SHIFT if bar_seconds == WEEK_SECONDS else timedelta(0)
    return cast(pd.Timestamp, (stamp + shift).ceil(f"{bar_seconds}s") - shift)


class MarketDataHandler:
    """
    统一数据调度门面类。

    只依赖数据源本身（不需要完整交易所客户端）：时间换算走 ccxt.Exchange.parse_timeframe
    这个静态方法，会话由调用方持有并负责关闭。
    """

    def __init__(self, source: BaseDataSource, data_root: Path | None = None) -> None:
        self.source = source
        self.data_root: Final[Path] = (
            data_root if data_root is not None else STORAGE_ROOT
        )

    def _get_file_path(self, symbol: str, timeframe: TimeFrameStr) -> Path:
        """生成标准化的本地持久化路径。"""
        safe_symbol = symbol.replace("/", "-").upper()
        return (
            self.data_root
            / "crypto"
            / self.source.label
            / timeframe
            / f"{safe_symbol}.parquet"
        )

    def prepare_data(
        self,
        symbol: str,
        timeframe: TimeFrameStr = "1d",
        days_back: int = 365,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """准备数据核心入口：缓存检查、历史补齐、增量更新与未收盘 K 线过滤。"""
        file_path = self._get_file_path(symbol, timeframe)
        df_local = self._load_local(file_path)
        now_utc = pd.Timestamp.now(tz=UTC_TZ)
        bar_seconds = int(ccxt.Exchange.parse_timeframe(timeframe))
        target_start_dt = _align_to_grid(
            now_utc - timedelta(days=days_back), bar_seconds
        )

        need_fetch = False
        fetch_start = cast(datetime, target_start_dt)
        if df_local is None or df_local.empty or force_update:
            need_fetch = True
        else:
            local_index = pd.DatetimeIndex(df_local.index)
            # 情况 A: 本地起点太晚，无法覆盖请求窗口；已问过更早起点就不再重下
            if local_index[0] > target_start_dt:
                if self._probed_earlier(file_path, target_start_dt):
                    logger.info(
                        f"[{symbol}] 本地起点 {local_index[0]:%Y-%m-%d} 晚于请求起点 "
                        f"{target_start_dt:%Y-%m-%d}，但已问过更早起点，跳过全量同步"
                    )
                else:
                    logger.info(
                        f"[{symbol}] 本地历史深度不足 ({local_index[0]:%Y-%m-%d}), 执行全量同步"
                    )
                    need_fetch = True
            # 情况 B: 本地最新时间太早，需要增量更新（从最后一根续抓，避免空洞）
            if not need_fetch and local_index[-1] < now_utc - timedelta(
                hours=STALE_HOURS
            ):
                logger.info(f"[{symbol}] 数据已过时, 准备增量更新...")
                need_fetch = True
                fetch_start = cast(datetime, local_index[-1])

        if need_fetch:
            # 取数失败会抛 DataSourceError，不再静默回退到过期缓存
            df_new = self._fetch(
                symbol, timeframe, fetch_start, cast(datetime, now_utc)
            )
            if fetch_start == target_start_dt:
                # 记下问过的起点：源侧给不出更早历史时，下次不再全量重下
                self._write_asked_start(file_path, target_start_dt)
            if not df_new.empty:
                # 永远合并、不替换：force_update 只该重问数据源，不该丢掉本地已有的历史
                # （曾因此把 8 年缓存截断成 20 根）。重复时间戳保留新值，见 _clean_and_save。
                merged = df_new if df_local is None else pd.concat([df_local, df_new])
                df_local = self._clean_and_save(merged, file_path)
            elif df_local is not None and not df_local.empty:
                logger.info(f"[{symbol}] 数据源本次未返回新数据")
            else:
                logger.warning(f"[{symbol}] 数据源返回空：该标的历史为空或起点越界")

        if df_local is None or df_local.empty:
            return pd.DataFrame()

        # 抓取后核对覆盖区间：短帧（不是空帧）同样会「静默过期」
        self._check_coverage(symbol, df_local, target_start_dt, now_utc)

        # 未收盘的最后一根不进回测：指标与信号会建立在还在变的 Bar 上，
        # 同一条命令在盘中与收盘后跑会给出不同结果（落盘仍保留，交给下次刷新覆盖）
        bar_delta = timedelta(seconds=bar_seconds)
        closed = df_local[df_local.index + bar_delta <= now_utc]
        return cast(pd.DataFrame, closed)

    def _fetch(
        self, symbol: str, timeframe: TimeFrameStr, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """单源抓取：数据源抛异常即上抛（「取数失败」与「没有数据」是两件事）。"""
        try:
            return self.source.fetch_bars(symbol, timeframe, start, end)
        except Exception as e:
            raise DataSourceError(
                f"{self.source.label} 抓取 {symbol} {timeframe} 失败: {e}"
            ) from e

    def _meta_path(self, file_path: Path) -> Path:
        return file_path.with_suffix(".meta.json")

    def _read_asked_start(self, file_path: Path) -> pd.Timestamp | None:
        """读「曾经问到过的最早起点」备忘（与 funding.py 的缓存旁挂 meta 同一套）。"""
        path = self._meta_path(file_path)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("asked_start")
        except (OSError, ValueError):
            return None
        return cast(pd.Timestamp, pd.Timestamp(value)) if value else None

    def _write_asked_start(self, file_path: Path, asked_start: pd.Timestamp) -> None:
        path = self._meta_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"asked_start": asked_start.isoformat()}), encoding="utf-8"
        )

    def _probed_earlier(self, file_path: Path, target_start_dt: pd.Timestamp) -> bool:
        """是否已经问过不晚于本次请求的起点（问过就不必再全量重下）。"""
        asked_start = self._read_asked_start(file_path)
        return asked_start is not None and asked_start <= target_start_dt

    def _check_coverage(
        self,
        symbol: str,
        df_local: pd.DataFrame,
        target_start_dt: pd.Timestamp,
        now_utc: pd.Timestamp,
    ) -> None:
        """抓取后核对覆盖区间：短帧不会经过空帧分支，必须在这里判定。"""
        index = pd.DatetimeIndex(df_local.index)
        if index[-1] < now_utc - timedelta(hours=STALE_HOURS):
            self._warn_stale(symbol, df_local, now_utc)
        if index[0] > target_start_dt:
            logger.warning(
                f"[{symbol}] 覆盖仍不足：源侧起点 {index[0]:%Y-%m-%d} 晚于请求起点 "
                f"{target_start_dt:%Y-%m-%d}（上市时间或源侧上限），窗口前段指标未热透"
            )

    def _warn_stale(
        self, symbol: str, df_local: pd.DataFrame, now_utc: pd.Timestamp
    ) -> None:
        """缓存落后于当前时间时明确报出截止日，别让人以为是最新的。"""
        last = cast(pd.Timestamp, df_local.index[-1])
        lag_days = (now_utc - last).total_seconds() / 86400
        logger.warning(
            f"[{symbol}] 本地缓存仍滞后：截止 {last.date()}（滞后 {lag_days:.1f} 天）"
        )

    def _load_local(self, path: Path) -> pd.DataFrame | None:
        """从本地磁盘加载 Parquet 缓存。"""
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                index = pd.to_datetime(df.index)
                if index.tz is None:
                    index = index.tz_localize(UTC_TZ)
                else:
                    index = index.tz_convert(UTC_TZ)
                df.index = index
            return df
        except Exception as e:
            logger.error(f"本地缓存读取失败 {path.name}: {e}")
            return None

    def _clean_and_save(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """清洗（去重、排序、值域校验）并落盘。

        值域校验放在采集侧：QMT 那类「成交额正常但 OHLC 全为 0」的脏行不该入库后再清。
        """
        cleaned_df = df.copy()
        # 用 .loc 而非 [] 索引：pandas-stubs 把 ndarray 布尔掩码的 [] 推断成 DataFrame | ndarray
        cleaned_df = cleaned_df.loc[~cleaned_df.index.duplicated(keep="last")]
        cleaned_df = cleaned_df.sort_index()

        bad = impossible_bars(cleaned_df)
        if bad.any():
            logger.warning(
                f"{path.name} 丢弃 {int(bad.sum())} 根非法 K 线（非正价 / 高低倒挂 / "
                f"包络违反 / 负成交量 / 缺失值）: {sample_times(cleaned_df.index[bad.to_numpy()])}"
            )
            cleaned_df = cleaned_df.loc[~bad]

        spikes = suspicious_spikes(cleaned_df)
        if spikes.any():
            logger.warning(
                f"{path.name} ⚠️ {int(spikes.sum())} 根 K 线偏离邻域中位数 "
                f"{SPIKE_RATIO:.0f} 倍以上（数据源污染或真实插针，已保留；"
                f"跑 scripts/crosscheck_crypto_bars.py 对账确认）: "
                f"{sample_times(cleaned_df.index[spikes.to_numpy()])}"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_parquet(path, compression="snappy")
            logger.info(f"缓存已更新: {path.name}（{len(cleaned_df)} 行）")
        except Exception as e:
            logger.error(f"持久化至本地失败 {path.name}: {e}")

        return cast(pd.DataFrame, cleaned_df)
