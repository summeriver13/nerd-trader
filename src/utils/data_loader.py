"""
src/utils/data_loader.py
核心数据调度器：负责缓存管理、预热缓冲计算及多源回退调度。
代码位置：src/utils/data_loader.py
遵循规范：Ruff + Python 3.12 + Pyright Strict
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Final, TypeAlias, cast

import ccxt
import pandas as pd

# 内部模块导入
from .data_sources.base import BaseDataSource
from .data_sources.crypto import CCXTSource
from .data_sources.stock_cn import AkShareSource, BaostockSource, TushareSource

# 配置日志
logger = logging.getLogger(__name__)

# 类型别名
SymbolStr: TypeAlias = str
TimeFrameStr: TypeAlias = str

# 常量定义
A_SHARE_BUFFER_FACTOR: Final[float] = 1.5
UTC_TZ: Final[str] = "UTC"


class MarketDataHandler:
    """
    统一数据调度门面类。
    职责：
    1. 动态计算指标计算所需的历史预热数据量。
    2. 管理本地 Parquet 缓存磁盘持久化。
    3. 在多个数据源之间执行回退逻辑（Fallback）。
    """

    def __init__(self, exchange: ccxt.Exchange | None = None) -> None:
        self.exchange = exchange
        self.data_root: Final[Path] = Path("data") / "storage"

        # 根据市场类型初始化不同的调度策略
        if self.exchange:
            self.sources: list[BaseDataSource] = [CCXTSource(self.exchange)]
            self.market_label: Final[str] = "crypto"
            self.source_label: Final[str] = str(
                getattr(self.exchange, "id", "unknown_cex")
            )
        else:
            # A股：Baostock优先(提供估值)，Tushare稳定，AkShare兜底
            self.sources: list[BaseDataSource] = [
                BaostockSource(),
                TushareSource(),
                AkShareSource(),
            ]
            self.market_label: Final[str] = "stock_cn"
            self.source_label: Final[str] = "combined"

    def _get_file_path(self, symbol: SymbolStr, timeframe: TimeFrameStr) -> Path:
        """生成标准化的本地持久化路径。"""
        safe_symbol = symbol.replace("/", "-").upper()
        return (
            self.data_root
            / self.market_label
            / self.source_label
            / timeframe
            / f"{safe_symbol}.parquet"
        )

    def prepare_data(
        self,
        symbol: SymbolStr,
        timeframe: TimeFrameStr = "1d",
        days_back: int = 365,
        required_buffer: int = 200,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        准备数据核心入口：自动处理缓存检查、历史补齐与增量更新。
        """
        file_path = self._get_file_path(symbol, timeframe)
        df_local = self._load_local(file_path)

        # 1. 统筹当前时间与目标范围
        now_utc = pd.Timestamp.now(tz=UTC_TZ)
        is_crypto = self.exchange is not None

        # 针对 A 股非交易日进行冗余计算
        actual_buffer = (
            required_buffer
            if is_crypto
            else int(required_buffer * A_SHARE_BUFFER_FACTOR)
        )
        target_start_dt = now_utc - pd.Timedelta(days=days_back + actual_buffer)

        need_fetch = False
        fetch_start = cast(datetime, target_start_dt)

        # 2. 数据水位分析
        if df_local is None or df_local.empty or force_update:
            need_fetch = True
        else:
            # 情况 A: 本地数据起始时间太晚，无法覆盖预热期
            if df_local.index[0] > target_start_dt:
                logger.info(
                    f"[{symbol}] 本地历史深度不足 ({df_local.index[0].date()}), 执行全量同步"
                )
                need_fetch = True

            # 情况 B: 本地数据最新时间太早，需要增量更新
            elif df_local.index[-1] < now_utc - pd.Timedelta(hours=24):
                logger.info(f"[{symbol}] 数据已过时, 准备增量更新...")
                need_fetch = True
                # 增量更新从本地最后一条数据开始，避免空洞
                fetch_start = cast(datetime, df_local.index[-1])

        # 3. 抓取与合并流程
        if need_fetch:
            df_new = self._fetch_from_sources(
                symbol, timeframe, fetch_start, cast(datetime, now_utc)
            )

            if not df_new.empty:
                # 只有在非强制刷新且本地有旧数据时才尝试合并
                if df_local is not None and not force_update and not df_local.empty:
                    # 检查是否有重叠，确保 concat 的逻辑正确
                    df_final = pd.concat([df_local, df_new])
                else:
                    df_final = df_new

                # 统一执行清洗与持久化
                df_local = self._clean_and_save(df_final, file_path)
            else:
                logger.warning(f"[{symbol}] 无法从在线数据源获取有效结果")

        if df_local is None or df_local.empty:
            return pd.DataFrame()

        # 返回完整带 Buffer 的数据，由策略层决定截断
        return df_local

    def _fetch_from_sources(
        self, symbol: SymbolStr, timeframe: TimeFrameStr, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """顺序遍历数据源链，实现 Fallback。"""
        for source in self.sources:
            try:
                # 逻辑解耦：在 Source 内部处理频率兼容性，此处仅执行调度
                if isinstance(source, TushareSource) and timeframe != "1d":
                    continue

                df = source.fetch_bars(symbol, timeframe, start, end)
                if not df.empty:
                    return df
            except Exception as e:
                logger.error(f"数据源 {source.__class__.__name__} 请求异常: {e}")
                continue
        return pd.DataFrame()

    def _load_local(self, path: Path) -> pd.DataFrame | None:
        """从本地磁盘加载 Parquet 缓存。"""
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                # 确保索引为时区感知的 DatetimeIndex
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(UTC_TZ)
                else:
                    df.index = df.index.tz_convert(UTC_TZ)
            return df
        except Exception as e:
            logger.error(f"本地缓存读取失败 {path.name}: {e}")
            return None

    def _clean_and_save(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """数据清洗标准化并存储。"""
        # 复制以避免影响原始引用
        cleaned_df = df.copy()

        # 1. 索引去重：以最新的数据覆盖旧数据 (keep='last')
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="last")]

        # 2. 时间升序排列
        cleaned_df = cleaned_df.sort_index()

        # 3. 存储
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_parquet(path, compression="snappy")
            logger.info(f"缓存已更新: {path.name} (当前行数: {len(cleaned_df)})")
        except Exception as e:
            logger.error(f"持久化至本地失败 {path.name}: {e}")

        return cleaned_df
