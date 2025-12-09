"""
src/utils/data_loader.py
市场数据处理器
修改点：在 prepare_data 返回前，根据 days_back 强制切片
"""

import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


class MarketDataHandler:
    """
    市场数据处理器：负责数据的获取、清洗、存储和加载
    遵循 ETL (Extract, Transform, Load) 原则
    """

    def __init__(self, exchange: ccxt.Exchange | None = None):
        """
        :param exchange: ccxt 交易所实例
        """
        self.exchange = exchange
        # 统一使用 data/cache 目录
        self.data_dir = os.path.join("data", "cache")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def prepare_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 365,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        智能数据准备：支持增量更新 (Incremental Update)
        并且只返回请求的时间窗口数据
        """
        safe_symbol = symbol.replace("/", "_")
        filename = f"{safe_symbol}_{timeframe}.parquet"

        # 1. 尝试加载本地
        df = self.load_from_parquet(filename)

        target_start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        now_utc = pd.Timestamp.now(tz="UTC")

        need_download = False
        is_append_mode = False  # 标记是否为追加模式
        download_start_dt = target_start_dt  # 默认下载起点

        # 2. 检查逻辑
        if df is None or df.empty or force_update:
            print(f"🌐 [数据] {symbol} 本地无数据或强制更新...")
            need_download = True
            is_append_mode = False
            download_start_dt = target_start_dt
        else:
            # 确保时区正确
            if df.index.tz is None:  # type: ignore
                df.index = df.index.tz_localize("UTC")  # type: ignore
            else:
                df.index = df.index.tz_convert("UTC")  # type: ignore

            local_start = df.index[0]
            local_end = df.index[-1]

            # 情况 A: 本地数据历史不够 (头部缺失) -> 全量重新下载
            # 允许 24h 的误差缓冲
            if local_start > target_start_dt + pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 历史数据不足 (始于 {local_start}), 需全量重新下载...")
                need_download = True
                is_append_mode = False
                download_start_dt = target_start_dt

            # 情况 B: 本地数据已过时 (尾部缺失) -> 增量下载
            elif local_end < now_utc - pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 数据已过时 (止于 {local_end}), 准备增量更新...")
                need_download = True
                is_append_mode = True
                download_start_dt = local_end  # 从本地最后一条开始下

        # 3. 下载流程 (如果需要)
        if need_download:
            # 格式化时间字符串
            since_str = download_start_dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
            print(f"⬇️ 正在获取: {symbol} 自 {since_str}...")

            # 下载新数据片段
            df_new = self.fetch_history_candles(symbol, timeframe, since_str)

            if is_append_mode and df is not None:
                # [优化] 增量合并: 旧数据 + 新数据
                print(f"🔗 [合并] 拼接旧数据({len(df)}) + 新数据({len(df_new)})")
                df_final = pd.concat([df, df_new])
            else:
                # 全量覆盖
                df_final = df_new

            # 统一清洗 (去重、排序、补全)
            df_final = self.clean_data(df_final, timeframe)

            # 保存完整缓存到硬盘
            self.save_to_parquet(df_final, filename)

            # 更新内存中的 df
            df = df_final

        if df is None or df.empty:
            return pd.DataFrame()

        # ==========================================
        # 4. [新增] 核心切片逻辑 (Slicing)
        # 无论缓存有多大，只返回用户请求的 days_back
        # ==========================================

        # 确保时区一致 (再次检查，防止刚下载的数据没有时区)
        if df.index.tz is None:  # type: ignore
            df.index = df.index.tz_localize("UTC")  # type: ignore
        else:
            df.index = df.index.tz_convert("UTC")  # type: ignore

        # 预留缓冲 (Buffer): 额外多取 5% 的天数，用于指标计算预热 (Warm-up)
        # 例如请求 100天，实际返回 105天，防止 MA30 在第1天是 NaN
        buffer_days = int(days_back * 0.05) + 1
        cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=days_back + buffer_days
        )

        # 执行切片
        df_sliced = df[df.index >= cutoff_dt].copy()

        # 如果切片后数据量变少了，打印提示
        if len(df_sliced) < len(df):
            print(
                f"✂️ [DataHandler] 缓存总量 {len(df)} -> 截取近 {days_back} 天 ({len(df_sliced)} 条)"
            )

        return df_sliced  # type: ignore

    def fetch_history_candles(
        self, symbol: str, timeframe: str, since_str: str, limit_per_req: int = 100
    ) -> pd.DataFrame:
        """
        循环分页拉取历史 K 线数据
        """
        if not self.exchange:
            raise ValueError("❌ 错误: 未初始化交易所实例，无法下载数据")

        # 转换开始时间为时间戳 (毫秒)
        since_dt = datetime.strptime(since_str, "%Y-%m-%d %H:%M:%S")
        since_dt = since_dt.replace(tzinfo=timezone.utc)
        since_ts = int(since_dt.timestamp() * 1000)

        all_candles = []
        current_since = since_ts
        now_ts = int(time.time() * 1000)

        max_retries = 5
        retries = 0

        while current_since < now_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=limit_per_req
                )
                retries = 0

                if not ohlcv:
                    break

                last_ts = ohlcv[-1][0]
                if last_ts <= current_since:
                    break

                all_candles.extend(ohlcv)
                # 减少打印频率，每 1000 条打印一次
                if len(all_candles) % 1000 == 0:
                    print(f"   已获取 {len(all_candles)} 条...")

                current_since = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                retries += 1
                print(f"❌ [下载异常] {e}, 重试 ({retries}/{max_retries})...")
                if retries > max_retries:
                    print("❌ 达到最大重试次数，停止下载")
                    break
                time.sleep(3)
                continue

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)

        print(f"✅ [下载] 本次拉取完成! 共 {len(df)} 条")
        return df

    def clean_data(self, df: pd.DataFrame, timeframe_str: str = "1h") -> pd.DataFrame:
        """
        标准数据清洗流程
        """
        if df.empty:
            return df

        # [优化] 显式拷贝，防止 SettingWithCopyWarning
        df = df.copy()

        # 1. 去重 (保留最后一次出现的)
        df = df[~df.index.duplicated(keep="last")]  # type: ignore

        # 2. 排序
        df = df.sort_index()

        # 3. 频率转换映射
        pd_freq = self._convert_to_pandas_freq(timeframe_str)

        # 4. 处理时间缺失 (Resample & Reindex)
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=pd_freq, tz="UTC"
        )

        if len(full_idx) != len(df):
            # print(f"⚠️ [清洗] 补全时间断点: {len(full_idx) - len(df)} 行")
            df = df.reindex(full_idx)

            # 填充逻辑
            df["close"] = df["close"].ffill()
            df["open"] = df["open"].fillna(df["close"])
            df["high"] = df["high"].fillna(df["close"])
            df["low"] = df["low"].fillna(df["close"])
            df["volume"] = df["volume"].fillna(0)
            df["timestamp"] = df.index.astype("int64") // 10**6

        # 5. 逻辑错误修复 (High < Low)
        mask_error = df["high"] < df["low"]
        if mask_error.any():
            # print(f"⚠️ [清洗] 修复 {mask_error.sum()} 行 High < Low 异常")
            temp = df.loc[mask_error, "high"].copy()
            df.loc[mask_error, "high"] = df.loc[mask_error, "low"]
            df.loc[mask_error, "low"] = temp

        df = df.dropna()
        return df

    def _convert_to_pandas_freq(self, timeframe: str) -> str:
        """
        将 CCXT/交易所的周期字符串转换为 Pandas 兼容的 freq 字符串
        """
        mapping = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1D",
            "1w": "1W",
            "1M": "1ME",
        }
        return mapping.get(timeframe, timeframe)

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.data_dir, filename)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype("int64")
        try:
            df.to_parquet(path)
            # print(f"💾 [存储] {path}")
        except Exception as e:
            print(f"❌ [存储失败] {e}")

    def load_from_parquet(self, filename: str) -> pd.DataFrame | None:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            print(f"❌ [读取失败] {e}")
            return None
