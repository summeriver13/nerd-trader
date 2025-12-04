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
        # [修改点] 统一使用 data/cache 目录
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
        智能数据准备：优先读取本地，本地数据不足或过旧时自动更新
        """
        safe_symbol = symbol.replace("/", "_")
        filename = f"{safe_symbol}_{timeframe}.parquet"

        # 1. 尝试加载本地
        df = self.load_from_parquet(filename)

        target_start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        need_download = False

        # 下载的起始时间默认为目标时间
        download_since_str = target_start_dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore

        # 2. 检查逻辑
        if df is None or df.empty or force_update:
            print("🌐 [数据] 本地无数据或强制更新，准备下载...")
            need_download = True
        else:
            # 确保时区正确
            if df.index.tz is None:  # type: ignore
                df.index = df.index.tz_localize("UTC")  # type: ignore
            else:
                df.index = df.index.tz_convert("UTC")  # type: ignore

            local_start = df.index[0]
            local_end = df.index[-1]
            now_utc = pd.Timestamp.now(tz="UTC")

            # 检查长度: 允许 24h 的误差缓冲
            if local_start > target_start_dt + pd.Timedelta(hours=24):
                print(
                    f"⚠️ [数据] 本地数据长度不足 (始于 {local_start})，准备重新下载..."
                )
                need_download = True
                # 这里目前采用简单策略：覆盖下载。
                # 进阶优化：可以只下载缺失的前半段然后 concat，但要注意重叠处理。

            # 检查时效: 如果数据滞后超过 24小时
            if local_end < now_utc - pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 本地数据已过时 (止于 {local_end})，准备更新...")
                need_download = True
                # 进阶优化：如果只是缺后半段，可以将 download_since_str 设置为 local_end

        # 3. 下载流程
        if need_download:
            print(f"⬇️ 正在获取数据: {symbol} 自 {download_since_str}...")

            # [优化] 下载新数据
            df_new = self.fetch_history_candles(symbol, timeframe, download_since_str)

            # [优化] 清洗数据 (传入 timeframe 以便 reindex)
            df_new = self.clean_data(df_new, timeframe)

            # 保存
            self.save_to_parquet(df_new, filename)
            return df_new

        if df is None:
            return pd.DataFrame()

        return df

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

        # [优化] 重试计数器
        max_retries = 5
        retries = 0

        print(f"⬇️ [下载] 开始获取 {symbol} {timeframe} 数据...")

        while current_since < now_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=limit_per_req
                )

                # 成功获取，重置重试次数
                retries = 0

                if not ohlcv:
                    print("⚠️ [下载] 未获取到更多数据，停止下载")
                    break

                last_ts = ohlcv[-1][0]

                # 防止死循环：如果交易所返回的数据没有推进时间
                if last_ts <= current_since:
                    break

                all_candles.extend(ohlcv)
                print(
                    f"   已获取 {len(ohlcv)} 条, 最新: {pd.to_datetime(last_ts, unit='ms', utc=True)}"
                )

                # 更新下一次请求的开始时间
                current_since = last_ts + 1

                # 遵守速率限制
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

        # 转换为 DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)

        print(f"✅ [下载] 完成! 共 {len(df)} 条")
        return df

    def clean_data(self, df: pd.DataFrame, timeframe_str: str = "1h") -> pd.DataFrame:
        """
        标准数据清洗流程
        """
        if df.empty:
            return df

        print("🧹 [清洗] 开始数据清洗...")
        original_count = len(df)

        # 1. 去重
        df = df[~df.index.duplicated(keep="last")]  # type: ignore

        # 2. 排序
        df = df.sort_index()

        # 3. [优化] 频率转换映射
        pd_freq = self._convert_to_pandas_freq(timeframe_str)

        # 4. 处理时间缺失 (Resample & Reindex)
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=pd_freq, tz="UTC"
        )

        if len(full_idx) != len(df):
            print(f"⚠️ [清洗] 补全时间断点: 理论 {len(full_idx)} vs 实际 {len(df)}")
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
            print(f"⚠️ [清洗] 修复 {mask_error.sum()} 行 High < Low 异常")
            temp = df.loc[mask_error, "high"].copy()
            df.loc[mask_error, "high"] = df.loc[mask_error, "low"]
            df.loc[mask_error, "low"] = temp

        df = df.dropna()
        print(f"✅ [清洗] 完成. {original_count} -> {len(df)}")
        return df

    def _convert_to_pandas_freq(self, timeframe: str) -> str:
        """
        [新增] 将 CCXT/交易所的周期字符串转换为 Pandas 兼容的 freq 字符串
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
            "1M": "1ME",  # Pandas 2.2+ 推荐 1ME (Month End)
        }
        # 如果找不到映射，默认返回原值（比如 1h 通常通用）
        return mapping.get(timeframe, timeframe)

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.data_dir, filename)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype("int64")
        try:
            df.to_parquet(path)
            print(f"💾 [存储] {path}")
        except Exception as e:
            print(f"❌ [存储失败] {e}")

    def load_from_parquet(self, filename: str) -> pd.DataFrame | None:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            # print(f"📂 [读取] 本地数据: {len(df)} 行")
            return df
        except Exception as e:
            print(f"❌ [读取失败] {e}")
            return None
