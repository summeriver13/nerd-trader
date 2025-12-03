import os
import time
from datetime import datetime, timezone

import ccxt

# 移除旧式类型导入
# from typing import Optional
import pandas as pd


class MarketDataHandler:
    """
    市场数据处理器：负责数据的获取、清洗、存储和加载
    遵循 ETL (Extract, Transform, Load) 原则
    """

    # 使用 Python 3.10+ 的联合类型写法 (Type | None)
    def __init__(self, exchange: ccxt.Exchange | None = None):
        """
        :param exchange: ccxt 交易所实例 (如果是读取本地数据不需要传)
        """
        self.exchange = exchange
        # 创建数据存储目录
        # [修改点] 将默认存储路径改为 data/cache，符合工程结构
        self.data_dir = os.path.join("data", "cache")
        if not os.path.exists(self.data_dir):
            # makedirs 会自动创建多级目录 (data 和 cache)
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
        :param symbol: 交易对
        :param timeframe: 周期
        :param days_back: 需要回测的天数
        :param force_update: 是否强制重新下载
        """
        safe_symbol = symbol.replace("/", "_")
        filename = f"{safe_symbol}_{timeframe}.parquet"

        # 1. 尝试加载本地
        df = self.load_from_parquet(filename)

        target_start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        need_download = False

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

            # 检查长度: 留 24h buffer
            if local_start > target_start_dt + pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 本地数据长度不足 (始于 {local_start})，需下载补充...")
                need_download = True

            # 检查时效: 如果数据滞后超过 24小时
            if local_end < now_utc - pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 本地数据已过时 (止于 {local_end})，需增量更新...")
                need_download = True

        # 3. 下载流程
        if need_download:
            # type: ignore 忽略 NaTType 误报
            since_str = target_start_dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
            print(f"⬇️ 正在获取数据: {symbol} 自 {since_str}...")

            # 这里简化处理：直接覆盖下载。
            df_new = self.fetch_history_candles(symbol, timeframe, since_str)
            df_new = self.clean_data(df_new, timeframe)
            self.save_to_parquet(df_new, filename)
            return df_new

        # 逻辑上如果 df 为 None，上面的 need_download 会为 True 并返回 df_new
        # 但类型检查器无法推断这一点，所以这里显式处理 None 的情况
        if df is None:
            return pd.DataFrame()

        return df

    def fetch_history_candles(
        self, symbol: str, timeframe: str, since_str: str, limit_per_req: int = 100
    ) -> pd.DataFrame:
        """
        循环分页拉取历史 K 线数据
        :param symbol: 交易对 (e.g., 'BTC/USDT')
        :param timeframe: 周期 (e.g., '1h', '1m')
        :param since_str: 开始时间 (e.g., '2023-01-01 00:00:00')
        :return: 原始 DataFrame
        """
        if not self.exchange:
            raise ValueError("❌ 错误: 未初始化交易所实例，无法下载数据")

        # 转换开始时间为时间戳 (毫秒)
        since_dt = datetime.strptime(since_str, "%Y-%m-%d %H:%M:%S")
        # 强制设为 UTC
        since_dt = since_dt.replace(tzinfo=timezone.utc)
        since_ts = int(since_dt.timestamp() * 1000)

        all_candles = []
        current_since = since_ts
        now_ts = int(time.time() * 1000)

        print(f"⬇️ [下载] 开始获取 {symbol} {timeframe} 数据，自 {since_str} (UTC)...")

        while current_since < now_ts:
            try:
                # OKX 的 fetch_ohlcv 支持 since 参数
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=limit_per_req
                )

                if not ohlcv:
                    print("⚠️ [下载] 未获取到更多数据，停止下载")
                    break

                # 获取最后一条数据的时间戳
                last_ts = ohlcv[-1][0]

                # 如果获取的数据最后一条都比请求的开始时间早（防死循环），则退出
                if last_ts <= current_since:
                    break

                all_candles.extend(ohlcv)
                print(
                    f"   已获取 {len(ohlcv)} 条, 最新时间: {pd.to_datetime(last_ts, unit='ms', utc=True)}"
                )

                # 更新下一次请求的开始时间 = 最后一条时间 + 1个周期 (大约)
                # 为了准确，直接用最后一条的时间 + 1ms 即可，ccxt 会处理重叠
                current_since = last_ts + 1

                # 遵守速率限制，稍微睡一下
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"❌ [下载异常] {e}")
                print("   尝试重试 (3s后)...")
                time.sleep(3)
                continue

        if not all_candles:
            return pd.DataFrame()

        # 转换为 DataFrame
        # 添加 # type: ignore 忽略 basedpyright 对 pandas 构造函数的误报
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)

        print(f"✅ [下载] 完成! 共 {len(df)} 条原始数据")
        return df

    def clean_data(self, df: pd.DataFrame, timeframe_str: str = "1h") -> pd.DataFrame:
        """
        标准数据清洗流程
        """
        if df.empty:
            return df

        print("🧹 [清洗] 开始数据清洗...")
        original_count = len(df)

        # 1. 去重 (保留最后一次出现的)
        df = df[~df.index.duplicated(keep="last")]  # type: ignore

        # 2. 排序
        df = df.sort_index()

        # 3. 处理时间缺失 (Resample & Reindex)
        # 生成完整的标准时间索引 (UTC)
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=timeframe_str, tz="UTC"
        )

        if len(full_idx) != len(df):
            print(
                f"⚠️ [清洗] 发现时间断点: 理论 {len(full_idx)} 行, 实际 {len(df)} 行, 补全 {len(full_idx) - len(df)} 行"
            )
            df = df.reindex(full_idx)

            # 补全逻辑
            df["close"] = df["close"].ffill()  # 收盘价前值填充
            # Open/High/Low 如果缺失，暂时认为等于 Close (市场静止)
            df["open"] = df["open"].fillna(df["close"])
            df["high"] = df["high"].fillna(df["close"])
            df["low"] = df["low"].fillna(df["close"])
            # 成交量填 0
            df["volume"] = df["volume"].fillna(0)
            df["timestamp"] = df.index.astype("int64") // 10**6  # 补全 timestamp 列

        # 4. 逻辑错误修复 (High < Low)
        mask_error = df["high"] < df["low"]
        if mask_error.any():
            print(f"⚠️ [清洗] 修复 {mask_error.sum()} 行 High < Low 异常")
            # 交换 High 和 Low
            temp = df.loc[mask_error, "high"].copy()
            df.loc[mask_error, "high"] = df.loc[mask_error, "low"]
            df.loc[mask_error, "low"] = temp

        # 5. 删除开头可能产生的 NaN
        df = df.dropna()

        # 使用 original_count 显示清洗前后的变化，解决 Ruff F841 警告
        print(f"✅ [清洗] 完成. 行数变化: {original_count} -> {len(df)}")
        return df

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """
        保存为 Parquet 格式 (比 CSV 快且保留类型)
        """
        path = os.path.join(self.data_dir, filename)
        # 确保 timestamp 是 int 类型，方便后续处理
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype("int64")

        try:
            df.to_parquet(path)
            print(f"💾 [存储] 数据已保存至: {path}")
        except Exception as e:
            print(f"❌ [存储失败] {e}")

    # 使用 | None 表示可能返回 None
    def load_from_parquet(self, filename: str) -> pd.DataFrame | None:
        """
        读取本地 Parquet 数据
        """
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            return None

        try:
            df = pd.read_parquet(path)
            print(f"📂 [读取] 加载本地数据: {len(df)} 行")
            return df
        except Exception as e:
            print(f"❌ [读取失败] {e}")
            return None
