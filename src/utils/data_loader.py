"""
src/utils/data_loader.py
[Fundamental Update]
全能数据加载器：支持 Crypto + A股股票 + A股ETF
✅ 新增：Baostock 接口现在会拉取 PE (市盈率) 和 PB (市净率) 数据
✅ 兼容：ETF 识别与免税逻辑保持不变
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import cast

import ccxt
import pandas as pd

# 尝试导入数据源库
try:
    import tushare as ts
except ImportError:
    ts = None
try:
    import akshare as ak
except ImportError:
    ak = None
try:
    import baostock as bs
except ImportError:
    bs = None

from config.settings import GLOBAL_CONFIG


class BaseDataSource(ABC):
    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        pass

    def normalize_symbol(self, symbol: str) -> str:
        return symbol

    def is_etf(self, symbol: str) -> bool:
        """
        [ETF 核心逻辑] 简单判断是否为 ETF 代码
        15/16: 深市 ETF/LOF
        51/56/58: 沪市 ETF/科创板ETF
        """
        s = symbol.split(".")[0]
        return s.startswith(("51", "56", "58", "15", "16"))


class CCXTSource(BaseDataSource):
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if not self.exchange:
            return pd.DataFrame()

        since_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        all_candles = []
        limit = 100

        print(f"⬇️ [CCXT] 获取 {symbol} ({timeframe}) ...")

        while since_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ts, limit=limit
                )
                if not ohlcv:
                    break

                last_ts = ohlcv[-1][0]
                if last_ts <= since_ts:
                    break

                all_candles.extend(ohlcv)
                since_ts = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                print(f"   ❌ CCXT Error: {e}")
                time.sleep(3)
                continue

        if not all_candles:
            return pd.DataFrame()

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(all_candles, columns=pd.Index(cols))

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        result = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return cast(pd.DataFrame, result)


class TushareSource(BaseDataSource):
    def __init__(self):
        token = getattr(GLOBAL_CONFIG, "TUSHARE_TOKEN", "")
        if ts and token and token != "你的Tushare_Token_填在这里":
            ts.set_token(token)
            self.pro = ts.pro_api()
        else:
            self.pro = None

    def normalize_symbol(self, symbol: str) -> str:
        # Tushare 格式: 600519.SH, 512880.SH, 159915.SZ
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol
        if symbol.startswith(("5", "6")):
            return f"{symbol}.SH"
        if symbol.startswith(("0", "1", "3")):
            return f"{symbol}.SZ"
        if symbol.startswith(("4", "8")):
            return f"{symbol}.BJ"
        return symbol

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if ts is None or self.pro is None:
            return pd.DataFrame()
        if timeframe != "1d":
            return pd.DataFrame()

        ts_code = self.normalize_symbol(symbol)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        asset_type = "E"
        if self.is_etf(symbol):
            asset_type = "FD"

        print(f"⬇️ [Tushare] 获取 {ts_code} (QFQ, Asset={asset_type}) ...")
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                adj="qfq",
                asset=asset_type,
                start_date=start_str,
                end_date=end_str,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={"trade_date": "datetime", "vol": "volume"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()

            dti = cast(pd.DatetimeIndex, df.index)
            df.index = dti.tz_localize("UTC")

            # Tushare pro_bar 默认不带 PE/PB，需要额外调用 daily_basic
            # 这里保持原样，只返回价格数据
            result = df[["open", "high", "low", "close", "volume"]]
            return cast(pd.DataFrame, result)
        except Exception as e:
            print(f"   ❌ Tushare Error: {e}")
            return pd.DataFrame()


class AkShareSource(BaseDataSource):
    def normalize_symbol(self, symbol: str) -> str:
        return symbol.split(".")[0]

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if not ak:
            return pd.DataFrame()

        code = self.normalize_symbol(symbol)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        period = "daily" if timeframe == "1d" else "weekly" if timeframe == "1w" else ""
        if not period:
            return pd.DataFrame()

        print(f"⬇️ [AkShare] 获取 {code} (QFQ) ...")
        try:
            if self.is_etf(code):
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period=period,
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period=period,
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )

            if df.empty:
                return pd.DataFrame()

            rename_map = {
                "日期": "datetime",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
            df = df.rename(columns=rename_map)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()

            dti = cast(pd.DatetimeIndex, df.index)
            df.index = dti.tz_localize("UTC")

            result = df[["open", "high", "low", "close", "volume"]]
            return cast(pd.DataFrame, result)
        except Exception as e:
            print(f"   ❌ AkShare Error: {e}")
            return pd.DataFrame()


class BaostockSource(BaseDataSource):
    def normalize_symbol(self, symbol: str) -> str:
        code = symbol.lower().split(".")[0]
        if code.startswith(("5", "6")):
            return f"sh.{code}"
        return f"sz.{code}"

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if bs is None:
            return pd.DataFrame()

        bs_code = self.normalize_symbol(symbol)
        freq_map = {"1d": "d", "5m": "5", "15m": "15", "30m": "30", "1h": "60"}
        freq = freq_map.get(timeframe, "")
        if not freq:
            return pd.DataFrame()

        # === [新增] 只有日线才去拉取 PE/PB ===
        if timeframe == "1d":
            # peTTM: 滚动市盈率, pbMRQ: 市净率
            fields = "date,open,high,low,close,volume,peTTM,pbMRQ"
        else:
            fields = "date,time,open,high,low,close,volume"

        print(f"⬇️ [Baostock] 获取 {bs_code} ({timeframe}) + 估值数据 ...")
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                frequency=freq,
                adjustflag="2",
            )

            if rs is None:
                bs.logout()
                return pd.DataFrame()

            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=pd.Index(rs.fields))

            # 1. 基础价格转数字
            price_cols = ["open", "high", "low", "close", "volume"]
            df[price_cols] = df[price_cols].apply(pd.to_numeric)

            # 2. [新增] 财务指标转数字并重命名
            # Baostock 返回 peTTM, pbMRQ -> 我们统一改为 pe, pb
            if "peTTM" in df.columns:
                df["pe"] = pd.to_numeric(df["peTTM"], errors="coerce")
            if "pbMRQ" in df.columns:
                df["pb"] = pd.to_numeric(df["pbMRQ"], errors="coerce")

            # 3. 时间处理
            if "time" in df.columns and df["time"].iloc[0]:
                df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S000")
            else:
                df["datetime"] = pd.to_datetime(df["date"])

            df = df.set_index("datetime").sort_index()

            dti = cast(pd.DatetimeIndex, df.index)
            df.index = dti.tz_localize("UTC")

            # 4. [新增] 动态返回列
            # 如果有 pe/pb 则一起返回，否则只返回价格
            final_cols = ["open", "high", "low", "close", "volume"]
            if "pe" in df.columns:
                final_cols.append("pe")
            if "pb" in df.columns:
                final_cols.append("pb")

            return cast(pd.DataFrame, df[final_cols])

        except Exception as e:
            print(f"   ❌ Baostock Error: {e}")
            return pd.DataFrame()


class MarketDataHandler:
    def __init__(self, exchange: ccxt.Exchange | None = None):
        self.exchange = exchange
        self.data_root = Path("data") / "storage"

        if self.exchange:
            self.primary_source = CCXTSource(self.exchange)
            self.fallback_sources = []
        else:
            self.primary_source = TushareSource()
            # [调整顺序] 将 Baostock 提到 AkShare 之前
            # 因为现在我们更想用 Baostock 的 PE/PB 数据来做基本面测试
            self.fallback_sources = [BaostockSource(), AkShareSource()]

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        if self.exchange:
            market = "crypto"
            source = getattr(self.exchange, "id", "ccxt_unknown")
        else:
            market = "stock_cn"
            source = "combined"

        safe_symbol = symbol.replace("/", "-").upper()
        return self.data_root / market / source / timeframe / f"{safe_symbol}.parquet"

    def prepare_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 365,
        force_update: bool = False,
    ) -> pd.DataFrame:
        file_path = self._get_file_path(symbol, timeframe)
        df = self.load_from_parquet(file_path)

        target_start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        download_start_dt: datetime = cast(datetime, target_start_dt)

        now_utc = pd.Timestamp.now(tz="UTC")
        need_download = False
        is_append = False

        if df is None or df.empty or force_update:
            need_download = True
        else:
            dti = cast(pd.DatetimeIndex, df.index)
            if dti.tz is None:
                df.index = dti.tz_localize("UTC")
            else:
                df.index = dti.tz_convert("UTC")

            dti = cast(pd.DatetimeIndex, df.index)
            last_dt = dti[-1]

            if last_dt < now_utc - pd.Timedelta(hours=24):
                print(f"⚠️ [数据] 数据已过时 (止于 {last_dt}), 准备增量更新...")
                need_download = True
                is_append = True
                download_start_dt = cast(datetime, last_dt)

        if need_download:
            df_new = self._fetch_from_sources(
                symbol, timeframe, download_start_dt, now_utc
            )

            if not df_new.empty:
                if is_append and df is not None:
                    # 只有当新旧数据的列一致时才能合并
                    # 如果旧数据没有 PE 而新数据有，可能会产生 NaN，Pandas 会自动处理
                    print(f"🔗 [合并] 旧({len(df)}) + 新({len(df_new)})")
                    df_final = pd.concat([df, df_new])
                else:
                    df_final = df_new

                df_final = self.clean_data(df_final, timeframe)
                self.save_to_parquet(df_final, file_path)
                df = df_final
            else:
                print("⚠️ 未能获取到新数据")

        if df is None or df.empty:
            return pd.DataFrame()

        dti = cast(pd.DatetimeIndex, df.index)
        if dti.tz is None:
            df.index = dti.tz_localize("UTC")
        else:
            df.index = dti.tz_convert("UTC")

        buffer_days = int(days_back * 0.05) + 1
        cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=days_back + buffer_days
        )

        result = df[df.index >= cutoff_dt].copy()
        return cast(pd.DataFrame, result)

    def _fetch_from_sources(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        # 动态调整策略：如果是请求日线，优先尝试 Baostock 以获取 PE/PB
        if timeframe == "1d" and not self.exchange:
            # 临时将 Baostock 提权，为了获取估值数据
            sources = [BaostockSource(), self.primary_source] + [
                s for s in self.fallback_sources if not isinstance(s, BaostockSource)
            ]
        else:
            sources = [self.primary_source] + self.fallback_sources

        for source in sources:
            if isinstance(source, TushareSource) and timeframe != "1d":
                continue
            if isinstance(source, AkShareSource) and timeframe not in ["1d", "1w"]:
                continue

            df = source.fetch_bars(symbol, timeframe, start, end)
            if not df.empty:
                return df

        return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame, timeframe_str: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")]  # type: ignore
        df = df.sort_index()

        if self.exchange is None:
            return df

        pd_freq = self._convert_to_pandas_freq(timeframe_str)
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=pd_freq, tz="UTC"
        )
        if len(full_idx) != len(df):
            df = df.reindex(full_idx)
            df["close"] = df["close"].ffill()
            df["open"] = df["open"].fillna(df["close"])
            df["high"] = df["high"].fillna(df["close"])
            df["low"] = df["low"].fillna(df["close"])
            df["volume"] = df["volume"].fillna(0)
            # PE/PB 也要前向填充 (基本面数据变化慢)
            if "pe" in df.columns:
                df["pe"] = df["pe"].ffill()
            if "pb" in df.columns:
                df["pb"] = df["pb"].ffill()

        mask = df["high"] < df["low"]
        if mask.any():
            t = df.loc[mask, "high"].copy()
            df.loc[mask, "high"] = df.loc[mask, "low"]
            df.loc[mask, "low"] = t

        result = df.dropna()
        return cast(pd.DataFrame, result)

    def _convert_to_pandas_freq(self, timeframe: str) -> str:
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

    def save_to_parquet(self, df: pd.DataFrame, file_path: Path):
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.copy().to_parquet(file_path)
        except Exception as e:
            print(f"❌ [存储失败] {e}")

    def load_from_parquet(self, file_path: Path) -> pd.DataFrame | None:
        if not file_path.exists():
            return None
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"❌ [读取失败] {e}")
            return None
