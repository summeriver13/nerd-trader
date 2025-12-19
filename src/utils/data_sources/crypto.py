# src/utils/data_sources/crypto.py
from __future__ import annotations

import time
from datetime import datetime
from typing import cast

import ccxt
import pandas as pd

from .base import BaseDataSource


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
        all_candles: list[list[float]] = []  # 明确内部类型
        limit = 100

        print(f"⬇️ [CCXT] 获取 {symbol} ({timeframe}) ...")

        while since_ts < end_ts:
            try:
                # CCXT fetch_ohlcv 返回类型通常为 list[list[float]]
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ts, limit=limit
                )
                if not ohlcv:
                    break

                last_ts = int(ohlcv[-1][0])
                if last_ts <= since_ts:
                    break

                all_candles.extend(ohlcv)  # type: ignore
                since_ts = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                print(f"   ❌ CCXT Error: {e}")
                time.sleep(3)
                continue

        if not all_candles:
            return pd.DataFrame()

        # --- 核心修复点：使用 pd.Index ---
        columns = pd.Index(["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles, columns=columns)

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # 转换并返回
        result = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return cast(pd.DataFrame, result)
