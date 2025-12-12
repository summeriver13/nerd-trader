from datetime import datetime
from typing import Iterator, Union

import pandas as pd

# 假设你有 BarData，如果没有，建议在 types.py 里加上
from src.core.types import BarData, TickData


class ReplayLoader:
    def __init__(self, file_path: str, symbol: str = "UNKNOWN", is_bar: bool = True):
        """
        :param is_bar: True 表示加载的是 K线数据(OHLC)，False 表示逐笔成交数据(Tick)
        """
        self.file_path = file_path
        self.symbol = symbol
        self.is_bar = is_bar

    def load_parquet(self) -> Iterator[Union[TickData, BarData]]:
        try:
            df = pd.read_parquet(self.file_path)
            # 兼容：如果 Parquet 里索引是 datetime，就不用 sort_values
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            elif "datetime" in df.columns:
                df = df.sort_values("datetime")

            print(
                f"Loader: 已加载 {len(df)} 条数据 (模式: {'Bar' if self.is_bar else 'Tick'})"
            )

            for row in df.itertuples():
                if self.is_bar:
                    yield self._row_to_bar(row)
                else:
                    yield self._row_to_tick(row)

        except FileNotFoundError:
            print(f"Error: 文件未找到 {self.file_path}")
            return

    def _row_to_bar(self, row) -> BarData:
        """解析 K线数据 (保留 OHLC)"""
        dt_val = getattr(row, "datetime", getattr(row, "Index", datetime.now()))
        if not isinstance(dt_val, datetime):
            dt_val = pd.to_datetime(dt_val).to_pydatetime()

        return BarData(
            gateway_name="BACKTEST",
            symbol=getattr(row, "symbol", self.symbol),
            exchange="MOCK",
            datetime=dt_val,
            interval="1m",  # 假设是1分钟，可作为参数传入
            open_price=float(getattr(row, "open", 0)),
            high_price=float(getattr(row, "high", 0)),
            low_price=float(getattr(row, "low", 0)),
            close_price=float(getattr(row, "close", 0)),
            volume=float(getattr(row, "volume", 0)),
        )

    def _row_to_tick(self, row) -> TickData:
        """解析 Tick 数据"""
        dt_val = getattr(row, "datetime", getattr(row, "Index", datetime.now()))
        if not isinstance(dt_val, datetime):
            dt_val = pd.to_datetime(dt_val).to_pydatetime()

        # 优先找 last_price, 其次 close
        price = getattr(row, "last_price", getattr(row, "close", 0.0))
        vol = float(getattr(row, "volume", 0))

        return TickData(
            gateway_name="BACKTEST",
            symbol=getattr(row, "symbol", self.symbol),
            exchange="MOCK",
            datetime=dt_val,
            last_price=float(price),
            volume=vol,
            # 模拟盘口：直接用最新价填充
            bid_price_1=float(price),
            ask_price_1=float(price),
            bid_volume_1=vol,
            ask_volume_1=vol,
        )
