"""
src/simulation/replay_loader.py
历史数据加载器：负责读取文件并转换为标准的 TickData/BarData 生成器
"""

from datetime import datetime
from typing import Iterator

import pandas as pd

from src.core.types import TickData


class ReplayLoader:
    """
    回放数据加载器
    支持 Parquet, CSV 等格式，统一产出 TickData
    """

    def __init__(self, file_path: str, symbol: str = "UNKNOWN"):
        self.file_path = file_path
        self.symbol = symbol

    def load_parquet(self) -> Iterator[TickData]:
        """
        读取 Parquet 并生成 TickData 流
        使用 yield 关键字，节省内存 (Generator)
        """
        try:
            df = pd.read_parquet(self.file_path)
            # 确保排序
            if "datetime" in df.columns:
                df = df.sort_values("datetime")
            else:
                df = df.sort_index()

            print(f"Loader: 已加载 {len(df)} 条数据，准备回放...")

            # 遍历并转换
            for row in df.itertuples():
                yield self._row_to_tick(row)

        except FileNotFoundError:
            print(f"Error: 文件未找到 {self.file_path}")
            return

    def _row_to_tick(self, row) -> TickData:
        """
        [私有方法] 将 Pandas 的一行转换为标准 TickData
        这里处理所有的 字段映射 和 类型转换 脏活
        """
        # 1. 解析时间
        dt_val = getattr(row, "datetime", None)
        if dt_val is None:
            dt_val = getattr(row, "Index", None)

        if dt_val is None:
            final_dt = datetime.now()
        else:
            final_dt = pd.to_datetime(dt_val).to_pydatetime()

        # 2. 解析价格 (兼容不同的列名)
        # 优先找 close，找不到找 last_price，再找不到找 price
        price = getattr(
            row, "close", getattr(row, "last_price", getattr(row, "price", 0.0))
        )

        vol = float(getattr(row, "volume", 0))

        return TickData(
            gateway_name="BACKTEST",
            symbol=getattr(row, "symbol", self.symbol),
            exchange="MOCK",
            datetime=final_dt,
            last_price=float(price),
            volume=vol,
            bid_price_1=float(price),
            ask_price_1=float(price),
        )
