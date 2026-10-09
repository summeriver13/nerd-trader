"""
src/simulation/replay_loader.py
回放数据加载器
包含：
1. ReplayLoader: 单标的加载器 (支持 Parquet/CSV)
2. MultiSymbolLoader: 多标的归并加载器 (基于优先队列合并)
"""

import heapq
from collections.abc import Generator, Iterator
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.types import BarData, TickData


class ReplayLoader:
    """
    单标的历史数据加载器
    负责读取 Parquet 文件并转换为系统标准的 TickData 或 BarData 对象
    """

    def __init__(self, file_path: str, symbol: str = "UNKNOWN", is_bar: bool = True):
        """
        :param file_path: 数据文件路径
        :param symbol: 标的代码
        :param is_bar: True=K线模式, False=Tick模式
        """
        self.file_path = file_path
        self.symbol = symbol
        self.is_bar = is_bar

    def load_data(self) -> Generator[TickData | BarData, None, None]:
        """
        加载数据并生成迭代器
        使用 yield 节省内存，支持流式回放
        """
        try:
            # 读取 Parquet (依赖 pyarrow 或 fastparquet)
            df = pd.read_parquet(self.file_path)

            # 1. 索引/排序列处理
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            elif "datetime" in df.columns:
                df = df.sort_values("datetime")
            else:
                print(f"Warning: 文件 {self.file_path} 缺少 datetime 字段或索引")
                return

            print(
                f"Loader: [{self.symbol}] 已加载 {len(df)} 条 (模式: {'Bar' if self.is_bar else 'Tick'})"
            )

            # 2. 使用 itertuples 遍历 (比 iterrows 快得多)
            for row in df.itertuples():
                if self.is_bar:
                    yield self._row_to_bar(row)
                else:
                    yield self._row_to_tick(row)

        except FileNotFoundError:
            print(f"Error: 文件未找到 {self.file_path}")
            return
        except Exception as e:
            print(f"Error: 加载 {self.symbol} 失败: {e}")
            return

    def _row_to_bar(self, row: Any) -> BarData:
        """解析 K线数据 (保留 OHLC)"""
        # 兼容 Index 为 datetime 的情况
        dt_val = getattr(row, "datetime", getattr(row, "Index", datetime.now()))
        if not isinstance(dt_val, datetime):
            dt_val = pd.to_datetime(dt_val).to_pydatetime()

        return BarData(
            gateway_name="BACKTEST",
            symbol=getattr(row, "symbol", self.symbol),
            exchange="MOCK",
            datetime=dt_val,
            interval="1m",  # 默认 interval, 也可以从文件名解析
            open_price=float(getattr(row, "open", 0)),
            high_price=float(getattr(row, "high", 0)),
            low_price=float(getattr(row, "low", 0)),
            close_price=float(getattr(row, "close", 0)),
            volume=float(getattr(row, "volume", 0)),
            turnover=float(getattr(row, "turnover", 0)),  # 支持成交额
            open_interest=float(getattr(row, "open_interest", 0)),
        )

    def _row_to_tick(self, row: Any) -> TickData:
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
            # 模拟盘口：回测如果没有盘口数据，用最新价填充
            bid_price_1=float(getattr(row, "bid_price_1", price)),
            ask_price_1=float(getattr(row, "ask_price_1", price)),
            bid_volume_1=float(getattr(row, "bid_volume_1", vol)),
            ask_volume_1=float(getattr(row, "ask_volume_1", vol)),
        )


class MultiSymbolLoader:
    """
    多标的归并加载器
    将多个 ReplayLoader 的数据流合并为一个按时间严格排序的流。
    原理：使用最小堆 (Min-Heap) 进行多路归并排序。
    """

    def __init__(self, loaders: list[ReplayLoader]):
        self.loaders = loaders

    def merge_streams(self) -> Iterator[TickData | BarData]:
        """
        生成归并后的数据流
        """
        # 1. 初始化所有加载器的生成器
        # iterators 列表存储 (生成器, 标的名称)
        iterators = []
        for loader in self.loaders:
            gen = loader.load_data()
            iterators.append(gen)

        # 2. 初始化堆
        # 堆元素结构: (datetime, unique_id, data, generator_index)
        # unique_id 用于在时间相同时打破平局，避免比较 data 对象
        heap = []
        for i, gen in enumerate(iterators):
            try:
                data = next(gen)
                # i 作为 unique_id，确保相同时间下，按列表顺序弹出
                heapq.heappush(heap, (data.datetime, i, data, i))
            except StopIteration:
                pass  # 该文件为空，跳过

        # 3. 循环弹出最小时间的数据
        while heap:
            # 弹出堆顶 (最早时间)
            timestamp, _, data, gen_idx = heapq.heappop(heap)

            yield data

            # 4. 从来源生成器补充下一条数据
            try:
                next_data = next(iterators[gen_idx])
                heapq.heappush(heap, (next_data.datetime, gen_idx, next_data, gen_idx))
            except StopIteration:
                pass  # 该生成器已耗尽，不再入堆
