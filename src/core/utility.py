"""
src/core/utility.py
通用工具组件：K线合成器 & 数组管理器
Refactored: Removed unused variables & Cleaned up logic
"""

from collections.abc import Callable
from copy import copy

import numpy as np
import talib

from .types import BarData, TickData


class BarGenerator:
    """
    K线合成器
    功能：接收 Tick，合成 1分钟 K线
    """

    def __init__(self, on_bar: Callable[[BarData], None], xmin: int = 0):
        """
        :param on_bar: 当K线合成完成时的回调函数
        :param xmin: 0=合成1分钟线, N=合成N分钟线 (暂未实现N分钟, 仅演示1分钟)
        """
        self.bar: BarData | None = None
        self.on_bar = on_bar
        self.last_tick: TickData | None = None

    def update_tick(self, tick: TickData):
        """
        收到 Tick 推送
        """
        # 1. 如果还没有 Bar，初始化第一个 (这通常是每天的第一个 Tick)
        if not self.bar:
            self.bar = BarData(
                gateway_name=tick.gateway_name,
                symbol=tick.symbol,
                exchange=tick.exchange,
                datetime=tick.datetime,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                volume=tick.volume,  # 实盘中通常需要做差分计算，这里简化
            )
        else:
            # 2. 如果上一笔 Tick 存在，检查是否跨越了分钟
            if (
                self.last_tick
                and self.last_tick.datetime.minute != tick.datetime.minute
            ):
                # A. 这是一个新分钟的开始 -> 上一根 Bar 结束了
                # 生成上一分钟的完整 Bar (使用深拷贝防止引用被修改)
                finished_bar = copy(self.bar)

                # 推送上一根完成的 Bar 给策略
                self.on_bar(finished_bar)

                # B. 重置当前 Bar 为新的一根 (使用当前 Tick 数据)
                self.bar.open_price = tick.last_price
                self.bar.high_price = tick.last_price
                self.bar.low_price = tick.last_price
                self.bar.close_price = tick.last_price
                self.bar.volume = 0  # 新的一分钟，成交量归零 (或者等于当前tick.volume)
                self.bar.datetime = tick.datetime
            else:
                # C. 还是同一分钟 -> 仅更新当前 Bar 的数据
                self.bar.high_price = max(self.bar.high_price, tick.last_price)
                self.bar.low_price = min(self.bar.low_price, tick.last_price)
                self.bar.close_price = tick.last_price
                self.bar.volume += tick.volume
                self.bar.datetime = tick.datetime

        # 更新上一笔 Tick 缓存
        self.last_tick = tick

    def update_bar(self, bar: BarData):
        """
        直接接收 Bar 数据 (用于 x分钟线的合成，例如 1m -> 15m)
        """
        pass


class ArrayManager:
    """
    K线数据序列容器
    功能：保存最近 N 根 K 线，提供技术指标计算
    """

    def __init__(self, size: int = 100):
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        # 使用 numpy 数组以获得高性能
        self.open_array = np.zeros(size)
        self.high_array = np.zeros(size)
        self.low_array = np.zeros(size)
        self.close_array = np.zeros(size)
        self.volume_array = np.zeros(size)

    def update_bar(self, bar: BarData):
        """
        放入一根新 Bar
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        # 整体左移一位 (抛弃最旧的)
        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]

        # 填入最新的
        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume

    @property
    def close(self) -> np.ndarray:
        return self.close_array

    @property
    def high(self) -> np.ndarray:
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        return self.low_array

    # --- 指标计算 (封装 talib) ---

    def sma(self, n: int, array: bool = False) -> float | np.ndarray:
        """简单移动平均"""
        result = talib.SMA(self.close, timeperiod=n)
        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False) -> float | np.ndarray:
        """平均真实波幅"""
        result = talib.ATR(self.high, self.low, self.close, timeperiod=n)
        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> float | np.ndarray:
        """RSI"""
        result = talib.RSI(self.close, timeperiod=n)
        if array:
            return result
        return result[-1]
