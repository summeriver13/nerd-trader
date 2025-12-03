from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类：所有具体策略都必须继承此类
    """

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标 (MA, RSI, etc.)
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        必须返回包含 'signal' 列的 DataFrame (1:做多, 0:空仓, -1:做空)
        """
        pass

    @abstractmethod
    def optimize(self, df: pd.DataFrame) -> dict:
        """
        参数优化接口
        """
        pass
