from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类：所有具体策略都必须继承此类
    """

    def __init__(self) -> None:
        """
        初始化策略状态
        """
        self.position: int = 0  # 当前仓位状态 (0:空, 1:多, -1:空)

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标 (MA, RSI, Bollinger, etc.)
        :param df: 原始 OHLCV 数据
        :return: 包含指标列的 DataFrame
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        必须返回包含 'signal' 列的 DataFrame (1:做多, 0:空仓, -1:做空)
        """
        pass

    def optimize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        参数优化接口 (可选实现)
        如果不实现，默认抛出异常提示
        """
        raise NotImplementedError(
            f"策略 {self.__class__.__name__} 未实现参数优化方法 (optimize)"
        )
