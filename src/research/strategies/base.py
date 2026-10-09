from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类：封装 Crypto 市场规则 (Market Rules)
    """

    def __init__(self) -> None:
        self.position: float = 0.0  # 当前仓位
        self.can_short: bool = True  # 永续合约允许做空
        self.max_leverage: float = 100.0  # 理论杠杆上限

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号"""
        pass
