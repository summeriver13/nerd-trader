from abc import ABC, abstractmethod

import pandas as pd

# EWM（ADX 等）没有固定窗口，按周期的倍数留收敛余量
EWM_CONVERGENCE_MULT = 5


class BaseStrategy(ABC):
    """
    策略基类：封装 Crypto 市场规则 (Market Rules)
    """

    def __init__(self) -> None:
        self.position: float = 0.0  # 当前仓位
        self.can_short: bool = True  # 永续合约允许做空
        self.max_leverage: float = 100.0  # 理论杠杆上限

    @property
    def warmup_bars(self) -> int:
        """指标预热所需根数：调用方据此决定抓多少额外历史、评估窗从哪根起步"""
        return 0

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号"""
        pass
