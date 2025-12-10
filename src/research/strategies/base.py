from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类：封装了通用的市场规则 (Market Rules)
    """

    def __init__(self, market_type: str = "crypto") -> None:
        """
        初始化策略状态与市场规则
        :param market_type: 'crypto' 或 'stock_cn'
        """
        self.market_type = market_type.lower()
        self.position: float = 0.0  # 当前仓位

        # === 统一配置市场规则 ===
        if self.market_type == "stock_cn":
            self.can_short = False  # A股禁止做空
            self.max_leverage = 1.0  # A股禁止杠杆
            self.t_plus_one = True  # T+1 限制 (虽然向量化很难模拟，但标记一下)
        else:
            # Crypto 默认规则
            self.can_short = True  # 允许做空
            self.max_leverage = 100.0  # 理论杠杆上限
            self.t_plus_one = False  # T+0

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号"""
        pass

    def optimize(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """参数优化接口 (默认实现)"""
        print(f"⚠️ 策略 {self.__class__.__name__} 未实现 optimize 方法，跳过优化。")
        return {}
