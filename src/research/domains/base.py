from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TradeCost:
    """
    交易成本明细数据类
    用于详细记录各项费用，方便后续分析（例如：分析到底交了多少印花税）
    """

    commission: pd.Series | float  # 佣金 (券商/交易所收取)
    tax: pd.Series | float  # 税费 (印花税等政府收取)
    slippage: pd.Series | float  # 滑点损耗 (冲击成本)
    total: pd.Series | float  # 总成本 = 佣金 + 税费 + 滑点


class MarketDomain(ABC):
    """
    市场领域模型抽象基类 (Abstract Base Class)

    职责：
    1. 定义计算交易成本的标准接口
    2. 定义信号调整的标准接口 (如手数取整、T+1限制等)
    """

    @abstractmethod
    def calculate_cost(
        self, trade_amount: pd.Series, is_sell: pd.Series | bool
    ) -> TradeCost:
        """
        计算交易成本

        :param trade_amount: 交易名义金额 (绝对值，总是 > 0)
        :param is_sell: 布尔序列，True 代表卖出，False 代表买入 (用于区分印花税等)
        :return: TradeCost 对象，包含各项费用的 Series
        """
        pass

    def adjust_signal(self, signals: pd.Series) -> pd.Series:
        """
        调整信号以符合市场规则 (Hook 方法)

        默认情况下不修改信号。
        子类可以重写此方法来实现最小下单量等交易所约束。

        :param signals: 原始策略信号
        :return: 调整后的信号
        """
        return signals
