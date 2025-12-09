"""
src/core/base_account.py
账户基类模块：定义所有市场通用的账户行为
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Type

from src.core.types import TradeData


@dataclass
class BasePosition:
    """
    基础持仓数据结构
    所有市场的持仓都至少包含这些字段
    """

    symbol: str
    volume: float = 0.0  # 持仓数量
    available_volume: float = 0.0  # <---【新增】可用持仓 (通用字段)
    avg_price: float = 0.0  # 持仓均价
    realized_pnl: float = 0.0  # 已实现盈亏

    @property
    def market_value(self) -> float:
        """持仓市值 (成本价值)"""
        return self.volume * self.avg_price


class BaseAccount(ABC):
    """
    抽象账户基类

    设计模式：模板方法模式 (Template Method)
    - calculate_equity: 通用逻辑，直接在基类实现
    - on_trade: 差异逻辑，强制子类实现
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        position_cls: Type[BasePosition] = BasePosition,
    ):
        """
        :param initial_balance: 初始资金
        :param position_cls: 该账户使用的持仓类 (允许子类传入 StockPosition 或 CryptoPosition)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.frozen = 0.0
        self.trades: List[TradeData] = []

        # 使用工厂函数创建默认字典，确保自动创建正确的 Position 类型
        self.positions: Dict[str, BasePosition] = defaultdict(
            lambda: position_cls(symbol="")
        )

    @abstractmethod
    def on_trade(self, trade: TradeData):
        """
        [抽象方法] 处理成交回报
        子类必须实现此方法，因为：
        1. A股需处理印花税，Crypto需处理Maker/Taker费
        2. A股需处理T+1可用持仓，Crypto是T+0
        """
        pass

    def settle_daily(self):
        """
        [钩子方法] 日终结算
        A股必须覆盖此方法处理 T+1 转换。
        Crypto 默认可以是空实现，或者用于处理资金费率(Funding Fee)。
        """
        pass

    def calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """
        [通用方法] 计算账户总权益
        逻辑通用：权益 = 余额 + 所有持仓的最新市值
        """
        equity = self.balance

        for symbol, pos in self.positions.items():
            if pos.volume == 0:
                continue

            # 如果没有最新价，兜底使用持仓均价
            market_price = current_prices.get(symbol, pos.avg_price)
            equity += pos.volume * market_price

        return equity

    def get_position(self, symbol: str) -> BasePosition:
        """安全获取持仓对象的辅助方法"""
        if symbol not in self.positions:
            self.positions[symbol].symbol = symbol
        return self.positions[symbol]
