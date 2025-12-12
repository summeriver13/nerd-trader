from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Type

from src.core.types import TradeData


@dataclass
class BasePosition:
    """
    基础持仓数据结构
    """

    symbol: str
    volume: float = 0.0  # 总持仓 (负数代表空头)
    available_volume: float = 0.0  # 可用持仓 (T+0/T+1)
    avg_price: float = 0.0  # 持仓均价
    realized_pnl: float = 0.0  # [新增] 已实现盈亏 (累计)

    @property
    def market_value(self) -> float:
        """持仓市值 (成本价值，非最新市值)"""
        return abs(self.volume) * self.avg_price


class BaseAccount(ABC):
    """
    抽象账户基类
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        position_cls: Type[BasePosition] = BasePosition,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.frozen = 0.0
        self.trades: List[TradeData] = []

        # 自动创建指定类型的 Position
        self.positions: Dict[str, BasePosition] = defaultdict(
            lambda: position_cls(symbol="")
        )

    @abstractmethod
    def on_trade(self, trade: TradeData):
        """子类处理成交"""
        pass

    def settle_daily(self):
        """日终结算 (默认空实现)"""
        pass

    def get_position(self, symbol: str) -> BasePosition:
        """安全获取持仓"""
        # 如果是新创建的默认对象，补全 symbol
        if symbol not in self.positions:
            self.positions[symbol].symbol = symbol
        return self.positions[symbol]
