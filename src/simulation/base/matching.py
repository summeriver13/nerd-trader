from abc import ABC, abstractmethod
from typing import List, Union

from src.core.types import BarData, OrderData, TickData, TradeData


class BaseMatchingEngine(ABC):
    def __init__(self, slippage: float = 0.0):
        self.slippage = slippage
        self.active_orders: dict[str, OrderData] = {}
        self.trade_count = 0

    def add_order(self, order: OrderData):
        self.active_orders[order.orderid] = order

    def cancel_order(self, orderid: str) -> OrderData | None:
        if orderid in self.active_orders:
            order = self.active_orders.pop(orderid)
            return order
        return None

    @abstractmethod
    def match(self, data: Union[TickData, BarData]) -> List[TradeData]:
        pass
