"""
src/core/base/matching.py
撮合引擎基类
"""

from abc import ABC, abstractmethod
from typing import Tuple

from src.core.types import BarData, OrderData, OrderStatus, TradeData


class BaseMatching(ABC):
    @abstractmethod
    def match_order(
        self, order: OrderData, bar: BarData
    ) -> Tuple[bool, TradeData | None]:
        """
        [抽象方法] 尝试撮合订单

        :param order: 待撮合的订单
        :param bar: 当前K线数据 (包含 Open, High, Low, Close, Volume)
        :return: (is_finished, trade_data)
                 - is_finished: 订单是否结束 (Success/Cancelled/Rejected)
                 - trade_data: 如果成交，返回成交细节；否则为 None
        """
        pass

    def check_trade_time(self, timestamp) -> bool:
        """
        [可选] 交易时间检查，默认全天候
        """
        return True
