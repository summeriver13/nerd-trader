"""
src/simulation/matching.py
撮合引擎：专门负责订单的撮合逻辑
Refactored: Decoupled from Gateway, handles OrderBook logic
"""

from src.core.types import (
    Direction,
    OrderData,
    OrderType,
    Status,
    TickData,
    TradeData,
)


class MatchingEngine:
    """
    撮合引擎
    职责：
    1. 维护挂单簿 (active_orders)
    2. 根据 Tick 行情撮合订单
    3. 生成成交单 (TradeData)
    """

    def __init__(self, slippage: float = 0.0):
        # 挂单字典: orderid -> OrderData
        self.active_orders: dict[str, OrderData] = {}
        self.slippage = slippage

    def add_order(self, order: OrderData):
        """接收新订单"""
        self.active_orders[order.orderid] = order

    def cancel_order(self, orderid: str) -> OrderData | None:
        """
        撤单
        :return: 如果撤单成功，返回被撤销的订单对象；否则返回 None
        """
        if orderid in self.active_orders:
            order = self.active_orders.pop(orderid)
            order.status = Status.CANCELLED
            return order
        return None

    def match(self, tick: TickData) -> list[TradeData]:
        """
        [核心] 接收行情，返回生成的成交单列表
        """
        trades = []

        # 遍历所有挂单 (注意：这里使用 list() 复制 keys，因为循环中可能删除字典元素)
        for orderid in list(self.active_orders.keys()):
            order = self.active_orders.get(orderid)
            if not order:
                continue

            trade = self._process_order(order, tick)
            if trade:
                trades.append(trade)

        return trades

    def _process_order(self, order: OrderData, tick: TickData) -> TradeData | None:
        """
        处理单个订单的撮合逻辑
        """
        # 1. 获取对手价
        ask_1 = tick.ask_price_1 or tick.last_price
        bid_1 = tick.bid_price_1 or tick.last_price

        traded_price = 0.0
        can_trade = False

        # 2. 判断市价单 (价格为0 或 类型为MARKET)
        is_market = (order.price == 0) or (order.type == OrderType.MARKET)

        # 3. 撮合判断
        if order.direction == Direction.LONG:
            # 买入：市价单，或者 限价 >= 卖一价
            if is_market or order.price >= ask_1:
                traded_price = ask_1 + self.slippage
                can_trade = True
        else:
            # 卖出：市价单，或者 限价 <= 买一价
            if is_market or order.price <= bid_1:
                traded_price = bid_1 - self.slippage
                can_trade = True

        # 4. 生成成交单
        if can_trade:
            # 更新订单状态
            order.status = Status.ALLTRADED
            order.traded = order.volume
            order.datetime = tick.datetime

            # 从挂单簿移除
            if order.orderid in self.active_orders:
                del self.active_orders[order.orderid]

            # 创建 TradeData
            trade = TradeData(
                gateway_name=order.gateway_name,
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=f"T_{order.orderid}",
                direction=order.direction,
                offset=order.offset,
                price=traded_price,
                volume=order.volume,
                datetime=tick.datetime,
            )
            return trade

        return None
