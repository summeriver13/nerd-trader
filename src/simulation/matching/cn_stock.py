"""
src/simulation/matching/cn_stock.py
A股撮合引擎
特点：
1. 兼容 Bar (OHLC) 和 Tick 数据
2. 严格的高低点穿价逻辑
3. 支持集合竞价跳空撮合 (Gap Logic)
"""

from typing import List, Union

from src.core.types import (
    BarData,
    Direction,
    OrderType,
    TickData,
    TradeData,
)
from src.simulation.base.matching import BaseMatchingEngine


class CNStockMatchingEngine(BaseMatchingEngine):
    """
    A股专用撮合引擎
    """

    def match(self, data: Union[TickData, BarData]) -> List[TradeData]:
        trades = []

        # 1. 提取价格信息
        if isinstance(data, BarData):
            # Bar模式：信息最全，撮合最准
            current_price = data.close_price  # 市价单基准
            high_price = data.high_price
            low_price = data.low_price
            open_price = data.open_price  # 用于判断跳空
            is_bar = True
        else:
            # Tick模式：退化为当前价撮合
            current_price = data.last_price
            high_price = data.last_price
            low_price = data.last_price
            open_price = data.last_price
            is_bar = False

        # 2. 遍历活动订单
        # 使用 list(keys) 复制列表，防止循环中删除元素导致报错
        for orderid in list(self.active_orders.keys()):
            order = self.active_orders[orderid]

            is_traded = False
            trade_price = 0.0

            # --- A. 市价单 (Market Order) ---
            if order.type == OrderType.MARKET:
                # 市价单假设立刻以当前价成交
                if order.direction == Direction.LONG:
                    trade_price = current_price + self.slippage
                else:
                    trade_price = current_price - self.slippage
                is_traded = True

            # --- B. 限价单 (Limit Order) ---
            elif order.type == OrderType.LIMIT:
                # === 买入撮合 ===
                if order.direction == Direction.LONG:
                    # 规则：市场最低价 <= 限价，才能买入
                    if low_price <= order.price:
                        # [进阶] 跳空低开处理
                        # 如果是Bar数据，且 开盘价 < 限价，说明直接买在更便宜的开盘价
                        # 例如：挂单10块买，今天低开在9.8，实际应该以9.8成交
                        if is_bar and open_price < order.price:
                            trade_price = open_price
                        else:
                            trade_price = order.price

                        trade_price += self.slippage
                        is_traded = True

                # === 卖出撮合 ===
                elif order.direction == Direction.SHORT:
                    # 规则：市场最高价 >= 限价，才能卖出
                    if high_price >= order.price:
                        # [进阶] 跳空高开处理
                        # 如果是Bar数据，且 开盘价 > 限价，说明直接卖在更贵的开盘价
                        # 例如：挂单10块卖，今天高开在10.2，实际应该以10.2成交
                        if is_bar and open_price > order.price:
                            trade_price = open_price
                        else:
                            trade_price = order.price

                        trade_price -= self.slippage
                        is_traded = True

            # 3. 生成成交记录
            if is_traded:
                self.trade_count += 1
                trade = TradeData(
                    gateway_name=order.gateway_name,
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=f"T_{self.trade_count}",
                    direction=order.direction,
                    offset=order.offset,
                    price=trade_price,
                    volume=order.volume,
                    datetime=data.datetime,
                )
                trades.append(trade)

                # 成交后从挂单簿移除
                del self.active_orders[orderid]

        return trades
