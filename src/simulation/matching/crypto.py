# src/simulation/matching/crypto.py

from typing import List, Union

from src.core.types import (
    BarData,
    Direction,
    OrderType,
    TickData,
    TradeData,
)
from src.simulation.base.matching import BaseMatchingEngine


class CryptoMatchingEngine(BaseMatchingEngine):
    """
    加密货币撮合引擎 (优化版)
    特点：
    1. T+0 (无限制)
    2. 7*24小时 (无停盘逻辑)
    3. 严谨撮合 (优先盘口价，限价单不吃亏，市价单计滑点)
    """

    def match(self, data: Union[TickData, BarData]) -> List[TradeData]:
        trades = []

        # -----------------------------
        # 1. 确定撮合基准价格
        # -----------------------------

        # 触发价：用于判断限价单是否成交
        # 成交价：用于计算市价单成交价格

        if isinstance(data, BarData):
            # --- Bar 数据模式 ---
            # 只有最低价跌破买入价，才能买入
            long_cross_price = data.low_price
            # 只有最高价涨破卖出价，才能卖出
            short_cross_price = data.high_price

            # 记录开盘价，用于处理“跳空”情况 (Gap)
            # 比如限价买单是100，今天直接低开在90，应该按90成交而不是100
            current_open_price = data.open_price

            # 市价单通常按收盘价撮合 (保守回测)
            market_buy_price = data.close_price + self.slippage
            market_sell_price = data.close_price - self.slippage

        else:
            # --- Tick 数据模式 ---
            if data.ask_price_1 > 0 and data.bid_price_1 > 0:
                # 有盘口深度：市价买入吃卖一(ask)，市价卖出吃买一(bid)
                market_buy_price = data.ask_price_1 + self.slippage
                market_sell_price = data.bid_price_1 - self.slippage

                # 限价单撮合基准
                # 买入看卖一(ask)，卖出看买一(bid)
                long_cross_price = data.ask_price_1
                short_cross_price = data.bid_price_1
            else:
                # 无盘口深度：降级使用最新价
                market_buy_price = data.last_price + self.slippage
                market_sell_price = data.last_price - self.slippage
                long_cross_price = data.last_price
                short_cross_price = data.last_price

            # Tick模式下，当前参考价即为最新价或盘口价
            current_open_price = long_cross_price

        # -----------------------------
        # 2. 遍历活动挂单进行撮合
        # -----------------------------

        # 使用 list(keys) 锁定迭代列表，避免在循环中删除字典元素报错
        for orderid in list(self.active_orders.keys()):
            order = self.active_orders[orderid]
            is_traded = False
            trade_price = 0.0

            # --- A. 市价单 (Market Order) ---
            if order.type == OrderType.MARKET:
                if order.direction == Direction.LONG:
                    trade_price = market_buy_price
                else:
                    trade_price = market_sell_price
                is_traded = True

            # --- B. 限价单 (Limit Order) ---
            elif order.type == OrderType.LIMIT:
                if order.direction == Direction.LONG:
                    # [买入逻辑]
                    # 如果市场最低价/卖一价 <= 限价，说明有机会成交
                    if long_cross_price <= order.price:
                        # 确定成交价：
                        # 1. 正常情况：价格在Bar内波动穿过限价，按限价成交。
                        # 2. 跳空情况：如果开盘/当前价已经比限价更便宜，按更便宜的价格成交（买入更优）。
                        # 注意：限价单绝对不能加 Slippage 导致价格高于 Limit
                        trade_price = min(order.price, current_open_price)
                        is_traded = True

                elif order.direction == Direction.SHORT:
                    # [卖出逻辑]
                    # 如果市场最高价/买一价 >= 限价，说明有机会成交
                    if short_cross_price >= order.price:
                        # 确定成交价：
                        # 1. 正常情况：按限价成交。
                        # 2. 跳空情况：如果开盘/当前价已经比限价更高，按更高的价格成交（卖出更优）。
                        trade_price = max(order.price, current_open_price)
                        is_traded = True

            # -----------------------------
            # 3. 生成成交回报
            # -----------------------------
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
                    volume=order.volume,  # 简化处理：默认全部成交
                    datetime=data.datetime,
                )
                trades.append(trade)

                # 成交后从活动订单列表移除
                del self.active_orders[orderid]

        return trades
