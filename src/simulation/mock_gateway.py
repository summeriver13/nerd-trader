"""
src/simulation/mock_gateway.py
仿真网关
功能：
1. 接收策略订单
2. 模拟撮合 (支持 Tick 和 Bar)
3. 维护账户资金与持仓
Refactored: 支持依赖注入，同时提供默认的简易撮合与账户管理
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Union

from src.core.event import EventEngine, EventType
from src.core.interfaces import GatewayBase
from src.core.types import (
    BarData,
    Direction,
    OrderData,
    OrderRequest,
    OrderType,
    PositionData,
    Status,
    TickData,
    TradeData,
)


DEFAULT_INITIAL_CAPITAL = 1_000_000.0


class MockGateway(GatewayBase):
    """
    仿真网关
    既可以独立运行 (使用内置的简易逻辑)，
    也可以配合外部注入的 MatchingEngine 和 AccountManager 使用。
    """

    def __init__(
        self,
        event_engine: EventEngine,
        gateway_name: str = "BACKTEST",
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        # [可选] 未来扩展：支持外部注入
        matching_engine=None,
        account_manager=None,
    ):
        super().__init__(event_engine, gateway_name)

        # 1. 内部状态管理 (简易版账户与订单系统)
        self.orders: Dict[str, OrderData] = {}  # orderid -> OrderData
        self.active_orders: Dict[str, OrderData] = {}  # orderid -> OrderData

        # 资金与持仓
        self.balance = initial_capital
        self.positions: Dict[str, PositionData] = defaultdict(
            lambda: PositionData(symbol="")
        )
        self.last_price: Dict[str, float] = {}  # symbol -> 最新价，供 mark-to-market 用
        self.subscribed: set[str] = set()  # 已订阅标的（回测里默认全订阅，仅作记录）

        # 2. 注入组件 (暂留接口，如果传入则使用外部逻辑，目前使用内部逻辑)
        self.matcher = matching_engine
        self.account = account_manager

    def connect(self) -> None:
        """连接"""
        self.write_log(f"仿真网关 [{self.gateway_name}] 启动成功")
        self.write_log(f"初始资金: {self.balance:,.2f}")

    def close(self) -> None:
        """断开（回测里只记录，真实网关在此释放连接与订阅）"""
        self.write_log(f"仿真网关 [{self.gateway_name}] 已断开")

    def subscribe(self, symbol: str) -> None:
        """订阅行情"""
        self.subscribed.add(symbol)

    def send_order(self, req: OrderRequest) -> str:
        """发送订单"""
        orderid = f"{self.gateway_name}.{datetime.now().strftime('%H%M%S%f')}"

        order = OrderData(
            gateway_name=self.gateway_name,
            symbol=req.symbol,
            exchange=req.exchange,
            orderid=orderid,
            direction=req.direction,
            offset=req.offset,
            type=req.type,
            price=req.price,
            volume=req.volume,
            status=Status.NOTTRADED,
            datetime=datetime.now(),
            traded=0,
        )

        self.orders[orderid] = order
        self.active_orders[orderid] = order

        # 推送订单状态：未成交
        self.on_event(EventType.ORDER, order)

        return orderid

    def cancel_order(self, orderid: str) -> None:
        """撤单：标的不必传，从本地订单里查"""
        if orderid in self.active_orders:
            order = self.active_orders.pop(orderid)
            order.status = Status.CANCELLED
            order.time = datetime.now().strftime("%H:%M:%S")
            self.on_event(EventType.ORDER, order)
            self.write_log(f"订单已撤销: {orderid} ({order.symbol})")
        else:
            self.write_log(f"撤单失败: 订单 {orderid} 不存在或已完结")

    def process_tick(self, tick: TickData):
        """处理 Tick 行情"""
        # 1. 更新最新价并推送行情
        self.last_price[tick.symbol] = tick.last_price
        self.on_event(EventType.TICK, tick)

        # 2. 撮合
        self._match_orders(tick)

    def process_bar(self, bar: BarData):
        """处理 Bar 行情"""
        # 1. 记录最新价，供 mark-to-market 估值
        self.last_price[bar.symbol] = bar.close_price

        # 2. 先撮合：此时挂着的订单来自上一根 K 线，只能用本根开盘价成交，避免前视
        self._match_orders(bar)

        # 3. 再推送给策略：策略本根做出的决策留给下一根 K 线成交
        self.on_event(EventType.BAR, bar)

    def equity_value(self) -> float:
        """账户权益 = 现金 + 持仓市值（按各标的最新价 mark-to-market）"""
        return self.balance + sum(
            pos.volume * self.last_price.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )

    def _match_orders(self, data: Union[TickData, BarData]):
        """
        [核心撮合逻辑] 基于当前行情撮合限价单与市价单
        """
        if not self.active_orders:
            return

        # 提取当前价格信息
        if isinstance(data, TickData):
            buy_cross_price = data.ask_price_1 if data.ask_price_1 else data.last_price
            sell_cross_price = data.bid_price_1 if data.bid_price_1 else data.last_price
            market_price = data.last_price
            current_time = data.datetime
        else:
            # BarData: 使用 High/Low 进行更精确的撮合
            buy_cross_price = data.low_price  # 买单只要最低价低于限价即可成交
            sell_cross_price = data.high_price  # 卖单只要最高价高于限价即可成交
            market_price = data.open_price  # 市价单以本根开盘价成交
            current_time = data.datetime

        traded_orders = []

        for orderid, order in list(self.active_orders.items()):
            # 过滤非本标的订单
            if order.symbol != data.symbol:
                continue

            # --- 撮合判断 ---
            is_traded = False
            trade_price = 0.0

            # 市价单：无条件按当前可得价格成交
            if order.type == OrderType.MARKET:
                is_traded = True
                trade_price = market_price

            # 买入 (Buy): 如果 当前价格 <= 订单价格
            elif order.direction == Direction.LONG:
                if order.price >= buy_cross_price:
                    is_traded = True
                    # 为了模拟真实，如果 Bar 的 Open 价格也满足，可以用 Open 价成交，这里简化为限价成交
                    trade_price = order.price

            # 卖出 (Sell): 如果 当前价格 >= 订单价格
            elif order.direction == Direction.SHORT:
                if order.price <= sell_cross_price:
                    is_traded = True
                    trade_price = order.price

            # --- 成交处理 ---
            if is_traded:
                order.traded = order.volume
                order.status = Status.ALLTRADED
                order.time = current_time.strftime("%H:%M:%S")

                # 1. 生成成交单
                trade = TradeData(
                    gateway_name=self.gateway_name,
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=f"T{order.orderid}",
                    direction=order.direction,
                    offset=order.offset,
                    price=trade_price,
                    volume=order.volume,
                    datetime=current_time,
                )

                # 2. 更新账户
                self._update_account(trade)

                # 3. 推送事件
                self.on_event(EventType.ORDER, order)
                self.on_event(EventType.TRADE, trade)

                traded_orders.append(orderid)

        # 移除已成交订单
        for oid in traded_orders:
            self.active_orders.pop(oid)

    def _update_account(self, trade: TradeData):
        """更新本地简易账户"""
        pos = self.positions[trade.symbol]
        pos.symbol = trade.symbol

        cost = trade.price * trade.volume

        # 简单处理：买入扣钱加仓，卖出加钱减仓 (不考虑保证金/手续费)
        if trade.direction == Direction.LONG:
            self.balance -= cost

            # 更新持仓均价
            new_cost = pos.avg_price * pos.volume + cost
            pos.volume += trade.volume
            pos.avg_price = new_cost / pos.volume if pos.volume else 0

        else:  # SHORT / SELL
            self.balance += cost
            pos.volume -= trade.volume
            # 卖出通常不更新持仓均价，只结算盈亏，这里简化处理

    def query_account(self) -> float:
        """查询账户资金"""
        return self.balance

    def query_position(self) -> Dict[str, PositionData]:
        """查询持仓"""
        return self.positions

    def write_log(self, msg: str) -> None:
        self.on_event(EventType.LOG, msg)
