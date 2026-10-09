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
from typing import override

from src.core.costs import SPOT_FEES, FeeSchedule
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
    """仿真网关（现货语义的 1x 现金账户：买入扣现、卖出加现，不建模保证金）。"""

    def __init__(
        self,
        event_engine: EventEngine,
        gateway_name: str = "BACKTEST",
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        fees: FeeSchedule = SPOT_FEES,
    ):
        super().__init__(event_engine, gateway_name)

        # 1. 内部状态管理 (简易版账户与订单系统)
        self.orders: dict[str, OrderData] = {}  # orderid -> OrderData
        self.active_orders: dict[str, OrderData] = {}  # orderid -> OrderData

        # 资金与持仓
        self.balance = initial_capital
        self.total_fee = 0.0  # 累计成本（手续费 + 滑点），便于对照研究侧口径
        self.fees = fees
        self.positions: dict[str, PositionData] = defaultdict(
            lambda: PositionData(symbol="")
        )
        self.last_price: dict[str, float] = {}  # symbol -> 最新价，供 mark-to-market 用
        self.subscribed: set[str] = set()  # 已订阅标的（回测里默认全订阅，仅作记录）
        self._order_seq = 0  # 订单号用计数器，不用墙钟时间，保证回测可复现
        self.last_market_time: datetime | None = None  # 最近一根行情的时间

    @override
    def connect(self) -> None:
        """连接"""
        self.write_log(f"仿真网关 [{self.gateway_name}] 启动成功")
        self.write_log(f"初始资金: {self.balance:,.2f}")
        self.write_log(
            f"费率: maker {self.fees.maker:.2%} / taker {self.fees.taker:.2%}"
            f" + 市价单滑点 {self.fees.slippage:.2%}"
        )

    @override
    def close(self) -> None:
        """断开（回测里只记录，真实网关在此释放连接与订阅）"""
        self.write_log(f"仿真网关 [{self.gateway_name}] 已断开")

    @override
    def subscribe(self, symbol: str) -> None:
        """订阅行情"""
        self.subscribed.add(symbol)

    @override
    def send_order(self, req: OrderRequest) -> str:
        """发送订单"""
        self._order_seq += 1
        orderid = f"{self.gateway_name}.{self._order_seq:06d}"
        # 订单时间用最近一根行情的时间，不用墙钟：回测的订单明细要可复现
        order_time = self.last_market_time or datetime.now()

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
            datetime=order_time,
            traded=0,
        )

        self.orders[orderid] = order
        self.active_orders[orderid] = order

        # 推送订单状态：未成交
        self.on_event(EventType.ORDER, order)

        return orderid

    @override
    def cancel_order(self, orderid: str) -> None:
        """撤单：标的不必传，从本地订单里查"""
        if orderid in self.active_orders:
            order = self.active_orders.pop(orderid)
            order.status = Status.CANCELLED
            order.time = (self.last_market_time or datetime.now()).strftime("%H:%M:%S")
            self.on_event(EventType.ORDER, order)
            self.write_log(f"订单已撤销: {orderid} ({order.symbol})")
        else:
            self.write_log(f"撤单失败: 订单 {orderid} 不存在或已完结")

    def process_tick(self, tick: TickData):
        """处理 Tick 行情"""
        # 1. 更新最新价
        self.last_price[tick.symbol] = tick.last_price
        self.last_market_time = tick.datetime

        # 2. 先撮合：与 bar 路径同序，本 tick 下的单留给下一个 tick 成交，避免前视
        self._match_orders(tick)

        # 3. 再推送给策略
        self.on_event(EventType.TICK, tick)

    def process_bar(self, bar: BarData):
        """处理 Bar 行情"""
        # 1. 记录最新价，供 mark-to-market 估值
        self.last_price[bar.symbol] = bar.close_price
        self.last_market_time = bar.datetime

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

    def _match_orders(self, data: TickData | BarData):
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
            crosses = False  # 吃价成交（taker）还是被动挂单成交（maker）
            trade_price = 0.0

            # 市价单：无条件按当前可得价格成交
            if order.type == OrderType.MARKET:
                is_traded = True
                crosses = True
                trade_price = market_price

            # 限价买单：挂在市场上方 → 立刻吃对手价成交；挂在下方 → 等 low 触及
            elif order.direction == Direction.LONG:
                if order.price >= market_price:
                    is_traded = True
                    crosses = True
                    trade_price = market_price
                elif order.price >= buy_cross_price:
                    is_traded = True
                    trade_price = order.price

            # 限价卖单：挂在市场下方 → 立刻吃对手价成交；挂在上方 → 等 high 触及
            elif order.direction == Direction.SHORT:
                if order.price <= market_price:
                    is_traded = True
                    crosses = True
                    trade_price = market_price
                elif order.price <= sell_cross_price:
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

                # 2. 更新账户：吃价成交按 taker + 滑点，被动成交按 maker
                self._update_account(trade, self.fees.net(crosses=crosses))

                # 3. 推送事件
                self.on_event(EventType.ORDER, order)
                self.on_event(EventType.TRADE, trade)

                traded_orders.append(orderid)

        # 移除已成交订单
        for oid in traded_orders:
            self.active_orders.pop(oid)

    def _update_account(self, trade: TradeData, rate: float):
        """更新本地简易账户，按 ``rate`` 收取单边成本（手续费 + 滑点）。"""
        pos = self.positions[trade.symbol]
        pos.symbol = trade.symbol

        notional = trade.price * trade.volume
        fee = notional * rate
        self.total_fee += fee

        # 现金账户口径：买入扣钱加仓，卖出加钱减仓（不建模保证金与杠杆）
        if trade.direction == Direction.LONG:
            self.balance -= notional + fee

            # 更新持仓均价
            new_cost = pos.avg_price * pos.volume + notional
            pos.volume += trade.volume
            pos.avg_price = new_cost / pos.volume if pos.volume else 0

        else:  # SHORT / SELL
            self.balance += notional - fee
            pos.volume -= trade.volume
            # 卖出通常不更新持仓均价，只结算盈亏，这里简化处理

    @override
    def query_account(self) -> float:
        """查询账户资金"""
        return self.balance

    @override
    def query_position(self) -> dict[str, PositionData]:
        """查询持仓"""
        return self.positions

    def write_log(self, msg: str) -> None:
        self.on_event(EventType.LOG, msg)
