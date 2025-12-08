"""
src/simulation/mock_gateway.py
仿真网关：伪装成一个交易所，拦截策略的订单请求
Refactored: Integrated MatchingEngine AND AccountManager
"""

from datetime import datetime as dt

from src.core.event import Event, EventEngine, EventType
from src.core.interfaces import GatewayBase
from src.core.types import (
    OrderData,
    OrderRequest,
    Status,
    TickData,
)

# [新增] 导入账户管理器
from src.simulation.account import AccountManager
from src.simulation.matching import MatchingEngine


class MockGateway(GatewayBase):
    """
    仿真网关
    职责：
    1. 作为策略和撮合引擎之间的桥梁
    2. 将 OrderRequest 转换为 OrderData
    3. 将撮合引擎产生的 TradeData 推送给 EventEngine
    4. [新增] 维护本地账户资金状态
    """

    def __init__(
        self,
        event_engine: EventEngine,
        gateway_name: str = "BACKTEST",
        slippage: float = 0.0,
        initial_balance: float = 100000.0,  # [新增] 初始资金参数
    ):
        super().__init__(event_engine, gateway_name)

        # 撮合引擎：负责订单成交
        self.matcher = MatchingEngine(slippage=slippage)

        # [新增] 账户管理器：负责资金结算
        self.account = AccountManager(initial_balance=initial_balance)

    def connect(self):
        """模拟连接"""
        self.write_log("仿真网关连接成功")
        # 模拟连接后推送一次初始资金状态
        self.query_account()

    def subscribe(self, symbol: str):
        """模拟订阅"""
        self.write_log(f"订阅成功: {symbol}")

    def send_order(self, req: OrderRequest) -> str:
        """
        收到策略发单请求
        """
        orderid = req.create_order_id(self.gateway_name)

        # 1. 构建订单对象
        order = OrderData(
            gateway_name=self.gateway_name,
            symbol=req.symbol,
            exchange="MOCK",
            orderid=orderid,
            direction=req.direction,
            type=req.type,
            offset=req.offset,
            price=req.price,
            volume=req.volume,
            traded=0,
            status=Status.NOTTRADED,
            datetime=dt.now(),
        )

        # 2. 推送订单状态 (已受理)
        self.on_event(EventType.ORDER, order)
        self.write_log(
            f"收到订单: {req.direction.value} {req.type.value} {req.price}@{req.volume}"
        )

        # 3. 转交给撮合引擎
        self.matcher.add_order(order)

        return orderid

    def cancel_order(self, orderid: str, symbol: str):
        """撤单"""
        cancelled_order = self.matcher.cancel_order(orderid)

        if cancelled_order:
            cancelled_order.datetime = dt.now()
            self.on_event(EventType.ORDER, cancelled_order)
            self.write_log(f"订单已撤销: {orderid}")
        else:
            self.write_log(f"撤单失败，订单不存在或已成交: {orderid}")

    def process_tick(self, tick: TickData):
        """
        外部驱动 Tick
        """
        # 1. 推送行情给策略
        self.on_event(EventType.TICK, tick)

        # 2. 调用撮合引擎进行撮合
        trades = self.matcher.match(tick)

        # 3. 处理成交结果
        for trade in trades:
            # [新增] 账户结算：扣钱/加仓
            self.account.on_trade(trade)

            # 推送订单完成状态
            finished_order = OrderData(
                gateway_name=self.gateway_name,
                symbol=trade.symbol,
                orderid=trade.orderid,
                direction=trade.direction,
                status=Status.ALLTRADED,
                traded=trade.volume,
                price=0,
                volume=trade.volume,
            )
            self.on_event(EventType.ORDER, finished_order)

            # 推送成交
            self.on_event(EventType.TRADE, trade)
            self.write_log(f"订单成交: {trade.symbol} {trade.price}@{trade.volume}")

    def query_account(self):
        """
        [实现] 查询账户资金
        """
        # 简单打印当前余额
        # 在实盘中这里会构造 AccountData 并推送
        self.write_log(f"当前可用余额: {self.account.balance:,.2f}")

    def query_position(self):
        """
        [实现] 查询持仓
        """
        for symbol, pos in self.account.positions.items():
            if pos.volume != 0:
                self.write_log(f"持仓 {symbol}: {pos.volume}@{pos.avg_price:.2f}")

    def write_log(self, msg: str):
        event = Event(type=EventType.LOG, data=msg)
        self.event_engine.put(event)
