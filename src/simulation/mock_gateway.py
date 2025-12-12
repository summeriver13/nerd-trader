from datetime import datetime as dt
from typing import Union

from src.core.event import Event, EventEngine, EventType
from src.core.interfaces import GatewayBase
from src.core.types import (
    BarData,  # [新增] 需要支持 BarData
    OrderData,
    OrderRequest,
    Status,
    TickData,
)

# [修改] 不再导入具体的实现类，而是导入抽象基类用于类型提示
from src.simulation.base.account import BaseAccount
from src.simulation.base.matching import BaseMatchingEngine


class MockGateway(GatewayBase):
    """
    仿真网关 (重构版)
    核心变化：不再自己创建 Engine/Account，而是通过外部注入。
    """

    def __init__(
        self,
        event_engine: EventEngine,
        gateway_name: str = "BACKTEST",
        # [新增] 依赖注入参数
        matching_engine: BaseMatchingEngine | None = None,
        account_manager: BaseAccount | None = None,
    ):
        super().__init__(event_engine, gateway_name)

        # [校验] 必须注入具体的实现
        if not matching_engine or not account_manager:
            raise ValueError(
                "MockGateway 初始化失败：必须注入具体的 matching_engine 和 account_manager"
            )

        # 接收外部注入的组件
        self.matcher = matching_engine
        self.account = account_manager

    def connect(self):
        """模拟连接"""
        self.write_log(f"仿真网关 [{self.gateway_name}] 连接成功")
        self.write_log(f"加载撮合引擎: {type(self.matcher).__name__}")
        self.write_log(f"加载账户系统: {type(self.account).__name__}")

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
        # self.write_log(f"收到订单: {req.direction.value} {req.type.value} {req.price}@{req.volume}")

        # 3. 转交给撮合引擎 (多态调用)
        self.matcher.add_order(order)

        return orderid

    def cancel_order(self, orderid: str, symbol: str):
        """撤单"""
        # 调用基类的 cancel_order 接口
        cancelled_order = self.matcher.cancel_order(orderid)

        if cancelled_order:
            cancelled_order.datetime = dt.now()
            self.on_event(EventType.ORDER, cancelled_order)
            self.write_log(f"订单已撤销: {orderid}")
        else:
            self.write_log(f"撤单失败，订单不存在或已成交: {orderid}")

    def process_tick(self, tick: TickData):
        """处理 Tick 数据"""
        # 1. 推送行情给策略
        self.on_event(EventType.TICK, tick)
        # 2. 触发撮合
        self._process_matching(tick)

    def process_bar(self, bar: BarData):
        """
        [新增] 处理 Bar 数据
        为了支持 '尾盘抓强' 策略，引擎可能会推送 BarData
        """
        # 1. 推送行情给策略 (如果策略订阅的是 Bar 事件，需在 EventEngine 注册 BAR 类型)
        # 这里为了简化，有些框架会统一用 TICK 事件，或者增加 BAR 事件
        # 假设我们增加了 EventType.BAR
        # self.on_event(EventType.BAR, bar)

        # 暂时用 Tick 兼容，或者策略自行处理 on_bar
        # 在这里我们主要关注撮合
        self._process_matching(bar)

    def _process_matching(self, data: Union[TickData, BarData]):
        """
        [内部方法] 统一撮合与结算逻辑
        """
        # 1. 调用撮合引擎 (多态：具体是 Stock 还是 Crypto 由注入的 matcher 决定)
        trades = self.matcher.match(data)

        # 2. 处理成交结果
        for trade in trades:
            # 账户结算 (多态：具体是 T+1 还是 T+0 由注入的 account 决定)
            self.account.on_trade(trade)

            # 推送订单完成状态
            finished_order = OrderData(
                gateway_name=self.gateway_name,
                symbol=trade.symbol,
                orderid=trade.orderid,
                direction=trade.direction,
                status=Status.ALLTRADED,
                traded=trade.volume,
                price=trade.price,  # 成交均价
                volume=trade.volume,
            )
            self.on_event(EventType.ORDER, finished_order)

            # 推送成交
            self.on_event(EventType.TRADE, trade)
            self.write_log(f"订单成交: {trade.symbol} {trade.price}@{trade.volume}")

    def query_account(self):
        """查询账户资金"""
        # 调用 BaseAccount 的 balance 属性
        self.write_log(f"当前可用余额: {self.account.balance:,.2f}")

    def query_position(self):
        """查询持仓"""
        # 调用 BaseAccount 的 positions 属性
        for symbol, pos in self.account.positions.items():
            if pos.volume != 0:
                self.write_log(f"持仓 {symbol}: {pos.volume}@{pos.avg_price:.2f}")

    def write_log(self, msg: str):
        event = Event(type=EventType.LOG, data=msg)
        self.event_engine.put(event)
