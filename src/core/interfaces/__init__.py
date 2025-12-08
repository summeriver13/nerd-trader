"""
src/core/interfaces/__init__.py
定义 Gateway 和 Strategy 的标准接口
Refactored: 使用 StrategyContext 替代 IEngine
"""

from abc import ABC, abstractmethod
from typing import Any, List, Protocol, runtime_checkable

from ..event import Event, EventEngine, EventType
from ..types import (
    BarData,
    Direction,
    Offset,
    OrderData,
    OrderRequest,
    OrderType,
    TickData,
    TradeData,
)


# --- [重命名] 定义策略上下文协议 ---
# StrategyContext 定义了策略运行所需的外部环境能力
@runtime_checkable
class StrategyContext(Protocol):
    def send_order(self, req: OrderRequest, gateway_name: str) -> str: ...

    def cancel_order(self, orderid: str, gateway_name: str): ...

    def write_log(self, msg: str, strategy_name: str): ...

    def query_account(self): ...


class GatewayBase(ABC):
    """
    交易网关抽象基类
    """

    def __init__(self, event_engine: EventEngine, gateway_name: str):
        self.event_engine = event_engine
        self.gateway_name = gateway_name

    def on_event(self, type: str | EventType, data: Any = None):
        """辅助函数：打包并推送事件"""
        event = Event(type=type, data=data)
        self.event_engine.put(event)

    @abstractmethod
    def connect(self):
        """连接交易所"""
        pass

    @abstractmethod
    def subscribe(self, symbol: str):
        """订阅行情"""
        pass

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """发送订单 (返回本地订单号 orderid)"""
        pass

    @abstractmethod
    def cancel_order(self, orderid: str, symbol: str):
        """撤单"""
        pass

    @abstractmethod
    def query_account(self):
        """查询账户资金"""
        pass

    @abstractmethod
    def query_position(self):
        """查询持仓"""
        pass


class StrategyBase(ABC):
    """
    策略抽象基类
    """

    # [变更] engine -> context
    def __init__(self, context: StrategyContext, strategy_name: str, settings: dict):
        self.context = context  # 策略运行的上下文环境
        self.strategy_name = strategy_name
        self.settings = settings

        # 定义需要持久化保存的变量名列表 (如: ["ma_window", "pos"])
        self.variables: List[str] = []

    # --- 生命周期 ---

    @abstractmethod
    def on_init(self):
        """初始化 (加载历史数据等)"""
        pass

    @abstractmethod
    def on_start(self):
        """启动"""
        pass

    @abstractmethod
    def on_stop(self):
        """停止"""
        pass

    # --- 行情回调 ---

    @abstractmethod
    def on_tick(self, tick: TickData):
        """Tick行情推送 (必须实现)"""
        pass

    def on_bar(self, bar: BarData):
        """K线行情推送 (可选实现)"""
        pass

    # --- 交易回调 ---

    def on_order(self, order: OrderData):
        """订单状态更新推送"""
        pass

    def on_trade(self, trade: TradeData):
        """成交推送"""
        pass

    # --- 核心交易功能 (语法糖) ---

    def buy(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """
        买入开仓
        :return: 本地订单号 (orderid)
        """
        req = OrderRequest(
            symbol=symbol,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            type=order_type,
            price=price,
            volume=volume,
        )
        # [变更] 使用 self.context
        return self.context.send_order(req, self.strategy_name)

    def sell(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """
        卖出平仓
        """
        req = OrderRequest(
            symbol=symbol,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            type=order_type,
            price=price,
            volume=volume,
        )
        return self.context.send_order(req, self.strategy_name)

    def cancel_order(self, orderid: str):
        """
        撤单
        """
        self.context.cancel_order(orderid, self.strategy_name)

    def write_log(self, msg: str):
        """
        记录日志
        """
        self.context.write_log(msg, self.strategy_name)
