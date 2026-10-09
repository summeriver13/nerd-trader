"""
src/core/interfaces/__init__.py
定义 Gateway 与 Strategy 的标准接口

- StrategyContext：策略看到的外部环境（依赖倒置：策略不 import 引擎）
- GatewayBase：交易网关抽象基类
- StrategyBase：策略抽象基类（事件驱动）
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..event import Event, EventEngine, EventType
from ..types import (
    BarData,
    Direction,
    Offset,
    OrderData,
    OrderRequest,
    OrderType,
    PositionData,
    TickData,
    TradeData,
)


# --- 策略上下文协议 ---
# StrategyContext 定义了策略运行所需的外部环境能力 (依赖倒置原则)
class StrategyContext(Protocol):
    def send_order(self, req: OrderRequest) -> str: ...

    def cancel_order(self, orderid: str) -> None: ...

    def write_log(self, msg: str, strategy_name: str) -> None: ...

    def query_account(self) -> float: ...

    def query_position(self) -> dict[str, PositionData]: ...


class GatewayBase(ABC):
    """
    交易网关抽象基类
    """

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        self.event_engine = event_engine
        self.gateway_name = gateway_name

    def on_event(self, event_type: str | EventType, data: Any = None) -> None:
        """辅助函数：打包并推送事件"""
        event = Event(type=event_type, data=data)
        self.event_engine.put(event)

    @abstractmethod
    def connect(self) -> None:
        """连接交易所"""
        pass

    @abstractmethod
    def close(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def subscribe(self, symbol: str) -> None:
        """订阅行情"""
        pass

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """发送订单 (返回本地订单号 orderid)"""
        pass

    @abstractmethod
    def cancel_order(self, orderid: str) -> None:
        """撤单（标的不必传：网关自己从订单里查）"""
        pass

    @abstractmethod
    def query_account(self) -> float:
        """查询账户资金"""
        pass

    @abstractmethod
    def query_position(self) -> dict[str, PositionData]:
        """查询持仓"""
        pass


class StrategyBase(ABC):
    """
    策略抽象基类
    支持：CTA (单标的择时) / Alpha (多标的选股 - 预留) / HFT (Tick级)
    """

    # 策略声明的参数名：上层据此做必填校验，缺键直接报错而不是静默取默认值
    parameters: list[str] = []

    def __init__(
        self, context: StrategyContext, strategy_name: str, settings: dict[str, Any]
    ) -> None:
        self.context = context  # 策略运行的上下文环境
        self.strategy_name = strategy_name
        self.settings = settings

    # --- 生命周期 ---

    @abstractmethod
    def on_init(self) -> None:
        """初始化 (加载历史数据等)"""
        pass

    @abstractmethod
    def on_start(self) -> None:
        """启动"""
        pass

    @abstractmethod
    def on_stop(self) -> None:
        """停止"""
        pass

    # --- 行情回调 ---
    # 按数据形态实现其中之一即可（引擎在 run() 时校验至少实现了一个）；
    # Tick 回放尚未接入，K 线策略不必再实现空的 on_tick。

    def on_tick(self, tick: TickData) -> None:
        """Tick 行情推送 (可选实现)"""
        pass

    def on_bar(self, bar: BarData) -> None:
        """单标的 K 线行情推送 (可选实现)"""
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """
        [预留] 全市场 K 线切片推送
        适用于: 截面选股策略 (Alpha)
        当 Engine 支持时间对齐时，会调用此接口
        """
        pass

    # --- 交易回调 ---

    def on_order(self, order: OrderData) -> None:
        """订单状态更新推送"""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """成交推送"""
        pass

    # --- 核心交易功能 (语法糖) ---

    def _send(
        self,
        symbol: str,
        price: float,
        volume: float,
        *,
        direction: Direction,
        offset: Offset,
        order_type: OrderType,
    ) -> str:
        """四个下单方法共用的发单实现"""
        req = OrderRequest(
            symbol=symbol,
            direction=direction,
            offset=offset,
            type=order_type,
            price=price,
            volume=volume,
        )
        return self.context.send_order(req)

    def buy(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """买入开仓 (股票/期货做多)"""
        return self._send(
            symbol,
            price,
            volume,
            direction=Direction.LONG,
            offset=Offset.OPEN,
            order_type=order_type,
        )

    def sell(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """卖出平仓 (股票卖出/期货平多)"""
        return self._send(
            symbol,
            price,
            volume,
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            order_type=order_type,
        )

    def short(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """[期货专用] 卖出开仓 (做空)"""
        return self._send(
            symbol,
            price,
            volume,
            direction=Direction.SHORT,
            offset=Offset.OPEN,
            order_type=order_type,
        )

    def cover(
        self,
        symbol: str,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> str:
        """[期货专用] 买入平仓 (平空)"""
        return self._send(
            symbol,
            price,
            volume,
            direction=Direction.LONG,
            offset=Offset.CLOSE,
            order_type=order_type,
        )

    def cancel_order(self, orderid: str) -> None:
        """撤单"""
        self.context.cancel_order(orderid)

    def write_log(self, msg: str) -> None:
        """记录日志"""
        self.context.write_log(msg, self.strategy_name)
