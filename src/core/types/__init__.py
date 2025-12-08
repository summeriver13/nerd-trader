"""
src/core/types/__init__.py
Refactored: Added 'type' field to OrderData
"""

from dataclasses import dataclass, field
from datetime import datetime as dt
from enum import StrEnum


# ... (枚举定义部分保持不变) ...
class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class Offset(StrEnum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    CLOSETODAY = "CLOSETODAY"
    NONE = "NONE"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class Status(StrEnum):
    SUBMITTING = "SUBMITTING"
    NOTTRADED = "NOTTRADED"
    PARTTRADED = "PARTTRADED"
    ALLTRADED = "ALLTRADED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# ... (BaseData, TickData, BarData, OrderRequest 保持不变) ...


@dataclass(slots=True)
class BaseData:
    gateway_name: str
    symbol: str
    exchange: str = ""


@dataclass(slots=True)
class TickData(BaseData):
    datetime: dt = field(default_factory=dt.now)
    last_price: float = 0.0
    volume: float = 0.0
    bid_price_1: float = 0.0
    bid_volume_1: float = 0.0
    ask_price_1: float = 0.0
    ask_volume_1: float = 0.0


@dataclass(slots=True)
class BarData(BaseData):
    datetime: dt | None = None
    interval: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    direction: Direction
    type: OrderType
    volume: float
    price: float = 0.0
    offset: Offset = Offset.NONE

    def create_order_id(self, gateway_name: str) -> str:
        return f"{gateway_name}_{self.symbol}_{dt.now().strftime('%H%M%S%f')}"


# --- [修改] 给 OrderData 增加 type 字段 ---
@dataclass(slots=True)
class OrderData(BaseData):
    """订单数据"""

    orderid: str = ""
    direction: Direction = Direction.LONG
    # [新增] 必须加上这个字段，撮合引擎才能判断是市价单还是限价单
    type: OrderType = OrderType.LIMIT
    offset: Offset = Offset.NONE
    price: float = 0.0
    volume: float = 0.0
    traded: float = 0.0
    status: Status = Status.SUBMITTING
    datetime: dt | None = None


@dataclass(slots=True)
class TradeData(BaseData):
    """成交数据"""

    orderid: str = ""
    tradeid: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.NONE
    price: float = 0.0
    volume: float = 0.0
    datetime: dt | None = None


@dataclass(slots=True)
class PositionData(BaseData):
    """持仓数据"""

    direction: Direction = Direction.LONG
    volume: float = 0.0
    frozen: float = 0.0
    price: float = 0.0
    pnl: float = 0.0
