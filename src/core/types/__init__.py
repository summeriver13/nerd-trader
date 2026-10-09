"""
src/core/types/__init__.py
定义系统基础数据类型 (Enums, Dataclasses)
Refactored: 增加 turnover, open_interest 字段支持，完善 Enum 定义
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# --- 枚举常量 ---


class Direction(Enum):
    """交易方向"""

    LONG = "多"
    SHORT = "空"
    NET = "净"


class Offset(Enum):
    """开平仓"""

    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSETODAY = "平今"
    CLOSEYESTERDAY = "平昨"


class Status(Enum):
    """订单状态"""

    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"


class OrderType(Enum):
    """订单类型"""

    LIMIT = "限价"
    MARKET = "市价"
    STOP = "止损"
    FAK = "FAK"  # FillAndKill
    FOK = "FOK"  # FillOrKill


class Exchange(Enum):
    """交易所 (示例)"""

    SSE = "SSE"  # 上交所
    SZSE = "SZSE"  # 深交所
    CFFEX = "CFFEX"  # 中金所
    SHFE = "SHFE"  # 上期所
    DCE = "DCE"  # 大商所
    CZCE = "CZCE"  # 郑商所
    INE = "INE"  # 能源中心
    BINANCE = "BINANCE"
    OKX = "OKX"
    MOCK = "MOCK"  # 模拟盘


class Interval(Enum):
    """K线周期"""

    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"
    TICK = "tick"


# --- 基础数据类 ---


@dataclass
class BaseData:
    """基础数据类"""

    gateway_name: str
    symbol: str
    exchange: str | Exchange
    datetime: datetime


@dataclass
class TickData(BaseData):
    """Tick 数据"""

    name: str = ""
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    last_price: float = 0
    last_volume: float = 0
    limit_up: float = 0
    limit_down: float = 0

    # 盘口深度 (简版只给一档，需要的可以自行扩充)
    bid_price_1: float = 0
    bid_volume_1: float = 0
    ask_price_1: float = 0
    ask_volume_1: float = 0


@dataclass
class BarData(BaseData):
    """K线数据"""

    interval: str | Interval = Interval.MINUTE
    volume: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0

    # [新增] 修复报错的关键字段
    turnover: float = 0  # 成交额
    open_interest: float = 0  # 持仓量


@dataclass
class OrderData(BaseData):
    """订单数据 (回报)"""

    orderid: str = ""
    type: OrderType = OrderType.LIMIT
    direction: Direction = Direction.LONG
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    time: str = ""


@dataclass
class TradeData(BaseData):
    """成交数据 (回报)"""

    orderid: str = ""
    tradeid: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    time: str = ""


@dataclass
class PositionData:
    """持仓数据（网关维护，策略与引擎通过 query_position 读）"""

    symbol: str
    volume: float = 0
    avg_price: float = 0.0


@dataclass
class OrderRequest:
    """发单请求"""

    symbol: str
    direction: Direction
    offset: Offset
    type: OrderType
    volume: float
    price: float = 0
    exchange: str | Exchange = Exchange.MOCK
