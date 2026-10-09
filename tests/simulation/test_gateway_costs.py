"""仿真网关的成本口径：与研究侧共用同一份费率（src/core/costs.py）。"""

from datetime import UTC, datetime

import pytest

from src.core.costs import (
    MARKET_SLIPPAGE,
    PERP_MAKER_FEE,
    PERP_TAKER_FEE,
    SPOT_MAKER_FEE,
    SPOT_TAKER_FEE,
    FeeSchedule,
)
from src.core.event import EventEngine
from src.core.types import (
    BarData,
    Direction,
    Offset,
    OrderRequest,
    OrderType,
    TickData,
)
from src.research.carry import ROUND_TRIP_COST
from src.research.domains.crypto import CryptoDomain
from src.research.execution import MakerPolicy
from src.simulation.mock_gateway import MockGateway

SYMBOL = "BTC-USDT"
BAR_TIME = datetime(2026, 1, 1, tzinfo=UTC)
INITIAL_CAPITAL = 100_000.0


def _bar(open_price: float, low: float, high: float) -> BarData:
    return BarData(
        gateway_name="TEST",
        symbol=SYMBOL,
        exchange="MOCK",
        datetime=BAR_TIME,
        open_price=open_price,
        high_price=high,
        low_price=low,
        close_price=open_price,
    )


def _tick(price: float) -> TickData:
    return TickData(
        gateway_name="TEST",
        symbol=SYMBOL,
        exchange="MOCK",
        datetime=BAR_TIME,
        last_price=price,
        bid_price_1=price,
        ask_price_1=price,
    )


def _order(
    order_type: OrderType,
    price: float,
    volume: float,
    direction: Direction = Direction.LONG,
) -> OrderRequest:
    return OrderRequest(
        symbol=SYMBOL,
        direction=direction,
        offset=Offset.OPEN,
        type=order_type,
        price=price,
        volume=volume,
    )


def test_market_order_pays_taker_fee_and_slippage():
    """市价单以本根开盘价成交，成本 = 现货 taker 手续费 + 滑点。"""
    gateway = MockGateway(EventEngine(), initial_capital=INITIAL_CAPITAL)

    gateway.send_order(_order(OrderType.MARKET, price=0.0, volume=2.0))
    gateway.process_bar(_bar(open_price=100.0, low=100.0, high=100.0))

    expected_fee = 200.0 * (SPOT_TAKER_FEE + MARKET_SLIPPAGE)
    assert gateway.balance == pytest.approx(INITIAL_CAPITAL - 200.0 - expected_fee)
    assert gateway.total_fee == pytest.approx(expected_fee)


def test_resting_limit_order_pays_maker_fee():
    """限价单挂在市价下方、被本根 low 触及才成交 → 被动成交，只收 maker 费。"""
    gateway = MockGateway(EventEngine(), initial_capital=INITIAL_CAPITAL)

    gateway.send_order(_order(OrderType.LIMIT, price=100.0, volume=2.0))
    gateway.process_bar(_bar(open_price=101.0, low=99.0, high=101.0))

    expected_fee = 200.0 * SPOT_MAKER_FEE
    assert gateway.balance == pytest.approx(INITIAL_CAPITAL - 200.0 - expected_fee)
    assert gateway.total_fee == pytest.approx(expected_fee)


def test_crossing_limit_order_fills_at_market_and_pays_taker():
    """挂在市场上方的买单立刻吃对手价成交：成交价是市价而不是限价，费率按 taker。"""
    gateway = MockGateway(EventEngine(), initial_capital=INITIAL_CAPITAL)

    gateway.send_order(_order(OrderType.LIMIT, price=102.0, volume=2.0))
    gateway.process_bar(_bar(open_price=101.0, low=99.0, high=102.0))

    assert gateway.positions[SYMBOL].avg_price == pytest.approx(101.0)
    expected_fee = 202.0 * (SPOT_TAKER_FEE + MARKET_SLIPPAGE)
    assert gateway.total_fee == pytest.approx(expected_fee)


def test_tick_order_does_not_fill_in_same_tick():
    """tick 路径先撮合后推送：本 tick 下的单不能在同一个 tick 成交（避免前视）。"""
    gateway = MockGateway(EventEngine(), initial_capital=INITIAL_CAPITAL)

    gateway.process_tick(_tick(100.0))
    gateway.send_order(_order(OrderType.MARKET, price=0.0, volume=1.0))
    assert gateway.positions[SYMBOL].volume == 0.0

    gateway.process_tick(_tick(101.0))
    assert gateway.positions[SYMBOL].volume == 1.0


def test_round_trip_costs_two_sides():
    """一次进出按两个单边计费：权益减少额 = 2 × 单边成本。"""
    gateway = MockGateway(EventEngine(), initial_capital=INITIAL_CAPITAL)

    gateway.send_order(_order(OrderType.MARKET, price=0.0, volume=2.0))
    gateway.process_bar(_bar(open_price=100.0, low=100.0, high=100.0))
    gateway.send_order(
        _order(OrderType.MARKET, price=0.0, volume=2.0, direction=Direction.SHORT)
    )
    gateway.process_bar(_bar(open_price=100.0, low=100.0, high=100.0))

    one_side = 200.0 * (SPOT_TAKER_FEE + MARKET_SLIPPAGE)
    assert gateway.balance == pytest.approx(INITIAL_CAPITAL - 2 * one_side)
    assert gateway.total_fee == pytest.approx(2 * one_side)


def test_fee_rates_have_single_source():
    """研究侧三处与仿真侧取的是同一份费率，任何一处漂移都会在这里失败。"""
    assert MakerPolicy().fees.taker == PERP_TAKER_FEE
    assert MakerPolicy().fees.maker == PERP_MAKER_FEE
    assert MakerPolicy().taker_net_cost == PERP_TAKER_FEE + MARKET_SLIPPAGE
    assert MakerPolicy().maker_net_cost == PERP_MAKER_FEE - MakerPolicy().offset
    assert CryptoDomain().fee_rate == PERP_TAKER_FEE
    assert CryptoDomain().slippage == MARKET_SLIPPAGE
    assert MockGateway(EventEngine()).fees.taker == SPOT_TAKER_FEE
    assert MockGateway(EventEngine()).fees.maker == SPOT_MAKER_FEE
    assert ROUND_TRIP_COST == 2 * (SPOT_TAKER_FEE + PERP_TAKER_FEE)


def test_crosses_decides_fee_tier():
    """吃价判定是费率的唯一入口：吃价 → taker + 滑点，被动 → maker − 价差改善。"""
    fees = FeeSchedule(maker=0.0002, taker=0.0005, slippage=0.0005)
    assert fees.net(crosses=True) == pytest.approx(0.0010)
    assert fees.net(crosses=False) == pytest.approx(0.0002)
    assert fees.net(crosses=False, offset=0.0001) == pytest.approx(0.0001)
