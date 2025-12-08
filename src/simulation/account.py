"""
src/simulation/account.py
虚拟账户模块：管理资金、持仓和盈亏计算
Refactored: Robust PnL calculation & Position Management
"""

from collections import defaultdict
from dataclasses import dataclass

from src.core.types import Direction, TradeData


@dataclass
class PositionInfo:
    """
    单标的持仓信息
    """

    symbol: str
    volume: float = 0.0  # 持仓数量 (+为多, -为空)
    avg_price: float = 0.0  # 持仓均价 (Weighted Average Price)
    realized_pnl: float = 0.0  # 该标的累计已实现盈亏

    @property
    def market_value(self) -> float:
        """持仓市值 (成本价值)"""
        return self.volume * self.avg_price


class AccountManager:
    """
    账户管理器
    职责：
    1. 资金结算 (Balance)
    2. 持仓维护 (Positions)
    3. 盈亏计算 (PnL)
    """

    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance  # 可用资金 (Cash)
        self.frozen = 0.0  # 冻结资金 (挂单未成交部分，暂留接口)

        # 持仓字典: symbol -> PositionInfo
        self.positions: dict[str, PositionInfo] = defaultdict(
            lambda: PositionInfo(symbol="")
        )

        # 交易记录流水
        self.trades: list[TradeData] = []

    def on_trade(self, trade: TradeData):
        """
        [核心] 收到成交回报，更新账户状态
        处理逻辑：
        1. 资金变更：买入扣钱，卖出加钱 (现货逻辑 / 保证金逻辑简化版)
        2. 持仓变更：加权平均价计算、平仓盈亏结算
        """
        self.trades.append(trade)
        symbol = trade.symbol

        # 修正：defaultdict 的 lambda 无法获取 key，需手动赋值 symbol
        if symbol not in self.positions:
            self.positions[symbol].symbol = symbol

        pos = self.positions[symbol]

        # 1. 计算交易金额
        # 简化模型：全额交易，不涉及保证金杠杆
        txn_amount = trade.price * trade.volume

        if trade.direction == Direction.LONG:
            self.balance -= txn_amount  # 买入花钱
            trade_vol = trade.volume  # 交易带来的持仓变化量
        else:
            self.balance += txn_amount  # 卖出得钱
            trade_vol = -trade.volume

        # 2. 更新持仓逻辑 (最复杂的部分)
        # 我们需要判断这笔交易是：A. 加仓(Opening)  B. 减仓(Closing)  C. 反手(Flipping)

        current_vol = pos.volume
        new_vol = current_vol + trade_vol

        # 情况 A: 加仓 (方向相同，或者从0开始)
        if (
            (current_vol == 0)
            or (current_vol > 0 and trade_vol > 0)
            or (current_vol < 0 and trade_vol < 0)
        ):
            # 加权平均价公式: (旧持仓市值 + 新交易市值) / 新总持仓量
            old_cost = abs(current_vol) * pos.avg_price
            new_cost = abs(trade_vol) * trade.price
            total_vol = abs(new_vol)

            if total_vol > 0:
                pos.avg_price = (old_cost + new_cost) / total_vol

            # 加仓不产生已实现盈亏

        # 情况 B: 减仓/平仓 (方向相反，且没反手)
        elif abs(new_vol) < abs(current_vol):
            # 结算盈亏 (Realized PnL)
            # 盈亏 = (卖价 - 买价) * 数量
            closed_vol = abs(trade_vol)

            if current_vol > 0:
                # 多头平仓: 卖出价 - 成本价
                pnl = (trade.price - pos.avg_price) * closed_vol
            else:
                # 空头平仓: 成本价 - 买入价
                pnl = (pos.avg_price - trade.price) * closed_vol

            pos.realized_pnl += pnl
            # 减仓时，持仓均价(成本)不变！只有加仓才变。

        # 情况 C: 反手 (例如 +10 -> -5)
        # 实际上是：先平掉 +10，再开 -5
        else:
            # 1. 先平掉旧持仓的所有数量
            closed_vol = abs(current_vol)
            if current_vol > 0:
                pnl = (trade.price - pos.avg_price) * closed_vol
            else:
                pnl = (pos.avg_price - trade.price) * closed_vol
            pos.realized_pnl += pnl

            # 2. 剩下的量作为新开仓
            # 新持仓的成本价就是当前成交价
            pos.avg_price = trade.price

        # 应用新数量
        pos.volume = new_vol

        # 如果持仓归零，重置均价 (美观起见)
        if pos.volume == 0:
            pos.avg_price = 0.0

    def calculate_equity(self, current_prices: dict[str, float]) -> float:
        """
        计算账户总权益 (Total Equity)
        权益 = 现金余额 + 持仓市值(按市价折算)

        :param current_prices: {symbol: current_price} 外部传入的最新行情
        """
        equity = self.balance

        for symbol, pos in self.positions.items():
            if pos.volume == 0:
                continue

            # 如果没有最新价，这就很尴尬，暂时用成本价兜底 (回测结束时可能发生)
            market_price = current_prices.get(symbol, pos.avg_price)

            # 持仓市值 = 数量 * 市价
            # 多头(+): 市值增加权益
            # 空头(-): 市值是负债，减少权益 (但前面卖出时balance已经加了钱，所以这里减去市值是正确的逻辑)
            # 举例：借货卖出得100元(balance)，现价80，赎回需80。权益 = 100 + (-1 * 80) = 20。
            equity += pos.volume * market_price

        return equity
