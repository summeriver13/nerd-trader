"""
src/simulation/crypto/account.py
加密货币账户实现：T+0，双向持仓，无印花税
"""

from src.core.base.account import BaseAccount, BasePosition
from src.core.types import Direction, TradeData


class CryptoAccount(BaseAccount):
    """
    加密货币/期货账户 (T+0)
    """

    def __init__(self, initial_balance: float = 100000.0):
        super().__init__(initial_balance, position_cls=BasePosition)

    def on_trade(self, trade: TradeData):
        self.trades.append(trade)
        pos = self.get_position(trade.symbol)

        # 1. 资金变动
        txn_amount = trade.price * trade.volume

        if trade.direction == Direction.LONG:
            self.balance -= txn_amount
            trade_vol = trade.volume
        else:
            self.balance += txn_amount
            trade_vol = -trade.volume

        # 2. 持仓与盈亏更新逻辑
        current_vol = pos.volume
        new_vol = current_vol + trade_vol

        # A. 加仓
        if (current_vol == 0) or (current_vol * trade_vol > 0):
            old_cost = abs(current_vol) * pos.avg_price
            new_cost = abs(trade_vol) * trade.price
            total_vol = abs(new_vol)

            if total_vol > 0:
                pos.avg_price = (old_cost + new_cost) / total_vol

        # B. 减仓/平仓
        elif abs(new_vol) < abs(current_vol):
            closed_vol = abs(trade_vol)
            if current_vol > 0:
                pnl = (trade.price - pos.avg_price) * closed_vol
            else:
                pnl = (pos.avg_price - trade.price) * closed_vol
            pos.realized_pnl += pnl

        # C. 反手
        else:
            closed_vol = abs(current_vol)
            if current_vol > 0:
                pnl = (trade.price - pos.avg_price) * closed_vol
            else:
                pnl = (pos.avg_price - trade.price) * closed_vol
            pos.realized_pnl += pnl
            pos.avg_price = trade.price

        # --- 【关键修复】 ---
        # 更新总持仓
        pos.volume = new_vol

        # T+0 市场：可用持仓必须时刻等于总持仓
        pos.available_volume = new_vol

        if pos.volume == 0:
            pos.avg_price = 0.0
