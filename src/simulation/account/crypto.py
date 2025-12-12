"""
src/simulation/account/crypto.py
加密货币账户实现：
1. T+0 (随买随卖)
2. 支持双向持仓 (正多负空)
3. 现金结算模型
"""

import math

from src.core.types import Direction, TradeData
from src.simulation.base.account import BaseAccount, BasePosition


class CryptoAccount(BaseAccount):
    """
    加密货币账户 (T+0, 双向)
    """

    def __init__(
        self, initial_balance: float = 100000.0, commission_rate: float = 0.0005
    ):
        super().__init__(initial_balance, position_cls=BasePosition)
        self.commission_rate = commission_rate

    def on_trade(self, trade: TradeData):
        self.trades.append(trade)
        pos = self.get_position(trade.symbol)

        # 1. 资金结算
        txn_value = trade.price * trade.volume
        commission = txn_value * self.commission_rate

        if trade.direction == Direction.LONG:
            # 买入：现金减少 (支付货款 + 手续费)
            self.balance -= txn_value + commission
            trade_vol = trade.volume
        else:
            # 卖出：现金增加 (收回货款 - 手续费)
            self.balance += txn_value - commission
            trade_vol = -trade.volume  # 卖出为负增量

        # 2. 持仓更新逻辑
        current_vol = pos.volume
        new_vol = current_vol + trade_vol

        # 判断是【加仓】还是【减仓/反手】
        is_opening = (current_vol == 0) or (current_vol * trade_vol > 0)

        if is_opening:
            # === 加仓 ===
            old_cost = abs(current_vol) * pos.avg_price
            new_cost = abs(trade_vol) * trade.price
            total_vol = abs(new_vol)

            if total_vol > 0:
                pos.avg_price = (old_cost + new_cost) / total_vol

        else:
            # === 减仓/反手 ===
            if current_vol * new_vol < 0:
                # 反手 (符号翻转)
                closed_vol = abs(current_vol)
                pos.avg_price = trade.price  # 剩余部分的成本重置为当前价
            else:
                # 纯减仓
                closed_vol = abs(trade_vol)

            # 计算已实现盈亏
            if current_vol > 0:
                # 多头卖平: (卖价 - 成本) * 量
                pnl = (trade.price - pos.avg_price) * closed_vol
            else:
                # 空头买平: (成本 - 买价) * 量
                pnl = (pos.avg_price - trade.price) * closed_vol

            # 此时 IDE 应该能识别 realized_pnl 了
            pos.realized_pnl += pnl

        # 3. 更新持仓 (精度控制)
        if math.isclose(new_vol, 0, abs_tol=1e-7):
            pos.volume = 0.0
            pos.available_volume = 0.0
            pos.avg_price = 0.0
        else:
            pos.volume = new_vol
            pos.available_volume = new_vol  # T+0 实时可用
