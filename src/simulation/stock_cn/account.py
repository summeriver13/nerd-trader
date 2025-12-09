"""
src/simulation/stock/account.py
A股账户实现：T+1，单向做多(默认)，含税费
"""

from src.core.base.account import BaseAccount, BasePosition
from src.core.types import Direction, TradeData


class StockAccount(BaseAccount):
    """
    A股账户
    特点：
    1. T+1: 今日买入，明日才进 available_volume
    2. 费用: 买卖佣金 + 卖出印花税
    3. 只能做多 (普通账户)
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        commission: float = 0.00025,  # 万2.5
        stamp_duty: float = 0.0005,  # 千0.5 (2023新规)
    ):
        # 修改点：直接使用通用的 BasePosition
        super().__init__(initial_balance, position_cls=BasePosition)

        self.commission = commission
        self.stamp_duty = stamp_duty

    def on_trade(self, trade: TradeData):
        """
        处理A股成交：计算税费 + 维护T+1状态
        """
        self.trades.append(trade)
        # 基类现在已经有 available_volume 字段，IDE 不会报错了
        pos = self.get_position(trade.symbol)

        txn_amount = trade.price * trade.volume

        # 1. 买入逻辑
        if trade.direction == Direction.LONG:
            # 费用：仅佣金
            fee = txn_amount * self.commission
            cost = txn_amount + fee

            # 扣款
            self.balance -= cost

            # 更新成本价 (将佣金摊入成本，更符合会计准则)
            old_total_cost = pos.volume * pos.avg_price
            new_total_cost = old_total_cost + cost  # 含佣金成本

            pos.volume += trade.volume
            if pos.volume > 0:
                pos.avg_price = new_total_cost / pos.volume

            # 【关键】T+1逻辑：买入时只加 volume，不加 available_volume

        # 2. 卖出逻辑
        elif trade.direction == Direction.SHORT:
            # A股普通账户的 SHORT 即为卖出平仓

            # 检查可用持仓 (双重保险)
            if pos.available_volume < trade.volume:
                raise ValueError(
                    f"[{trade.symbol}] 可用持仓不足 T+1 限制! 申请: {trade.volume}, 可用: {pos.available_volume}"
                )

            # 费用：佣金 + 印花税
            fee = txn_amount * self.commission
            tax = txn_amount * self.stamp_duty
            revenue = txn_amount - fee - tax

            # 入账
            self.balance += revenue

            # 计算已实现盈亏 PnL = (Price - AvgPrice) * Vol - Fee - Tax
            pnl = (trade.price - pos.avg_price) * trade.volume - fee - tax
            pos.realized_pnl += pnl

            # 扣减持仓
            pos.volume -= trade.volume
            pos.available_volume -= trade.volume  # 卖出时同时扣减总仓和可用

            if pos.volume == 0:
                pos.avg_price = 0.0

    def settle_daily(self):
        """
        [钩子方法实现]
        收盘结算：将所有的今日买入转为明日可用
        """
        for pos in self.positions.values():
            if pos.volume > 0:
                # A股 T+1 核心逻辑：收盘后，将总持仓同步给可用持仓
                pos.available_volume = pos.volume
