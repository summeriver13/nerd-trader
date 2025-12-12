"""
src/simulation/account/cn_stock.py
A股账户实现：
1. T+1 (买入冻结，收盘解冻)
2. 费用 (佣金最低5元 + 印花税)
3. 只能做多
"""

from src.core.types import Direction, TradeData
from src.simulation.base.account import BaseAccount, BasePosition


class CNStockAccount(BaseAccount):
    """
    A股专用账户
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        commission: float = 0.00025,  # 万2.5
        stamp_duty: float = 0.0005,  # 千0.5 (2023新规: 卖出方单边征收)
        min_commission: float = 5.0,  # [新增] 最低佣金 5元
    ):
        super().__init__(initial_balance, position_cls=BasePosition)

        self.commission = commission
        self.stamp_duty = stamp_duty
        self.min_commission = min_commission

    def on_trade(self, trade: TradeData):
        """
        处理A股成交
        """
        self.trades.append(trade)
        pos = self.get_position(trade.symbol)

        txn_amount = trade.price * trade.volume

        # --- 1. 计算佣金 (买卖双向，有最低门槛) ---
        raw_commission = txn_amount * self.commission
        commission_fee = max(self.min_commission, raw_commission)

        # --- 2. 买入逻辑 (Long) ---
        if trade.direction == Direction.LONG:
            # 买入总成本 = 货款 + 佣金
            total_cost = txn_amount + commission_fee

            # 扣款
            self.balance -= total_cost

            # 更新持仓成本 (加权平均)
            # 注：这里将手续费摊入了持仓成本，这是个人投资者的常见记账方式
            # 机构通常将手续费单列为财务费用
            old_cost = pos.volume * pos.avg_price
            new_cost = old_cost + total_cost

            pos.volume += trade.volume
            if pos.volume > 0:
                pos.avg_price = new_cost / pos.volume

            # 【T+1 关键】：买入时，只增加 volume，不增加 available_volume
            # available_volume 保持不变

        # --- 3. 卖出逻辑 (Short/Sell) ---
        elif trade.direction == Direction.SHORT:
            # 检查 T+1 可用持仓
            if pos.available_volume < trade.volume:
                # 回测中如果策略写错了，这里直接报错提醒
                raise ValueError(
                    f"[{trade.symbol}] 卖出失败：可用持仓不足 (T+1限制)。"
                    f"申请: {trade.volume}, 可用: {pos.available_volume}"
                )

            # 计算印花税 (仅卖出收取)
            tax_fee = txn_amount * self.stamp_duty

            # 卖出净回款 = 货款 - 佣金 - 印花税
            net_revenue = txn_amount - commission_fee - tax_fee

            # 入账
            self.balance += net_revenue

            # 计算平仓盈亏 (Realized PnL)
            # PnL = (卖价 - 成本价) * 数量 - 所有费用
            # 注意：这里的成本价 avg_price 在买入时已经包含了买入佣金，
            # 所以这里减去的费用是 "卖出时的费用"
            total_sell_fee = commission_fee + tax_fee
            pnl = (trade.price - pos.avg_price) * trade.volume - total_sell_fee
            pos.realized_pnl += pnl

            # 扣减持仓
            pos.volume -= trade.volume
            pos.available_volume -= trade.volume  # 卖出时必须同时扣减

            # 清仓后重置均价
            if pos.volume <= 0:
                pos.volume = 0
                pos.available_volume = 0
                pos.avg_price = 0.0

    def settle_daily(self):
        """
        [收盘结算]
        将今日买入的冻结持仓，转为明日可用
        """
        for pos in self.positions.values():
            if pos.volume > 0:
                # 简单粗暴：收盘后，所有持仓变为可用
                pos.available_volume = pos.volume
