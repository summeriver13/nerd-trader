"""
src/strategies/dual_ma.py
双均线策略 - 事件驱动版 (Event-Driven)
Refactored: Pass 0 as price for Market Orders to ensure execution
"""

from src.core.interfaces import StrategyBase, StrategyContext
from src.core.types import BarData, OrderData, OrderType, TickData, TradeData
from src.core.utility import ArrayManager, BarGenerator


class DualMaStrategy(StrategyBase):
    """
    双均线策略 (Event-Driven)
    """

    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma", "slow_ma", "pos"]

    def __init__(self, context: StrategyContext, strategy_name: str, settings: dict):
        super().__init__(context, strategy_name, settings)

        self.fast_window = settings.get("fast_window", 10)
        self.slow_window = settings.get("slow_window", 20)
        self.fixed_size = settings.get("fixed_size", 1.0)

        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.pos = 0

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=self.slow_window + 10)

    def on_init(self):
        self.write_log("策略初始化完成")

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)

        if not self.am.inited:
            return

        fast_ma = self.am.sma(self.fast_window)
        slow_ma = self.am.sma(self.slow_window)

        self.fast_ma = fast_ma
        self.slow_ma = slow_ma

        cross_over = fast_ma > slow_ma
        cross_below = fast_ma < slow_ma

        # --- 交易逻辑 ---

        if self.pos == 0:
            if cross_over:
                self.write_log("金叉 -> 买入")
                # [修改] 价格传 0，确保作为市价单成交
                self.buy(bar.symbol, 0, self.fixed_size, OrderType.MARKET)
                self.pos += self.fixed_size

            elif cross_below:
                self.write_log("死叉 -> 卖出")
                # [修改] 价格传 0
                self.sell(bar.symbol, 0, self.fixed_size, OrderType.MARKET)
                self.pos -= self.fixed_size

        elif self.pos > 0:
            if cross_below:
                self.write_log("死叉平仓")
                # [修改] 价格传 0
                self.sell(bar.symbol, 0, self.pos, OrderType.MARKET)
                self.pos = 0

        elif self.pos < 0:
            if cross_over:
                self.write_log("金叉平仓")
                # [修改] 价格传 0
                self.buy(bar.symbol, 0, abs(self.pos), OrderType.MARKET)
                self.pos = 0

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        """
        成交推送
        """
        self.write_log(
            f"成交回报: {trade.direction.value} {trade.price}@{trade.volume}"
        )
        pass
