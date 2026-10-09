"""
src/strategies/dual_ma.py
双均线策略 - 事件驱动版 (Event-Driven)
Refactored: Pass 0 as price for Market Orders to ensure execution
"""

from typing import Any, override

from src.core.interfaces import StrategyBase, StrategyContext
from src.core.types import BarData, OrderData, OrderType, TickData, TradeData
from src.core.utility import ArrayManager, BarGenerator


class DualMaStrategy(StrategyBase):
    """
    双均线策略 (Event-Driven)
    """

    parameters = ["fast_window", "slow_window", "fixed_size"]

    def __init__(
        self, context: StrategyContext, strategy_name: str, settings: dict[str, Any]
    ) -> None:
        super().__init__(context, strategy_name, settings)

        # 参数缺一个就报错，不用 .get 静默回退：写错键名会直接停，而不是悄悄变回 1 份仓位
        missing = [key for key in self.parameters if key not in settings]
        if missing:
            raise ValueError(f"{strategy_name} 缺少必需参数: {missing}")

        self.fast_window = settings["fast_window"]
        self.slow_window = settings["slow_window"]
        self.fixed_size = settings["fixed_size"]

        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.pos = 0

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=self.slow_window + 10)

    @override
    def on_init(self):
        self.write_log("策略初始化完成")

    @override
    def on_start(self):
        self.write_log("策略启动")

    @override
    def on_stop(self):
        self.write_log("策略停止")

    @override
    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    @override
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
                self.write_log("死叉 -> 开空")
                # [修改] 价格传 0，确保作为市价单成交
                self.short(bar.symbol, 0, self.fixed_size, OrderType.MARKET)
                self.pos -= self.fixed_size

        elif self.pos > 0 and cross_below:
            self.write_log("死叉平仓")
            # [修改] 价格传 0
            self.sell(bar.symbol, 0, self.pos, OrderType.MARKET)
            self.pos = 0

        elif self.pos < 0 and cross_over:
            self.write_log("金叉 -> 平空")
            # [修改] 价格传 0，确保作为市价单成交
            self.cover(bar.symbol, 0, abs(self.pos), OrderType.MARKET)
            self.pos = 0

    @override
    def on_order(self, order: OrderData):
        pass

    @override
    def on_trade(self, trade: TradeData):
        """
        成交推送
        """
        self.write_log(
            f"成交回报: {trade.direction.value} {trade.price}@{trade.volume}"
        )
        pass
