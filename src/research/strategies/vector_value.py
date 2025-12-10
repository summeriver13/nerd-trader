"""
src/research/strategies/vector_value.py
低估值价值投资策略 (PE/PB Valuation Strategy)
逻辑：基于 Baostock 的 peTTM 或 pbMRQ 数据进行择时
"""

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class ValueStrategy(BaseStrategy):
    """
    低估值择时策略
    """

    def __init__(
        self,
        market_type: str = "stock_cn",
        # === 核心参数 ===
        factor_name: str = "pe",  # 使用哪个因子? 'pe' (市盈率) 或 'pb' (市净率)
        buy_threshold: float = 10.0,  # 低于此值买入 (低估)
        sell_threshold: float = 20.0,  # 高于此值卖出 (高估)
        # === 附加风控 ===
        stop_loss_pct: float = 0.10,  # 10% 止损 (防止价值陷阱)
    ):
        super().__init__(market_type=market_type)
        self.factor_name = factor_name
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检查是否有基本面数据
        """
        if self.factor_name not in df.columns:
            # 如果数据源没返回 PE/PB (比如 Crypto)，填充 NaN
            df[self.factor_name] = np.nan
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.factor_name not in df.columns:
            print(f"⚠️ 数据缺失 {self.factor_name} 指标，策略失效")
            df["signal"] = 0
            return df

        # 获取因子序列 (pe 或 pb)
        factor_values = df[self.factor_name].values
        close_arr = df["close"].values

        length = len(df)
        signals = np.zeros(length)

        position = 0
        entry_price = 0.0

        for i in range(1, length):
            current_val = factor_values[i]

            # 如果因子是 NaN (比如上市早期数据缺失)，保持空仓
            if np.isnan(current_val):
                continue

            # === 1. 持仓管理 ===
            if position == 1:
                # A. 估值修复完成 (高估) -> 止盈
                if current_val > self.sell_threshold:
                    position = 0

                # B. 价值陷阱风控 (越跌越贵，或者基本面暴雷) -> 止损
                # 计算当前亏损幅度
                elif entry_price > 0:
                    loss_pct = (entry_price - close_arr[i]) / entry_price
                    if loss_pct > self.stop_loss_pct:
                        position = 0

            # === 2. 空仓开单 ===
            else:
                # 击穿低估线 -> 买入
                # 且因子必须大于0 (过滤掉亏损股 PE 为负的情况)
                if 0 < current_val < self.buy_threshold:
                    position = 1
                    entry_price = close_arr[i]

            signals[i] = position

        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list:
        # 简单起见，复用上面的逻辑记录交易，或者暂时返回空
        # 为了回测图表画点，Engine 主要依赖 signal 列，这里可以简化
        return []

    def optimize(self, df, **kwargs):
        # 简单策略不需要复杂优化，手动调参即可
        return {}
