from typing import cast, override

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy
from src.utils.statistics import calculate_z_score


class BollingerMRStrategy(BaseStrategy):
    """
    布林带/Z-Score 均值回归策略 (Bollinger Mean Reversion)

    增强版: 加入趋势过滤器 (Trend Filter)
    """

    def __init__(
        self,
        window: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        stop_loss_std: float = 4.0,
        trend_ma_period: int = 200,  # [新增] 趋势均线周期
    ) -> None:
        """
        :param trend_ma_period: 趋势过滤均线 (如200)，用于判断大势。0 表示不启用过滤。
        """
        super().__init__()
        self.window = window
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.stop_loss_std = stop_loss_std
        self.trend_ma_period = trend_ma_period  # [新增]

    @property
    @override
    def warmup_bars(self) -> int:
        """Z-Score 窗口与趋势均线取较长者（趋势过滤关闭时即窗口本身）"""
        return max(self.window, self.trend_ma_period)

    @override
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. 计算 Z-Score (短期回归指标)
        df["z_score"] = calculate_z_score(cast(pd.Series, df["close"]), self.window)

        # 2. [新增] 计算趋势均线 (长期趋势指标)
        if self.trend_ma_period > 0:
            df["trend_ma"] = df["close"].rolling(window=self.trend_ma_period).mean()

        # 3. (可选) 传统布林带用于画图
        roll = df["close"].rolling(self.window)
        boll_mid = roll.mean()
        boll_std = roll.std()
        df["boll_mid"] = boll_mid
        df["boll_upper"] = boll_mid + self.entry_std * boll_std
        df["boll_lower"] = boll_mid - self.entry_std * boll_std

        return df

    @override
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "z_score" not in df.columns:
            df = self.calculate_indicators(df)

        z_values = df["z_score"].to_numpy()
        close_prices = df["close"].to_numpy()

        # [新增] 获取趋势均线数组
        if self.trend_ma_period > 0:
            trend_ma_values = df["trend_ma"].to_numpy()
        else:
            trend_ma_values = np.zeros(len(close_prices))  # 占位符

        signals = np.zeros(len(z_values), dtype=int)
        current_position = 0

        for i, z in enumerate(z_values):
            if np.isnan(z):
                signals[i] = 0
                continue

            # [新增] 判断当前趋势状态
            # 如果启用过滤器，且 trend_ma 有值
            is_bull_trend = True  # 默认允许做多
            is_bear_trend = True  # 默认允许做空

            if self.trend_ma_period > 0 and not np.isnan(trend_ma_values[i]):
                # 价格 > 均线 -> 牛市 -> 只做多 (禁止做空)
                if close_prices[i] > trend_ma_values[i]:
                    is_bull_trend = True
                    is_bear_trend = False
                # 价格 < 均线 -> 熊市 -> 只做空 (禁止做多)
                else:
                    is_bull_trend = False
                    is_bear_trend = True

            z_val = float(z)
            abs_z = abs(z_val)

            # --- 1. 止损逻辑 ---
            if abs_z > self.stop_loss_std:
                current_position = 0

            # --- 2. 开仓逻辑 ---
            elif current_position == 0:
                # 只有在“非熊市”且超卖时，才开多
                if z_val < -self.entry_std and is_bull_trend:
                    current_position = 1

                # 只有在“非牛市”且超买时，才开空
                elif z_val > self.entry_std and is_bear_trend:
                    current_position = -1

            # --- 3. 平仓逻辑 ---
            elif (
                current_position == 1
                and z_val > -self.exit_std
                or current_position == -1
                and z_val < self.exit_std
            ):
                current_position = 0

            signals[i] = current_position

        df["signal"] = signals
        return df
