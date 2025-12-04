import numpy as np
import pandas as pd

# 假设你的 BaseStrategy 定义在 src.strategies.base
# 如果找不到，请确保该文件存在或调整路径
from src.strategies.base import BaseStrategy
from src.utils.statistics import calculate_z_score


class BollingerMRStrategy(BaseStrategy):
    """
    布林带/Z-Score 均值回归策略 (Bollinger Mean Reversion)

    核心思想：
    利用 Z-Score (标准分) 衡量价格偏离均线的程度。
    假设价格服从正态分布，偏离 2 个标准差的概率仅为 5%，因此大概率会回归。
    """

    def __init__(
        self,
        window: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        stop_loss_std: float = 4.0,
    ) -> None:
        """
        初始化策略参数
        :param window: 均线回顾窗口 (例如 20)
        :param entry_std: 开仓阈值 (标准差倍数, 通常为 2.0)
        :param exit_std: 平仓阈值 (回归到均值附近的范围, 通常为 0.5 或 0)
        :param stop_loss_std: 止损阈值 (如果偏离过大，说明进入单边趋势，必须止损)
        """
        super().__init__()  # 假如父类有初始化逻辑
        self.window = window
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.stop_loss_std = stop_loss_std

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的指标 (Z-Score)
        """
        # 避免 SettingWithCopyWarning，显式拷贝
        df = df.copy()

        # 1. 计算 Z-Score
        df["z_score"] = calculate_z_score(df["close"], self.window)  # type: ignore

        # 2. (可选) 计算传统布林带用于画图观察，不参与核心逻辑
        # 使用 assign 可以链式调用，或者保持现在的写法
        roll = df["close"].rolling(self.window)
        boll_mid = roll.mean()
        boll_std = roll.std()

        df["boll_mid"] = boll_mid
        df["boll_upper"] = boll_mid + self.entry_std * boll_std
        df["boll_lower"] = boll_mid - self.entry_std * boll_std

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号 (Target Position: 1=Long, -1=Short, 0=Empty)
        使用 NumPy 数组遍历以提升性能并消除 Pandas 类型警告。
        """
        # 确保 z_score 已计算
        if "z_score" not in df.columns:
            df = self.calculate_indicators(df)

        # --- 核心优化：转为 NumPy 数组处理 ---
        # 相比 itertuples，numpy 迭代速度快得多，且不会有属性访问的 lint 警告
        z_values = df["z_score"].to_numpy()

        # 初始化信号数组 (全 0)
        signals = np.zeros(len(z_values), dtype=int)

        # 记录当前持仓状态: 0空仓, 1多头, -1空头
        current_position = 0

        # 遍历数组
        for i, z in enumerate(z_values):
            # 处理 NaN (数据初期)
            if np.isnan(z):
                signals[i] = 0
                continue

            z_val = float(z)  # 显式转换为 float，辅助类型检查
            abs_z = abs(z_val)

            # --- 1. 止损逻辑 (优先级最高) ---
            # 偏离过大 -> 趋势爆发 -> 认输出局
            if abs_z > self.stop_loss_std:
                current_position = 0

            # --- 2. 开仓逻辑 (仅在空仓时检查) ---
            elif current_position == 0:
                if z_val < -self.entry_std:
                    # 价格超卖 -> 做多
                    current_position = 1
                elif z_val > self.entry_std:
                    # 价格超买 -> 做空
                    current_position = -1

            # --- 3. 平仓逻辑 (仅在持仓时检查) ---
            elif current_position == 1:
                # 持多单：Z-Score 回升到 -0.5 以上 (均值回归)
                if z_val > -self.exit_std:
                    current_position = 0

            elif current_position == -1:
                # 持空单：Z-Score 下降到 0.5 以下 (均值回归)
                if z_val < self.exit_std:
                    current_position = 0

            # 记录当前时间点的目标持仓
            signals[i] = current_position

        # 将 numpy 数组赋值回 DataFrame
        df["signal"] = signals
        return df
