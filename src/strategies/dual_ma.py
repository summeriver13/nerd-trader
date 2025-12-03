import itertools

import numpy as np
import pandas as pd

from .base import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """
    双均线趋势策略 (Dual Moving Average) + ATR 移动止损
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 60,
        atr_period: int = 14,
        atr_multiplier: float = 3.0,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.sort_index()

        # 1. 均线
        df["MA_Fast"] = df["close"].rolling(self.fast_period).mean()
        df["MA_Slow"] = df["close"].rolling(self.slow_period).mean()

        # 2. ATR (Average True Range)
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成信号 (包含移动止损逻辑)
        """
        # 确保指标已计算
        if "MA_Fast" not in df.columns or "ATR" not in df.columns:
            df = self.calculate_indicators(df)

        # 调用核心模拟逻辑
        signals = self._run_simulation_logic(
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            self.atr_multiplier,
        )

        df["signal"] = signals
        return df

    def _run_simulation_logic(
        self, close_arr, high_arr, low_arr, ma_fast_arr, ma_slow_arr, atr_arr, atr_mult
    ):
        """
        核心交易模拟逻辑 (Numpy 加速版)
        """
        length = len(close_arr)
        signals = np.zeros(length)

        position = 0  # 0: 空仓, 1: 持仓
        stop_price = 0.0  # 当前止损价
        highest_price = 0.0  # 开仓后的最高价

        for i in range(1, length):
            if (
                np.isnan(ma_fast_arr[i])
                or np.isnan(ma_slow_arr[i])
                or np.isnan(atr_arr[i])
            ):
                continue

            if position == 1:
                # 1. 检查止损
                if low_arr[i] < stop_price:
                    position = 0
                # 2. 检查死叉
                elif ma_fast_arr[i] < ma_slow_arr[i]:
                    position = 0
                # 3. 持有并更新止损
                else:
                    if high_arr[i] > highest_price:
                        highest_price = high_arr[i]
                        new_stop = highest_price - (atr_mult * atr_arr[i])
                        stop_price = max(stop_price, new_stop)
            else:
                # 检查金叉
                if ma_fast_arr[i] > ma_slow_arr[i]:
                    position = 1
                    highest_price = high_arr[i]
                    stop_price = highest_price - (atr_mult * atr_arr[i])

            signals[i] = position

        return signals

    def optimize(
        self,
        df: pd.DataFrame,
        fast_range=range(5, 30, 2),
        slow_range=range(30, 100, 5),
        atr_mult_range=[3.0, 4.0, 5.0, 6.0],
    ) -> dict:  # 新增 ATR 范围
        """
        全真网格搜索：包含 ATR 止损逻辑的参数优化
        """
        print("\n🚀 [策略优化] 启动全真网格搜索 (含 ATR 止损)...")
        results = []

        # 预先提取 Numpy 数组以加速
        close_arr = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values

        # 预计算 ATR (ATR 周期固定为14，不参与优化以减少维度，只优化倍数)
        # 如果还没算过，先算一遍基础指标
        temp_strat = DualMAStrategy(atr_period=self.atr_period)
        df_temp = temp_strat.calculate_indicators(df.copy())
        atr_arr = df_temp["ATR"].values

        combinations = list(itertools.product(fast_range, slow_range, atr_mult_range))
        print(f"   测试组合数: {len(combinations)}...")

        for fast, slow, atr_mult in combinations:
            if fast >= slow:
                continue

            # 动态计算均线
            # 添加 # type: ignore 忽略 basedpyright 对 pandas 返回类型的误判
            ma_fast = pd.Series(close_arr).rolling(fast).mean().values  # type: ignore
            ma_slow = pd.Series(close_arr).rolling(slow).mean().values  # type: ignore

            # 运行核心模拟逻辑
            signals = self._run_simulation_logic(
                close_arr, high_arr, low_arr, ma_fast, ma_slow, atr_arr, atr_mult
            )

            # 快速计算收益
            pos = np.roll(signals, 1)  # shift(1)
            pos[0] = 0

            # [修正] 标准向量化收益率计算：(P_t - P_{t-1}) / P_{t-1}
            # 使用 np.zeros_like 初始化，确保第一个元素为 0
            pct_change = np.zeros_like(close_arr)
            # 计算从索引 1 开始的收益率
            pct_change[1:] = np.diff(close_arr) / close_arr[:-1]  # type: ignore

            strategy_ret = pct_change * pos

            # 交易次数 (从0变1，或1变0)
            trades = np.sum(np.abs(np.diff(pos)))

            # 扣费净收益 (简单估算)
            net_ret = np.sum(strategy_ret) - (trades * 0.0005)

            results.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "atr_mult": atr_mult,
                    "net_ret": net_ret,
                    "trades": trades,
                }
            )

        if not results:
            return {"fast": 10, "slow": 60, "atr_mult": 3.0}

        res_df = pd.DataFrame(results)
        # 排序取最优
        best = res_df.sort_values(by="net_ret", ascending=False).iloc[0]

        print(
            f"🏆 最佳参数: Fast={int(best['fast'])}, Slow={int(best['slow'])}, ATR倍数={best['atr_mult']}"
        )
        print(
            f"   预估收益: {best['net_ret'] * 100:.2f}%, 交易次数: {int(best['trades'])}"
        )

        self.fast_period = int(best["fast"])
        self.slow_period = int(best["slow"])
        self.atr_multiplier = float(best["atr_mult"])

        return {
            "fast": self.fast_period,
            "slow": self.slow_period,
            "atr_mult": self.atr_multiplier,
        }
