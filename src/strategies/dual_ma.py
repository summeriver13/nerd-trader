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

        # 2. ATR
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MA_Fast" not in df.columns or "ATR" not in df.columns:
            df = self.calculate_indicators(df)

        signals, _ = self._run_simulation_logic(
            df["open"].values,
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

    def get_trade_history(self, df: pd.DataFrame) -> list:
        """
        [新增] 获取详细的交易流水记录
        """
        if "MA_Fast" not in df.columns:
            df = self.calculate_indicators(df)

        return self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            self.atr_multiplier,
            record_trades=True,
            timestamps=df.index,
        )  # type: ignore

    def _run_simulation_logic(
        self,
        open_arr,
        close_arr,
        high_arr,
        low_arr,
        ma_fast_arr,
        ma_slow_arr,
        atr_arr,
        atr_mult,
        record_trades=False,
        timestamps=None,
    ):
        """
        核心逻辑 (支持返回信号数组 或 交易流水列表)
        """
        length = len(close_arr)
        signals = np.zeros(length)
        returns = np.zeros(length)
        trade_logs = []

        position = 0
        stop_price = 0.0
        highest_price = 0.0

        # 记录单笔交易的临时变量
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, length):
            if (
                np.isnan(ma_fast_arr[i])
                or np.isnan(ma_slow_arr[i])
                or np.isnan(atr_arr[i])
            ):
                continue

            prev_close = close_arr[i - 1]
            bar_return = (close_arr[i] - prev_close) / prev_close  # 默认持有收益

            exit_reason = ""
            exit_price = 0.0

            # === 持仓逻辑 ===
            if position == 1:
                # 1. 检查止损
                if low_arr[i] < stop_price:
                    position = 0
                    exit_reason = "StopLoss"
                    exit_price = stop_price
                    # 处理跳空低开
                    if open_arr[i] < stop_price:
                        exit_price = open_arr[i]

                    bar_return = (exit_price - prev_close) / prev_close

                # 2. 检查死叉
                elif ma_fast_arr[i] < ma_slow_arr[i]:
                    position = 0
                    exit_reason = "DeathCross"
                    exit_price = close_arr[i]

                # 3. 持有更新
                else:
                    if high_arr[i] > highest_price:
                        highest_price = high_arr[i]
                        new_stop = highest_price - (atr_mult * atr_arr[i])
                        stop_price = max(stop_price, new_stop)

                # 如果触发了平仓，记录交易
                if position == 0 and record_trades:
                    trade_pnl = (exit_price - entry_price) / entry_price
                    trade_logs.append(
                        {
                            "entry_time": timestamps[entry_idx],  # type: ignore
                            "entry_price": entry_price,
                            "exit_time": timestamps[i],  # type: ignore
                            "exit_price": exit_price,
                            "reason": exit_reason,
                            "pnl": trade_pnl,
                            "bars_held": i - entry_idx,
                        }
                    )

            # === 空仓逻辑 ===
            else:
                if ma_fast_arr[i] > ma_slow_arr[i]:
                    position = 1
                    highest_price = high_arr[i]
                    stop_price = highest_price - (atr_mult * atr_arr[i])
                    # 记录开仓信息 (假设以收盘价成交)
                    entry_price = close_arr[i]
                    entry_idx = i
                    bar_return = 0.0  # 进场当根不计算收益(下根算)

            signals[i] = position
            if signals[i - 1] == 1:
                returns[i] = bar_return
            else:
                returns[i] = 0.0

        if record_trades:
            return trade_logs
        return signals, returns

    # ... optimize 方法保持不变，无需修改 ...
    def optimize(
        self,
        df: pd.DataFrame,
        fast_range=range(5, 30, 2),
        slow_range=range(30, 100, 5),
        atr_mult_range=[3.0, 4.0, 5.0, 6.0],
    ) -> dict:
        print("\n🚀 [策略优化] 启动全真网格搜索 (含 ATR 止损)...")
        results = []

        close_arr = df["close"].values
        open_arr = df["open"].values
        high_arr = df["high"].values
        low_arr = df["low"].values

        temp_strat = DualMAStrategy(atr_period=self.atr_period)
        df_temp = temp_strat.calculate_indicators(df.copy())
        atr_arr = df_temp["ATR"].values

        combinations = list(itertools.product(fast_range, slow_range, atr_mult_range))

        for fast, slow, atr_mult in combinations:
            if fast >= slow:
                continue
            # 添加 # type: ignore
            ma_fast = pd.Series(close_arr).rolling(fast).mean().values  # type: ignore
            ma_slow = pd.Series(close_arr).rolling(slow).mean().values  # type: ignore

            signals, strategy_ret_arr = self._run_simulation_logic(
                open_arr,
                close_arr,
                high_arr,
                low_arr,
                ma_fast,
                ma_slow,
                atr_arr,
                atr_mult,
            )

            trades = np.sum(np.abs(np.diff(signals)))
            net_ret = np.sum(strategy_ret_arr) - (trades * 0.0005)

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
        best = res_df.sort_values(by="net_ret", ascending=False).iloc[0]

        self.fast_period = int(best["fast"])
        self.slow_period = int(best["slow"])
        self.atr_multiplier = float(best["atr_mult"])

        return {
            "fast": self.fast_period,
            "slow": self.slow_period,
            "atr_mult": self.atr_multiplier,
        }
