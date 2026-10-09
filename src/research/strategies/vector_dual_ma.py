"""
src/research/strategies/vector_dual_ma.py
自适应双均线策略 - 向量化回测版 (Vectorized)
[最终架构适配版]
修改点:
1. 适配 BaseStrategy 基类，移除重复的市场判断逻辑
2. 直接调用 self.can_short 控制空头权限
"""

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """
    自适应双均线策略 (Adaptive Dual MA)
    """

    def __init__(
        self,
        # === 趋势判断 ===
        trend_period: int = 120,
        # === 牛市参数 ===
        bull_fast: int = 14,
        bull_slow: int = 25,
        bull_adx_thresh: int = 30,
        bull_atr_mult: float = 3.0,
        # === 熊市参数 ===
        bear_fast: int = 20,
        bear_slow: int = 75,
        bear_adx_thresh: int = 20,
        bear_atr_mult: float = 3.0,
        # === 通用参数 ===
        atr_period: int = 14,
        adx_period: int = 14,
    ):
        # 1. 初始化父类 (父类会自动设置 self.can_short, self.max_leverage)
        super().__init__()

        self.trend_period = trend_period

        self.bull_fast = bull_fast
        self.bull_slow = bull_slow
        self.bull_adx_thresh = bull_adx_thresh
        self.bull_atr_mult = bull_atr_mult

        self.bear_fast = bear_fast
        self.bear_slow = bear_slow
        self.bear_adx_thresh = bear_adx_thresh
        self.bear_atr_mult = bear_atr_mult

        self.atr_period = atr_period
        self.adx_period = adx_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.sort_index()

        # 1. 长期趋势
        df["MA_Trend"] = df["close"].rolling(self.trend_period).mean()

        # 2. 均线组
        df["Bull_Fast"] = df["close"].rolling(self.bull_fast).mean()
        df["Bull_Slow"] = df["close"].rolling(self.bull_slow).mean()
        df["Bear_Fast"] = df["close"].rolling(self.bear_fast).mean()
        df["Bear_Slow"] = df["close"].rolling(self.bear_slow).mean()

        # 3. ATR & ADX
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        # ADX
        up = df["high"] - df["high"].shift(1)
        down = df["low"].shift(1) - df["low"]
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0), index=df.index
        )

        alpha = 1 / self.adx_period
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)

        sum_di = pd.Series(plus_di + minus_di, index=df.index).replace(0, np.nan)
        dx = 100 * (abs(plus_di - minus_di) / sum_di)
        df["ADX"] = dx.fillna(0).ewm(alpha=alpha, adjust=False).mean()  # type: ignore

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        signals, _ = self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Trend"].values,
            df["Bull_Fast"].values,
            df["Bull_Slow"].values,
            df["Bear_Fast"].values,
            df["Bear_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_adx_thresh,
            self.bear_atr_mult,
        )
        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        return self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Trend"].values,
            df["Bull_Fast"].values,
            df["Bull_Slow"].values,
            df["Bear_Fast"].values,
            df["Bear_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_adx_thresh,
            self.bear_atr_mult,
            record_trades=True,
            timestamps=df.index,
        )  # type: ignore

    def _run_simulation_logic(
        self,
        open_arr,
        close_arr,
        high_arr,
        low_arr,
        trend_ma_arr,
        bull_fast,
        bull_slow,
        bear_fast,
        bear_slow,
        atr_arr,
        adx_arr,
        bull_adx_thresh,
        bull_atr_mult,
        bear_adx_thresh,
        bear_atr_mult,
        record_trades=False,
        timestamps=None,
    ):
        length = len(close_arr)
        signals = np.zeros(length)
        returns = np.zeros(length)
        trade_logs = []

        position = 0
        stop_price = 0.0
        extreme_price = 0.0
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, length):
            if (
                np.isnan(trend_ma_arr[i])
                or np.isnan(bull_slow[i])
                or np.isnan(bear_slow[i])
            ):
                continue

            prev_close = close_arr[i - 1]
            bar_return = 0.0

            # === 1. 市场状态识别 ===
            is_bull_market = close_arr[i] > trend_ma_arr[i]

            if is_bull_market:
                curr_fast = bull_fast[i]
                curr_slow = bull_slow[i]
                curr_adx_thresh = bull_adx_thresh
                curr_atr_mult = bull_atr_mult
                allow_short = False  # 牛市死叉仅平仓，不做空
            else:
                curr_fast = bear_fast[i]
                curr_slow = bear_slow[i]
                curr_adx_thresh = bear_adx_thresh
                curr_atr_mult = bear_atr_mult
                # [关键简化] 直接读取基类属性
                allow_short = self.can_short

            exit_reason = ""
            exit_price = 0.0

            # === 2. 持仓管理 ===
            if position == 1:
                bar_return = (close_arr[i] - prev_close) / prev_close
                # 止损
                if low_arr[i] < stop_price:
                    position = 0
                    exit_reason = "Long_StopLoss"
                    exit_price = (
                        min(open_arr[i], stop_price)
                        if open_arr[i] < stop_price
                        else stop_price
                    )
                    bar_return = (exit_price - prev_close) / prev_close
                # 死叉平仓
                elif curr_fast < curr_slow:
                    position = 0
                    exit_reason = "Long_DeadCross"
                    exit_price = close_arr[i]
                    # 反手开空 (仅当允许做空时)
                    if allow_short and adx_arr[i] > curr_adx_thresh:
                        position = -1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = low_arr[i]
                        stop_price = extreme_price + (curr_atr_mult * atr_arr[i])
                # 移动止损
                else:
                    if high_arr[i] > extreme_price:
                        extreme_price = high_arr[i]
                        new_stop = extreme_price - (curr_atr_mult * atr_arr[i])
                        stop_price = max(stop_price, new_stop)

            elif position == -1:
                bar_return = -1 * (close_arr[i] - prev_close) / prev_close
                # 止损
                if high_arr[i] > stop_price:
                    position = 0
                    exit_reason = "Short_StopLoss"
                    exit_price = (
                        max(open_arr[i], stop_price)
                        if open_arr[i] > stop_price
                        else stop_price
                    )
                    bar_return = -1 * (exit_price - prev_close) / prev_close
                # 金叉平仓
                elif curr_fast > curr_slow:
                    position = 0
                    exit_reason = "Short_GoldenCross"
                    exit_price = close_arr[i]
                    # 反手开多
                    if adx_arr[i] > curr_adx_thresh:
                        position = 1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = high_arr[i]
                        stop_price = extreme_price - (curr_atr_mult * atr_arr[i])
                # 移动止损
                else:
                    if low_arr[i] < extreme_price:
                        extreme_price = low_arr[i]
                        new_stop = extreme_price + (curr_atr_mult * atr_arr[i])
                        stop_price = min(stop_price, new_stop)

            # === 3. 空仓开单 ===
            else:
                if adx_arr[i] > curr_adx_thresh:
                    if curr_fast > curr_slow:
                        position = 1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = high_arr[i]
                        stop_price = extreme_price - (curr_atr_mult * atr_arr[i])
                    elif allow_short and (curr_fast < curr_slow):
                        position = -1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = low_arr[i]
                        stop_price = extreme_price + (curr_atr_mult * atr_arr[i])

            signals[i] = position
            returns[i] = bar_return

            if record_trades and position == 0 and exit_reason:
                pnl = (exit_price - entry_price) / entry_price
                if "Short" in exit_reason:
                    pnl *= -1
                trade_logs.append(
                    {
                        "entry_time": timestamps[entry_idx],  # type: ignore
                        "exit_time": timestamps[i],  # type: ignore
                        "reason": exit_reason,
                        "pnl": pnl,
                    }
                )

        if record_trades:
            return trade_logs
        return signals, returns

