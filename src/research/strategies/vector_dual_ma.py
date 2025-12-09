"""
src/research/strategies/vector_dual_ma.py
自适应双均线策略 - 向量化回测版 (Vectorized)
核心改进:
1. 引入"市场状态识别" (Regime Switching): 基于长均线(MA120)判断牛熊
2. 牛市使用激进参数只做多，熊市使用稳健参数多空双吃
3. 动态调整 ADX 阈值和止损幅度
"""

import itertools

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
        trend_period: int = 120,  # 比如 120周期(半年线)作为牛熊分界
        # === 牛市参数 (激进/进攻) ===
        bull_fast: int = 14,
        bull_slow: int = 25,
        bull_adx_thresh: int = 30,
        bull_atr_mult: float = 3.0,
        # === 熊市参数 (稳健/防守) ===
        bear_fast: int = 20,
        bear_slow: int = 75,
        bear_adx_thresh: int = 20,
        bear_atr_mult: float = 3.0,
        # === 通用参数 ===
        atr_period: int = 14,
        adx_period: int = 14,
    ):
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

        # 1. 计算长期趋势线 (牛熊分界)
        df["MA_Trend"] = df["close"].rolling(self.trend_period).mean()

        # 2. 计算两套均线系统
        # 牛市激进组
        df["Bull_Fast"] = df["close"].rolling(self.bull_fast).mean()
        df["Bull_Slow"] = df["close"].rolling(self.bull_slow).mean()
        # 熊市稳健组
        df["Bear_Fast"] = df["close"].rolling(self.bear_fast).mean()
        df["Bear_Slow"] = df["close"].rolling(self.bear_slow).mean()

        # 3. ATR & ADX 计算 (通用)
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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

        df["ADX"] = (
            pd.Series(dx, index=df.index)
            .fillna(0)
            .ewm(alpha=alpha, adjust=False)
            .mean()
        )

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
        """
        自适应核心逻辑循环
        """
        length = len(close_arr)
        signals = np.zeros(length)
        returns = np.zeros(length)
        trade_logs = []

        position = 0  # 0:空仓, 1:多头, -1:空头
        stop_price = 0.0
        extreme_price = 0.0
        entry_price = 0.0
        entry_idx = 0

        for i in range(1, length):
            # 基础数据检查
            if (
                np.isnan(trend_ma_arr[i])
                or np.isnan(bull_slow[i])
                or np.isnan(bear_slow[i])
            ):
                continue

            prev_close = close_arr[i - 1]
            bar_return = 0.0

            # === 1. 市场状态识别 (Regime Detection) ===
            # 如果收盘价在长期均线之上 -> 牛市 (Bull)
            is_bull_market = close_arr[i] > trend_ma_arr[i]

            # 根据状态选择参数
            if is_bull_market:
                curr_fast = bull_fast[i]
                curr_slow = bull_slow[i]
                curr_adx_thresh = bull_adx_thresh
                curr_atr_mult = bull_atr_mult
                allow_short = False  # [牛市不做空]
            else:
                curr_fast = bear_fast[i]
                curr_slow = bear_slow[i]
                curr_adx_thresh = bear_adx_thresh
                curr_atr_mult = bear_atr_mult
                allow_short = True  # [熊市允许做空]

            exit_reason = ""
            exit_price = 0.0

            # === 2. 持仓管理 ===
            if position == 1:  # 持有多头
                bar_return = (close_arr[i] - prev_close) / prev_close

                # 止损检查
                if low_arr[i] < stop_price:
                    position = 0
                    exit_reason = "Long_StopLoss"
                    exit_price = (
                        min(open_arr[i], stop_price)
                        if open_arr[i] < stop_price
                        else stop_price
                    )
                    bar_return = (exit_price - prev_close) / prev_close

                # 死叉反转
                elif curr_fast < curr_slow:
                    position = 0
                    exit_reason = "Long_DeadCross"
                    exit_price = close_arr[i]

                    # 如果允许做空 且 趋势够强 -> 反手开空
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

            elif position == -1:  # 持有空头
                bar_return = -1 * (close_arr[i] - prev_close) / prev_close

                # 止损检查
                if high_arr[i] > stop_price:
                    position = 0
                    exit_reason = "Short_StopLoss"
                    exit_price = (
                        max(open_arr[i], stop_price)
                        if open_arr[i] > stop_price
                        else stop_price
                    )
                    bar_return = -1 * (exit_price - prev_close) / prev_close

                # 金叉反转
                elif curr_fast > curr_slow:
                    position = 0
                    exit_reason = "Short_GoldenCross"
                    exit_price = close_arr[i]

                    # 反手开多 (如果 ADX 达标)
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
            else:  # position == 0
                if adx_arr[i] > curr_adx_thresh:
                    # 金叉开多
                    if curr_fast > curr_slow:
                        position = 1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = high_arr[i]
                        stop_price = extreme_price - (curr_atr_mult * atr_arr[i])

                    # 死叉开空 (仅在允许做空时)
                    elif allow_short and (curr_fast < curr_slow):
                        position = -1
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = low_arr[i]
                        stop_price = extreme_price + (curr_atr_mult * atr_arr[i])

            # 记录数据
            signals[i] = position
            if position != 0:
                pass  # 占位符
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

    # 简单的 Optimize 接口 (暂时只优化 Trend Period，因为参数太多了)
    def optimize(self, df: pd.DataFrame, **kwargs) -> dict:
        print("⚠️ 自适应策略参数较为复杂，暂不支持全自动网格搜索。")
        print("建议使用默认的 '黄金参数' 组合。")
        return {}
