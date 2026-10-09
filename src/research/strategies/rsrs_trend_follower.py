"""
src/research/strategies/rsrs_trend_follower.py
RSRS (Resistance Support Relative Strength) 趋势增强策略
代码规范: Ruff + Python 3.12 + Pyright Strict
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class RSRSTrendStrategy(BaseStrategy):
    """
    RSRS 趋势追踪策略。
    """

    def __init__(
        self,
        rsrs_n: int = 18,
        rsrs_m: int = 600,
        buy_threshold: float = 0.7,
        sell_threshold: float = -0.7,
        max_leverage: float = 1.0,
        atr_period: int = 14,
        stop_loss_mult: float = 2.5,
    ) -> None:
        super().__init__()
        self.rsrs_n = rsrs_n
        self.rsrs_m = rsrs_m
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.atr_period = atr_period
        self.stop_loss_mult = stop_loss_mult
        self.effective_max_lev = min(max_leverage, self.max_leverage)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        利用向量化计算 RSRS 指标。
        修复了 ndarray 没有 replace 属性的问题，并优化了除零处理。
        """
        if df is None or df.empty:
            return df

        df = df.sort_index().copy()

        # 1. 向量化计算 RSRS 斜率
        rolling_h = df["high"].rolling(window=self.rsrs_n)

        # 计算协方差和方差
        cov_hl = rolling_h.cov(df["low"])
        var_h = rolling_h.var()

        # 使用 numpy.where 替代 Series.replace，处理分母为 0 的情况
        # 这样即便 Pyright 推断为 ndarray 也能正常工作
        df["RSRS_Slope"] = np.where(var_h != 0, cov_hl / var_h, np.nan)

        # 2. 标准化 Z-Score
        rolling_slope = df["RSRS_Slope"].rolling(window=self.rsrs_m)
        slope_mean = rolling_slope.mean()
        slope_std = rolling_slope.std()

        # 同样处理标准差为 0 的情况
        df["RSRS_Zscore"] = np.where(
            slope_std != 0, (df["RSRS_Slope"] - slope_mean) / slope_std, np.nan
        )

        # 3. 计算 ATR (Average True Range)
        prev_close = df["close"].shift(1)
        # 显式计算 TR
        tr_all = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        )

        df["ATR"] = tr_all.max(axis=1).rolling(window=self.atr_period).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成信号序列。"""
        if "RSRS_Zscore" not in df.columns:
            df = self.calculate_indicators(df)

        # Pyright 现在能正确识别此处返回 tuple[np.ndarray, None]
        signals, _ = self._run_simulation_logic(
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["RSRS_Zscore"].to_numpy(),
            df["ATR"].to_numpy(),
            self.buy_threshold,
            self.sell_threshold,
            self.stop_loss_mult,
            self.effective_max_lev,
            record_trades=False,
            timestamps=df.index.to_numpy(),
        )
        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """获取交易记录。"""
        if "RSRS_Zscore" not in df.columns:
            df = self.calculate_indicators(df)

        # Pyright 现在能正确识别此处返回 list[dict[str, Any]]
        return self._run_simulation_logic(
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["RSRS_Zscore"].to_numpy(),
            df["ATR"].to_numpy(),
            self.buy_threshold,
            self.sell_threshold,
            self.stop_loss_mult,
            self.effective_max_lev,
            record_trades=True,
            timestamps=df.index.to_numpy(),
        )

    # --- 修复后的重载声明 ---

    @overload
    def _run_simulation_logic(
        self,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        rsrs_z: np.ndarray,
        atr: np.ndarray,
        buy_t: float,
        sell_t: float,
        stop_mult: float,
        max_lev: float,
        record_trades: Literal[True],
        timestamps: np.ndarray | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def _run_simulation_logic(
        self,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        rsrs_z: np.ndarray,
        atr: np.ndarray,
        buy_t: float,
        sell_t: float,
        stop_mult: float,
        max_lev: float,
        record_trades: Literal[False] = False,
        timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, None]: ...

    def _run_simulation_logic(
        self,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        rsrs_z: np.ndarray,
        atr: np.ndarray,
        buy_t: float,
        sell_t: float,
        stop_mult: float,
        max_lev: float,
        record_trades: bool = False,
        timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, None] | list[dict[str, Any]]:
        """实现具体的仿真逻辑。"""
        length = len(close_arr)
        signals = np.zeros(length)
        trade_logs: list[dict[str, Any]] = []

        position = 0.0
        entry_price = 0.0
        stop_price = 0.0
        extreme_price = 0.0
        entry_idx = 0

        for i in range(1, length):
            if np.isnan(rsrs_z[i]) or np.isnan(atr[i]):
                continue

            if position > 0:
                if high_arr[i] > extreme_price:
                    extreme_price = high_arr[i]
                    stop_price = max(stop_price, extreme_price - (stop_mult * atr[i]))

                if rsrs_z[i] < sell_t or low_arr[i] < stop_price:
                    if record_trades and timestamps is not None:
                        is_stop = low_arr[i] < stop_price
                        exit_price = stop_price if is_stop else close_arr[i]
                        trade_logs.append(
                            {
                                "entry_time": timestamps[entry_idx],
                                "exit_time": timestamps[i],
                                "reason": "StopLoss" if is_stop else "RSRS_Exit",
                                "pnl": (exit_price - entry_price) / entry_price,
                                "leverage": position,
                            }
                        )
                    position = 0.0
                signals[i] = position
            elif rsrs_z[i] > buy_t:
                position = max_lev
                entry_price = close_arr[i]
                entry_idx = i
                extreme_price = high_arr[i]
                stop_price = entry_price - (stop_mult * atr[i])
                signals[i] = position

        return trade_logs if record_trades else (signals, None)
