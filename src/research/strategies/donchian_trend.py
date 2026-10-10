"""
src/research/strategies/donchian_trend.py
经典唐奇安突破趋势策略（海龟简化版）。

入场：收盘价突破过去 N 根最高价。
出场：收盘价跌破过去 M 根最低价，或触及 ATR 追踪止损。

与自适应双均线的差别在于「没有自适应」——参数固定、不做市场状态判断、
不用 ADX 过滤，只有突破入场与追踪止损出场。这样才构成干净的对照组。
"""

from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy

DEFAULT_ENTRY_PERIOD = 55
DEFAULT_EXIT_PERIOD = 20
DEFAULT_ATR_PERIOD = 20
DEFAULT_ATR_MULT = 2.0


class DonchianTrendStrategy(BaseStrategy):
    """唐奇安突破 (Donchian Breakout)，只做多。"""

    def __init__(
        self,
        entry_period: int = DEFAULT_ENTRY_PERIOD,
        exit_period: int = DEFAULT_EXIT_PERIOD,
        atr_period: int = DEFAULT_ATR_PERIOD,
        atr_mult: float = DEFAULT_ATR_MULT,
    ) -> None:
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    @property
    @override
    def warmup_bars(self) -> int:
        """最长回看窗口再加一根，保证通道与 ATR 都已就绪。"""
        return max(self.entry_period, self.exit_period, self.atr_period) + 1

    @override
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_index()

        # shift(1)：突破只能看已完成的历史，当根的极值不参与通道计算
        df["DC_Upper"] = df["high"].rolling(self.entry_period).max().shift(1)
        df["DC_Lower"] = df["low"].rolling(self.exit_period).min().shift(1)

        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = true_range.rolling(self.atr_period).mean()

        return df

    @override
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "DC_Upper" not in df.columns:
            df = self.calculate_indicators(df)

        df["signal"] = self._run_breakout(
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["DC_Upper"].to_numpy(),
            df["DC_Lower"].to_numpy(),
            df["ATR"].to_numpy(),
        )
        return df

    def _run_breakout(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        atr: np.ndarray,
    ) -> np.ndarray:
        """逐根推进的状态机：空仓等突破，持仓等破位或追踪止损。"""
        signals = np.zeros(len(close))
        holding = False
        peak = 0.0

        for i in range(len(close)):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                continue

            if not holding:
                if close[i] > upper[i]:
                    holding = True
                    peak = high[i]
                    signals[i] = 1.0
                continue

            peak = max(peak, high[i])
            stop_hit = (
                self.atr_mult > 0
                and not np.isnan(atr[i])
                and low[i] <= peak - self.atr_mult * atr[i]
            )
            if stop_hit or close[i] < lower[i]:
                holding = False
            else:
                signals[i] = 1.0

        return signals
