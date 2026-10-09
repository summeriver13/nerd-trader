"""
src/research/strategies/buy_and_hold.py
基准策略：始终满仓持有，任何策略必须先打败它。
"""

from __future__ import annotations

from typing import override

import pandas as pd

from src.research.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """买入持有基准：不算指标、不设仓位变化，持仓恒为 1。"""

    @override
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @override
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 1.0
        return df
