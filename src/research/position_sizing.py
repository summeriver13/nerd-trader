"""
仓位缩放层：把策略的 0/1 信号换算成按波动率目标调节的 0~100% 暴露。

与 alpha 无关：只把「给定风险预算」翻译成仓位。标的年化波动率越高仓位越小，
使实际承担的风险稳定在设定水平附近；换手成本由 rebalance_every 控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

ANNUAL_DAYS = 365
DEFAULT_VOL_TARGET = 0.40
DEFAULT_VOL_LOOKBACK = 30
DEFAULT_REBALANCE_BARS = 7  # 日线上一周调一次，避免每日再平衡吃掉收益


@dataclass(frozen=True)
class VolatilityTarget:
    """波动率目标参数。"""

    target_vol: float = DEFAULT_VOL_TARGET
    lookback: int = DEFAULT_VOL_LOOKBACK
    rebalance_every: int = DEFAULT_REBALANCE_BARS
    max_exposure: float = 1.0

    def exposure(self, close: pd.Series) -> pd.Series:
        """按滚动实现波动率给出仓位系数（0 ~ max_exposure）。"""
        realized = close.pct_change().rolling(self.lookback).std() * np.sqrt(ANNUAL_DAYS)
        scale = (self.target_vol / realized.replace(0.0, np.nan)).clip(
            lower=0.0, upper=self.max_exposure
        )
        return _hold_until_rebalance(scale.ffill().fillna(0.0), self.rebalance_every)


def _hold_until_rebalance(exposure: pd.Series, rebalance_every: int) -> pd.Series:
    """每隔 rebalance_every 根才更新一次仓位，其余时间保持不变。"""
    if rebalance_every <= 1:
        return exposure
    values = exposure.to_numpy(dtype=float)
    anchors = np.arange(values.size) // rebalance_every * rebalance_every
    return pd.Series(values[anchors], index=exposure.index)


def apply_volatility_target(df: pd.DataFrame, sizing: VolatilityTarget) -> pd.DataFrame:
    """把 0/1 信号缩放成 0~max_exposure 的仓位，方向不变。"""
    out = df.copy()
    close = cast(pd.Series, out["close"])
    out["signal"] = out["signal"] * sizing.exposure(close)
    return out
