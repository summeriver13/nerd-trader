"""
src/utils/bar_quality.py
K 线数值合理性校验：不可能的 K 线丢弃，可疑尖峰只告警。

已知污染（实测）：ccxt 的 OKX 路径会把别处的值塞进 high/close，给出 H=3616 万、
L=0.1、C=13.3 万这类不可能的 K 线，而同日 REST history-candles 是干净的——是取数
路径的问题，不是交易所的问题。真实行情（上市暴涨、连环爆仓）是连续多根，只有污染
才会孤立单根跳 5 倍，但两者本地无法区分，所以尖峰只告警：确认要靠
scripts/crosscheck_crypto_bars.py 对账。
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SPIKE_WINDOW: Final[int] = 11  # 中心邻域根数
SPIKE_RATIO: Final[float] = 5.0  # 偏离邻域中位数此倍数即报可疑
WARN_SAMPLE: Final[int] = 5  # 告警里最多列出几个时间戳


def impossible_bars(df: pd.DataFrame) -> pd.Series:
    """数值上不可能成立的 K 线：非正价、高低倒挂、包络违反、负成交量、缺失值。

    包络（high ≥ max(open, close) ≥ min(open, close) ≥ low）是 OHLC 的定义，
    违反它只可能是数据源混了字段，不是行情。
    """
    op, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
    return (
        (op <= 0)
        | (hi <= 0)
        | (lo <= 0)
        | (cl <= 0)
        | (hi < lo)
        | (hi < np.maximum(op, cl))
        | (lo > np.minimum(op, cl))
        | (df["volume"] < 0)
        | op.isna()
        | hi.isna()
        | lo.isna()
        | cl.isna()
    )


def suspicious_spikes(df: pd.DataFrame) -> pd.Series:
    """可疑尖峰：单根 high/low/close 偏离中心邻域中位数 SPIKE_RATIO 倍以上。

    边缘（窗口不足 SPIKE_WINDOW 根）不判定，避免把上市首根当尖峰。
    """
    med = (
        df["close"]
        .rolling(SPIKE_WINDOW, center=True, min_periods=SPIKE_WINDOW)
        .median()
    )
    far = (
        (df["high"] > med * SPIKE_RATIO)
        | (df["low"] < med / SPIKE_RATIO)
        | (df["close"] > med * SPIKE_RATIO)
        | (df["close"] < med / SPIKE_RATIO)
    )
    return far.fillna(False).astype(bool)


def sample_times(index: pd.Index, limit: int = WARN_SAMPLE) -> str:
    """告警里列出时间戳样本：光给个数没法复核，脏行位置必须可追溯。"""
    stamps = ", ".join(str(t)[:16] for t in index[:limit])
    return stamps + (f" …（共 {len(index)} 根）" if len(index) > limit else "")
