"""
src/research/factors/evaluation.py
截面因子的评估台：前向收益与 RankIC 显著性。

读结论前必须先看这两条口径：

1. **重叠样本**：日频 IC 是「每天算一次 h 根前向收益」，相邻期高度重叠。本模块
   只输出非重叠子样本的 t 值——重叠样本的 t 会把有效观测数放大 h 倍，实测
   （2026-10-09 主池）同一因子的重叠 t 是独立 t 的 √h 倍，足以把噪声读成显著。
2. **IC 显著不等于能赚钱**：IC 测的是截面排序，天然不含市场 beta；而组合收益 =
   市场 + beta × 市场 + alpha，市场年化波动（实测 104.8%）可以是超额收益的十几
   倍。判定能不能赚钱要看 portfolio.py 的复利口径与基准对比，不能只看 IC。
   非重叠组合的期数远少于 IC 口径，两者不一致时以组合口径为准。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel

# 截面太窄时 RankIC 没有意义：少于这么多标的的期次直接跳过，不参与统计
MIN_NAMES_PER_PERIOD: Final[int] = 10


@dataclass(frozen=True)
class ICStats:
    """一个因子在一个前向窗口上的 RankIC 统计。"""

    factor: str
    horizon: int
    mean: float
    std: float
    t_stat: float
    mean_independent: float
    periods: int
    independent: int
    names: float


def forward_returns(panel: Panel, horizon: int) -> pd.DataFrame:
    """h 根前向收益：全项目唯一允许「看未来」的一行（它是被解释变量，不是因子）。"""
    return panel.close.shift(-horizon) / panel.close - 1


def rank_ic_series(
    values: pd.DataFrame, forward: pd.DataFrame, mask: pd.DataFrame
) -> pd.Series:
    """逐期截面 Spearman 相关，只在可交易集合内排名。

    两类期次直接跳过：标的太少、以及因子值在截面上完全相同的期（相关系数无定义，
    算出来是 NaN 并会让 pandas 抛 ConstantInputWarning）。筛选要在算相关之前做，
    否则警告照报。
    """
    masked_values = values.where(mask)
    masked_forward = forward.where(mask)
    valid = (masked_values.notna().sum(axis=1) >= MIN_NAMES_PER_PERIOD) & (
        masked_values.nunique(axis=1) >= MIN_NAMES_PER_PERIOD
    )
    if not valid.any():
        return pd.Series(dtype=float)
    ic = masked_values[valid].corrwith(masked_forward[valid], axis=1, method="spearman")
    return ic.dropna()


def ic_stats(
    spec: FactorSpec, panel: Panel, mask: pd.DataFrame, horizon: int
) -> ICStats:
    """因子 IC 统计：点估计用重叠样本（更平滑），显著性只认非重叠子样本。"""
    ic = rank_ic_series(spec.values(panel), forward_returns(panel, horizon), mask)
    if ic.empty:
        return ICStats(spec.key, horizon, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    t_stat, independent_count = non_overlapping_t(ic, horizon)
    return ICStats(
        factor=spec.key,
        horizon=horizon,
        mean=float(ic.mean()),
        std=float(ic.std(ddof=1)),
        t_stat=t_stat,
        mean_independent=float(ic.iloc[::horizon].mean()),
        periods=int(ic.size),
        independent=independent_count,
        names=float(
            spec.values(panel).where(mask).notna().sum(axis=1).reindex(ic.index).mean()
        ),
    )


# === 秩空间的统计原语（正交化检验与 IC 表共用，口径只在这里出一次）===


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """秩相关：两边各自排名后算 Pearson（不为一次相关引入 scipy）。"""
    a = pd.Series(left).rank().to_numpy(dtype=float)
    b = pd.Series(right).rank().to_numpy(dtype=float)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(np.sqrt((a_centered**2).sum() * (b_centered**2).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denominator)


def non_overlapping_t(series: pd.Series, horizon: int) -> tuple[float, int]:
    """非重叠子样本的 t 值与独立期数：全项目唯一出处（含 IC 表与正交化表）。"""
    independent = series.iloc[::horizon]
    if independent.size < 2:
        return 0.0, int(independent.size)
    std = float(independent.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return 0.0, int(independent.size)
    t_stat = float(independent.mean() / (std / independent.size**0.5))
    return t_stat, int(independent.size)


def residual_ic_series(
    target_rank: pd.DataFrame, control_rank: pd.DataFrame, forward: pd.DataFrame
) -> pd.Series:
    """逐期残差 IC：残差 = 目标秩 − 目标秩对控制秩的 OLS 拟合值。

    控制变量在截面上完全相同时（零方差）斜率取 0，而不是丢掉这一期：去均值是单调
    变换、秩不变，所以此时残差 IC 恰好等于目标自己的 IC。丢期会让「控制变量越退化、
    样本越少」变成看不见的偏差。
    """
    target = target_rank.to_numpy(dtype=float)
    control = control_rank.to_numpy(dtype=float)
    ahead = forward.to_numpy(dtype=float)
    values: dict[pd.Timestamp, float] = {}
    for position, stamp in enumerate(target_rank.index):
        valid = (
            np.isfinite(target[position])
            & np.isfinite(control[position])
            & np.isfinite(ahead[position])
        )
        if int(valid.sum()) < MIN_NAMES_PER_PERIOD:
            continue
        y = target[position][valid]
        x = control[position][valid]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = float((x_centered**2).sum())
        slope = (
            float((x_centered * y_centered).sum()) / denominator
            if denominator > 0.0
            else 0.0
        )
        residual = y_centered - slope * x_centered
        ic = spearman(residual, ahead[position][valid])
        if np.isfinite(ic):
            values[stamp] = ic
    return pd.Series(values, dtype=float)
