"""
回测记录质量（de Prado 第 14/15 章）：PSR、最小记录长度、回撤与水下时间、收益集中度、策略风险。

DSR/PBO 回答的是「试了这么多次，这个夏普还能算技能吗」；这批统计量回答它回答不了的两件事：

1. **记录本身厚不厚**：夏普的概率显著性（PSR）、达到该显著性所需的最小记录长度
   （MinTRL）、最大回撤与最长水下时间，以及**收益在各笔下注上的集中度（HHI）**。
   一条靠单笔暴利撑起来的记录，夏普再高也不可复制——本仓库的收益分布极端重尾，
   这是最该看的诊断。
2. **这份记录需不需要靠「赢少赔多」**：见 `strategy_risk.py`（第 15 章）——那边的输入是
   下注规则参数（止损/止盈/下注频率），不是收益序列。

口径与原文出处（逐条核到作者 Ch 14/15 片段）：

- **集中度**（Snippet 14.3）：`w = r / Σr`，`HHI = Σw²`，再标准化成
  `(HHI − 1/N) / (1 − 1/N)`——均匀下注给 0、只有一笔非零给 1，少于三笔给 nan。
  原文对正负收益各算一次；负收益那支因 Σr<0 使权重仍为正，等价于对 |r| 算。
- **回撤**（Snippet 14.4）：高水位是净值/累计 P&L 的扩张最大值，每个高水位到下一次
  创新高之间记一次回撤，水下时间＝从高水位到收复它的年数（末段算到观测结束）。
  **原文的相对回撤 `1 − min / hwm` 只对恒正净值成立**；本仓策略 P&L 会走负，故按
  P&L 自身单位算（等价于原文 `dollars=True` 那支）。
- **PSR**（Snippet 14.5）：`Φ((SR − SR*)·√(n−1) / √(1 − γ3·SR + (γ4−1)/4·SR²))`，
  SR 是**非年化**夏普，峰度用非超额口径。
- **MinTRL**：`1 + (1 − γ3·SR + (γ4−1)/4·SR²)·(Z_{1−α} / (SR − SR*))²`，单位是观测数。
  原文没防 `SR ≤ SR*`（此时公式仍给出正数但没有意义），本仓返回 inf。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

from src.analytics.metrics import sharpe_ratio
from src.research.sizing import PERIODS_PER_YEAR

# 集中度至少要三笔才有内容（原文口径：不足三笔给 nan，而不是 0）
MIN_CONCENTRATION_BETS: Final[int] = 3
# PSR / MinTRL 至少要这么多观测，否则偏度与峰度本身没有意义
MIN_TRACK_OBSERVATIONS: Final[int] = 30
DEFAULT_CONFIDENCE: Final[float] = 0.95

_NORMAL = NormalDist()

# 这些函数的输入是「一串收益」：Series、ndarray、列表都行（内部统一 asarray）
type ReturnSeries = pd.Series | np.ndarray | Sequence[float]


def _finite(values: ReturnSeries) -> np.ndarray:
    """去掉 nan / inf，统一成 float 数组。"""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


# === 收益集中度（Snippet 14.3）===


def concentration(returns: ReturnSeries) -> float:
    """各笔下注收益的归一化 HHI：0 = 完全均匀，1 = 只有一笔非零。

    **输入必须同号**（全非负或全非正）。归一化前的权重是 `r / Σr`，只有同号时权重
    才落在 [0, 1] 内、结果才保证落在 [0, 1]；混号会让权重超过 1 甚至变负，实测
    `[0.5, −0.3, 0.2, −0.1, 0.05]` 给出 3.755 这种假数。原文的对策是把正负分成
    两支算（`all_bets_concentration`），本仓取同一个意思：混号直接报错。
    """
    arr = _finite(returns)
    if arr.size and float(arr.min()) < 0.0 < float(arr.max()):
        raise ValueError("集中度只在同号收益上有意义；混号请用 concentration_by_sign")
    if arr.size < MIN_CONCENTRATION_BETS:
        return float("nan")
    total = float(arr.sum())
    if total == 0.0:
        return float("nan")
    weights = arr / total
    hhi = float((weights**2).sum())
    inverse = 1.0 / arr.size
    return (hhi - inverse) / (1.0 - inverse)


def concentration_by_sign(returns: ReturnSeries) -> tuple[float, float]:
    """正收益与负收益各自的集中度（原文把两类分开算的那两支）。

    混号收益的正确入口就是它：同号子集内部才满足上面那条前提。
    """
    arr = _finite(returns)
    return concentration(arr[arr >= 0]), concentration(arr[arr < 0])


# === 回撤与水下时间（Snippet 14.4）===


def drawdown_and_time_under_water(
    level: pd.Series, periods_per_year: float = PERIODS_PER_YEAR
) -> tuple[pd.Series, pd.Series]:
    """回撤与水下时间，输入是**净值/累计 P&L 序列**（不是逐期收益）。

    返回两个 Series，索引都是高水位的发生时刻：回撤深度（P&L 单位）与水下时间（年）。
    还没收复的那个高水位算到观测结束，与原文一致。
    """
    series = pd.Series(level, dtype=float).dropna()
    if series.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    values = series.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(values)
    new_peak = np.concatenate(([True], peaks[1:] > peaks[:-1]))
    starts = np.flatnonzero(new_peak)
    ends = np.append(starts[1:], values.size)
    depths: list[float] = []
    spans: list[float] = []
    positions: list[int] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        deepest = float(values[start:end].min())
        if deepest < peaks[start]:
            depths.append(float(peaks[start]) - deepest)
            spans.append((end - start) / periods_per_year)
            positions.append(int(start))
    if not depths:
        empty = pd.Series(dtype=float)
        return empty, empty
    index = series.index[np.asarray(positions)]
    return pd.Series(depths, index=index), pd.Series(spans, index=index)


def max_drawdown(level: pd.Series) -> float:
    """最大回撤（P&L 单位；单调不亏给 0）。"""
    depths, _ = drawdown_and_time_under_water(level)
    return float(depths.max()) if not depths.empty else 0.0


def longest_time_under_water(
    level: pd.Series, periods_per_year: float = PERIODS_PER_YEAR
) -> float:
    """最长水下时间（年；从未回过撤给 0）。"""
    _, spans = drawdown_and_time_under_water(level, periods_per_year)
    return float(spans.max()) if not spans.empty else 0.0


# === 概率夏普与最小记录长度（Snippet 14.5）===


def sharpe_z_score(
    sharpe: float, observations: int, skew: float, kurtosis: float, benchmark: float
) -> float | None:
    """PSR 的检验统计量；非正态性把方差压到非正时返回 None（原文此处会得到复数）。"""
    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if observations < 2 or denominator <= 0:
        return None
    return (sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(denominator)


def _moments(arr: np.ndarray) -> tuple[float, float]:
    """样本偏度与**非超额**峰度（原文口径）。"""
    series = pd.Series(arr)
    return float(series.skew()), float(series.kurt()) + 3.0


def probabilistic_sharpe_ratio(returns: ReturnSeries, benchmark: float = 0.0) -> float:
    """夏普高于 benchmark 的概率（Snippet 14.5）；观测不足或方差非正给 nan。"""
    arr = _finite(returns)
    if arr.size < MIN_TRACK_OBSERVATIONS:
        return float("nan")
    skew, kurtosis = _moments(arr)
    z_score = sharpe_z_score(sharpe_ratio(arr), arr.size, skew, kurtosis, benchmark)
    return float("nan") if z_score is None else float(_NORMAL.cdf(z_score))


def minimum_track_record_length(
    returns: ReturnSeries,
    benchmark: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """达到给定置信度所需的最小观测数（原文 minTRL）。

    观测夏普不高于 benchmark 时给 inf——这不是「需要更久」，而是「这么久的记录
    也证明不了任何事」，不能当成一个很大的数来比较。
    """
    arr = _finite(returns)
    if arr.size < MIN_TRACK_OBSERVATIONS:
        return float("nan")
    observed = sharpe_ratio(arr)
    if observed <= benchmark:
        return float("inf")
    skew, kurtosis = _moments(arr)
    spread = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if spread <= 0:
        return float("inf")
    quantile = _NORMAL.inv_cdf(confidence)
    return 1.0 + spread * (quantile / (observed - benchmark)) ** 2


# === 一次算齐（报告用）===


@dataclass(frozen=True)
class TrackRecord:
    """一条策略记录的 Ch 14 体检结果。回撤与集中度都在**逐期收益**上算。"""

    label: str
    observations: int
    sharpe: float
    psr: float
    min_track_record: float
    max_drawdown: float
    longest_time_under_water: float
    positive_concentration: float
    negative_concentration: float

    @property
    def sufficient(self) -> bool:
        """观测数够不够算 PSR/MinTRL（不够时报「样本不足」，不能当成未通过）。"""
        return self.observations >= MIN_TRACK_OBSERVATIONS

    def describe(self) -> str:
        """报告里的一行说明。"""
        if not self.sufficient:
            return f"{self.label}：样本不足（{self.observations} < {MIN_TRACK_OBSERVATIONS}）"
        track = (
            "inf（夏普未过基准）"
            if math.isinf(self.min_track_record)
            else f"{self.min_track_record:.0f}"
        )
        return (
            f"{self.label}：夏普 {self.sharpe:+.3f}｜PSR {self.psr:.3f}｜"
            f"MinTRL {track}｜最大回撤 {self.max_drawdown:.1f}｜"
            f"最长水下 {self.longest_time_under_water:.2f} 年｜"
            f"集中度 正 {self.positive_concentration:.3f} / 负 {self.negative_concentration:.3f}"
        )


def track_record(
    label: str,
    returns: ReturnSeries,
    benchmark: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> TrackRecord:
    """把 Ch 14 的体检项一次算齐；回撤在累计 P&L 上算（逐期收益内部累加）。"""
    series = pd.Series(returns, dtype=float).dropna()
    level = series.cumsum()
    positive, negative = concentration_by_sign(series)
    return TrackRecord(
        label=label,
        observations=int(series.size),
        sharpe=sharpe_ratio(series.to_numpy(dtype=float)),
        psr=probabilistic_sharpe_ratio(series, benchmark),
        min_track_record=minimum_track_record_length(series, benchmark, confidence),
        max_drawdown=max_drawdown(level),
        longest_time_under_water=longest_time_under_water(level, periods_per_year),
        positive_concentration=positive,
        negative_concentration=negative,
    )
