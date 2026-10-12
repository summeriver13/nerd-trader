"""
策略风险（de Prado 第 15 章）：给定下注规则与目标夏普，反解所需胜率、下注次数与失败概率。

与第 14 章（`track_record.py`）的分工：那边衡量「这条记录本身厚不厚」，这边回答
「这个玩法在结构上能不能成立」——胜率、盈亏比、下注频率三者是绑死的，改一个就得动另外两个。

口径与原文出处（逐条核到作者 Ch 15 片段）：

- **`binSR`**：由 (sl, pt, freq, tSR) 解二次方程得所需最低胜率。直觉是「盈亏比越差、
  越想拿高夏普，所需胜率越高；每年下注越多，所需胜率越低」——最后这条是「广度换精度」。
- **`binFreq`**：反解每年需要的下注次数。原文自带增根检验（解出的频率必须能反解回
  同一目标夏普，否则返回 None）。
- **`probFailure`**：用 `Φ(p_required; p, p(1−p))` 给出策略失败概率——原文自注为
  「bootstrap 近似」，且把 `p(1−p)` 直接当**标准差**（按抽样理论，比例的标准误是
  `√(p(1−p)/n)`，比它小得多）。本仓照原文实现、不冒充精确值：要精确值应当自己
  bootstrap 用 `required_precision` 做检验。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

_NORMAL = NormalDist()

# 抽样估计胜率至少要这么多笔，否则「胜率」本身就是噪声
MIN_BETS: Final[int] = 30

# 这些函数的输入是「一串收益」：Series、ndarray、列表都行（内部统一 asarray）
type ReturnSeries = pd.Series | np.ndarray | Sequence[float]


def _finite(values: ReturnSeries) -> np.ndarray:
    """去掉 nan / inf，统一成 float 数组。"""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def required_precision(
    stop_loss: float,
    profit_target: float,
    bets_per_year: float,
    target_sharpe: float,
) -> float:
    """达到目标年化夏普所需的最低胜率（原文 binSR）；无解时给 nan。

    止损/止盈按收益单位给（例如 −0.05 与 +0.05）。
    """
    spread = profit_target - stop_loss
    if spread <= 0 or bets_per_year <= 0:
        raise ValueError("止盈必须高于止损、每年下注数必须为正")
    coefficient_a = (bets_per_year + target_sharpe**2) * spread**2
    coefficient_b = (
        2.0 * bets_per_year * stop_loss - target_sharpe**2 * spread
    ) * spread
    coefficient_c = bets_per_year * stop_loss**2
    discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
    if discriminant < 0:
        return float("nan")
    return (-coefficient_b + math.sqrt(discriminant)) / (2.0 * coefficient_a)


def required_bets_per_year(
    stop_loss: float,
    profit_target: float,
    precision: float,
    target_sharpe: float,
) -> float | None:
    """给定胜率下达到目标年化夏普所需的每年下注次数（原文 binFreq）。

    反解不出来（增根）时返回 None——这不是「需要 0 次」也不是「需要无穷次」。

    原文的检验写作 `isclose(binSR(sl, pt, freq, p), tSR)`：左边是**胜率**、右边是
    **夏普**，量纲不同，几乎永远不成立（书里的 `main()` 走 probFailure，恰好绕过这
    一行）。本仓按它的本意改成「反解回的胜率必须等于给定胜率」——这样才真能挡住增根。
    """
    spread = profit_target - stop_loss
    if spread <= 0 or not 0.0 < precision < 1.0:
        raise ValueError("止盈必须高于止损、胜率必须在 (0, 1) 内")
    frequency = (
        (target_sharpe * spread) ** 2
        * precision
        * (1.0 - precision)
        / (spread * precision + stop_loss) ** 2
    )
    if frequency <= 0:
        return None
    recovered = required_precision(stop_loss, profit_target, frequency, target_sharpe)
    if not math.isfinite(recovered) or not math.isclose(
        recovered, precision, rel_tol=1e-9, abs_tol=1e-12
    ):
        return None
    return float(frequency)


def strategy_failure_probability(
    returns: ReturnSeries, bets_per_year: float, target_sharpe: float
) -> float:
    """策略失败概率（原文 probFailure）：实际胜率离「所需胜率」有多远。

    胜率按样本里正收益的占比估计，正负收益的均值当作止盈/止损。样本不足
    `MIN_BETS` 笔、或只有单边收益（全正或全负，算不出盈亏比）时给 nan。
    """
    arr = _finite(returns)
    if arr.size < MIN_BETS:
        return float("nan")
    winners = arr[arr > 0]
    losers = arr[arr <= 0]
    if winners.size == 0 or losers.size == 0:
        return float("nan")
    precision = winners.size / arr.size
    threshold = required_precision(
        float(losers.mean()), float(winners.mean()), bets_per_year, target_sharpe
    )
    if not math.isfinite(threshold):
        return float("nan")
    # 原文把 p(1−p) 当标准差用（自注为 bootstrap 近似），此处照原文，见模块文档
    deviation = precision * (1.0 - precision)
    if deviation <= 0:
        return float("nan")
    # NormalDist 的标准差是构造参数，不像 scipy 那样能逐次传 loc/scale，故自己标准化
    return float(_NORMAL.cdf((threshold - precision) / deviation))
