"""
src/research/pairs/selection.py
形成期选对：相关预筛 + Engle-Granger 两步法 + 半衰期，分三段筛，全部只用形成期数据。

为什么是这套：

1. **协整检验是「价差会回来」的直接证据**；距离法（Gatev 2006）只看价差历史，
   无法拒绝「两只币同涨同跌但差价永不回归」的对子——那类对子在这里只会贡献成本。
2. **相关预筛必须在 ADF 之前**：主池上百只标的 ≈ 5 千对，逐个做 ADF 是几分钟的
   纯浪费；对数收益的相关性是同一件事的廉价代理，先砍掉明显不相关的。
3. **半衰期是交易可行性的硬约束**：半衰期 90 天、交易期 60 天的对子，在窗口内
   等不到回归，回测里它只会贡献成本与噪声。

选对只看形成期（`[t0 − formation, t0)`），交易期一根不进；β 与 α 也冻结在形成期
末值，不用滚动重估——滚动重估会让「价差」跟着价格漂移，回测里表现为几乎不交易的
漂亮结果，而实盘那是在追价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# AR(1) 回归至少要有这么多观测才敢报半衰期
MIN_AR_BARS: int = 30
# ADF 无法完成时记这个 p 值：保守到「一定不选」，而不是把失败当显著
ADF_FAILED_PVALUE: float = 1.0


@dataclass(frozen=True)
class Pair:
    """一对标的及其形成期参数（冻结，交易期内不变）。"""

    left: str
    right: str
    alpha: float
    beta: float
    corr: float
    adf_p: float
    half_life: float
    # 形成期价差的均值/标准差：z 的分母母体，冻结到交易期
    spread_mean: float
    spread_std: float

    @property
    def label(self) -> str:
        return f"{self.left}|{self.right}"


@dataclass(frozen=True)
class SelectionOptions:
    """选对口径。默认值取文献常见区间，稳健性另外扫。"""

    formation_bars: int = 180
    min_corr: float = 0.7
    max_adf_p: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 30.0
    # 对冲比的合理区间：β ≤ 0 的对子「空 B 对冲多 A」名不副实；β 过大时
    # 一腿名义极小，等于押注另一腿的方向
    min_beta: float = 0.1
    max_beta: float = 10.0
    top_pairs: int = 10
    min_names: int = 20

    def describe(self) -> str:
        return (
            f"形成期 {self.formation_bars} 根 | 相关 ≥{self.min_corr:.2f} | "
            f"ADF p <{self.max_adf_p:.2f} | 半衰期 {self.min_half_life:.0f}–"
            f"{self.max_half_life:.0f} 天 | β ∈ [{self.min_beta:g}, {self.max_beta:g}] | "
            f"取前 {self.top_pairs} 对"
        )


def select_pairs(
    log_close: pd.DataFrame,
    tradable: pd.DataFrame,
    t0: int,
    options: SelectionOptions,
) -> list[Pair]:
    """在形成期末尾（位置 t0）选对，只读 ``[t0 − formation, t0)`` 的数据。

    返回按 ADF p 值升序（越平稳越靠前）的至多 ``top_pairs`` 对，且**任一标的最多
    出现在一对里**：一个标的出现在多对里会让组合的「多空各半」名不副实，
    净敞口会朝着重复出现的那一侧漂。
    """
    window = log_close.iloc[t0 - options.formation_bars : t0]
    mask_window = tradable.iloc[t0 - options.formation_bars : t0]
    eligible = [
        symbol
        for symbol in log_close.columns
        if bool(mask_window[symbol].to_numpy(dtype=bool).all())
    ]
    if len(eligible) < options.min_names:
        return []

    window = window[eligible]
    candidates = _candidate_pairs(window, options.min_corr)
    scored = [
        pair
        for pair in (
            _score(window, left, right, corr) for left, right, corr in candidates
        )
        if pair is not None and passes_filters(pair, options)
    ]
    ranked = sorted(scored, key=lambda pair: (pair.adf_p, -pair.corr))
    return _greedy_dedupe(ranked, options.top_pairs)


def _candidate_pairs(
    window: pd.DataFrame, min_corr: float
) -> list[tuple[str, str, float]]:
    """对数收益的相关预筛：返回 (left, right, corr)，只保留上三角。"""
    returns = window.diff().iloc[1:]
    corr = returns.corr()
    out: list[tuple[str, str, float]] = []
    columns = list(window.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.at[left, right])
            if np.isfinite(value) and value >= min_corr:
                out.append((left, right, value))
    return out


def _score(window: pd.DataFrame, left: str, right: str, corr: float) -> Pair | None:
    """对一对候选做 EG 两步法 + 半衰期，β 非有限时返回 None。"""
    alpha, beta, spread = fit_spread(
        cast(pd.Series, window[left]), cast(pd.Series, window[right])
    )
    if not np.isfinite(beta):
        return None
    return Pair(
        left=left,
        right=right,
        alpha=alpha,
        beta=beta,
        corr=corr,
        adf_p=adf_pvalue(spread),
        half_life=half_life_of(spread),
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=1)),
    )


def _greedy_dedupe(ranked: list[Pair], top_pairs: int) -> list[Pair]:
    """按排名贪心取对，每个标的最多出现一次。"""
    used: set[str] = set()
    chosen: list[Pair] = []
    for pair in ranked:
        if pair.left in used or pair.right in used:
            continue
        chosen.append(pair)
        used.update((pair.left, pair.right))
        if len(chosen) >= top_pairs:
            break
    return chosen


def fit_spread(
    log_left: pd.Series, log_right: pd.Series
) -> tuple[float, float, pd.Series]:
    """EG 第一步：OLS ``log_left = alpha + beta·log_right + eps``，返回 (α, β, eps)。"""
    design = sm.add_constant(log_right.to_numpy(dtype=float))
    result = sm.OLS(log_left.to_numpy(dtype=float), design).fit()
    alpha = float(result.params[0])
    beta = float(result.params[1])
    return alpha, beta, log_left - (alpha + beta * log_right)


def adf_pvalue(spread: pd.Series) -> float:
    """EG 第二步：残差的 ADF 平稳性 p 值（带常数项、AIC 自动定阶）。

    检验跑不动时返回 1.0（`ADF_FAILED_PVALUE`）：把「没测出来」和「不平稳」归到
    同一档，宁可漏选也不误选。
    """
    values = spread.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < MIN_AR_BARS:
        return ADF_FAILED_PVALUE
    try:
        # 显式 result_object=False：statsmodels 0.15 已宣告「下次改返对象」的警告，
        # 这里现在就锁定元组返回，避免将来升级后 [1] 静默变成别的字段
        return float(
            adfuller(values, regression="c", autolag="AIC", result_object=False)[1]
        )
    except (ValueError, np.linalg.LinAlgError):
        return ADF_FAILED_PVALUE


def half_life_of(spread: pd.Series) -> float:
    """由 AR(1) 估半衰期：``Δε = λ·ε_{t−1} + c`` → ``−ln2 / λ``。

    λ ≥ 0（不回归）返回 inf，由调用方的上限拦掉；这不影响 ADF 的判定，两者是
    独立的证据。
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    if delta.size < MIN_AR_BARS:
        return float("inf")
    design = sm.add_constant(lagged.to_numpy(dtype=float))
    result = sm.OLS(delta.to_numpy(dtype=float), design).fit()
    lam = float(result.params[1])
    if not np.isfinite(lam) or lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


def passes_filters(pair: Pair, options: SelectionOptions) -> bool:
    """选对的硬条件：对冲比合理 + 价差可标准化 + ADF 显著 + 半衰期在交易可行区间。"""
    return (
        beta_ok(pair, options)
        and pair.adf_p < options.max_adf_p
        and options.min_half_life <= pair.half_life <= options.max_half_life
    )


def beta_ok(pair: Pair, options: SelectionOptions) -> bool:
    """数值安全条件（不构成选对优势）：β 在对冲比区间内、价差标准差有限且为正。

    随机对照必须保留这一条——否则 β ≈ −1 的对子会让 ``1/(1+β)`` 爆掉，
    对照就变成了数值事故而不是「随机选对」的公平比较。
    """
    return (
        options.min_beta <= pair.beta <= options.max_beta
        and np.isfinite(pair.spread_std)
        and pair.spread_std > 0
    )


def random_pairs(
    log_close: pd.DataFrame,
    tradable: pd.DataFrame,
    t0: int,
    options: SelectionOptions,
    seed: int,
) -> list[Pair]:
    """同池随机选对：证伪用的安慰剂组。

    与 ``select_pairs`` 只差一处——**不查 ADF、不查半衰期**，其余（同池、同数量、
    同 β/α/价差均值与标准差）完全一样。所以两组差异只能来自「协整筛选」本身。
    """
    window = log_close.iloc[t0 - options.formation_bars : t0]
    mask_window = tradable.iloc[t0 - options.formation_bars : t0]
    eligible = [
        symbol
        for symbol in log_close.columns
        if bool(mask_window[symbol].to_numpy(dtype=bool).all())
    ]
    if len(eligible) < options.min_names:
        return []
    window = window[eligible]
    returns = window.diff().iloc[1:]
    corr = returns.corr()
    shuffled = list(np.random.default_rng(seed + t0).permutation(eligible))
    out: list[Pair] = []
    for i in range(0, len(shuffled) - 1, 2):
        if len(out) >= options.top_pairs:
            break
        left, right = shuffled[i], shuffled[i + 1]
        pair = _score(window, left, right, float(corr.at[left, right]))
        if pair is not None and beta_ok(pair, options):
            out.append(pair)
    return out
