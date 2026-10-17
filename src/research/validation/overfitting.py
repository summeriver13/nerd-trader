"""
多重检验校正：去膨胀夏普（DSR）与回测过拟合概率（PBO）。

回答的问题是：在「一共试过 N 个候选」的前提下，这个策略的夏普还能否算作技能？
试得越多，最好的那个看起来越漂亮；不做校正的夏普没有决策价值。

参考：Bailey & López de Prado (2014) The Deflated Sharpe Ratio；
      Bailey, Borwein, López de Prado & Zhu (2017) The Probability of Backtest Overfitting。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.analytics.metrics import sharpe_ratio
from src.research.study.track_record import sharpe_z_score
from src.research.validation.overlap import embargo_from_ratio, purge_mask

EULER_MASCHERONI = 0.5772156649015329
DSR_THRESHOLD = 0.95  # 去膨胀夏普门槛：低于它无法与「试 N 次碰运气」区分
PBO_THRESHOLD = 0.50  # 过拟合概率门槛
DEFAULT_BLOCKS = 10  # CSCV 分块数（须为偶数）
MIN_BLOCKS = 4
MIN_BARS_PER_BLOCK = 60
MIN_CANDIDATES = 2
# DSR 至少需要这么多个观测：3 个观测算出的 DSR 没有意义，应报「样本不足」而不是 0
MIN_DSR_OBSERVATIONS = 30

_NORMAL = NormalDist()


def expected_max_sharpe(variance_of_sharpe: float, trials: int) -> float:
    """零假设（无技能）下 N 次试验的夏普最大值期望。"""
    if variance_of_sharpe <= 0 or trials < MIN_CANDIDATES:
        return 0.0
    z_first = _NORMAL.inv_cdf(1 - 1 / trials)
    z_last = _NORMAL.inv_cdf(1 - 1 / (trials * math.e))
    weight = EULER_MASCHERONI
    return float(
        math.sqrt(variance_of_sharpe) * ((1 - weight) * z_first + weight * z_last)
    )


@dataclass(frozen=True)
class SharpeTest:
    """单个候选的去膨胀夏普检验结果。"""

    label: str
    sharpe: float
    expected_max: float
    dsr: float
    observations: int

    @property
    def sufficient(self) -> bool:
        """观测数是否够算 DSR（不够时报「样本不足」，不能当成「未通过」）。"""
        return self.observations >= MIN_DSR_OBSERVATIONS

    @property
    def passed(self) -> bool:
        return self.sufficient and self.dsr >= DSR_THRESHOLD


@dataclass(frozen=True)
class PboResult:
    """CSCV 回测过拟合概率：样本内冠军在样本外跌到中位数以下的比例。"""

    blocks: int
    splits: int
    pbo: float
    logit_median: float
    winner_oos_sharpe_median: float
    purge: int = 0
    embargo: int = 0  # 已折算成观测数

    @property
    def passed(self) -> bool:
        return self.pbo < PBO_THRESHOLD


@dataclass(frozen=True)
class OverfittingReport:
    """一次稳健性检验的多重检验校正结果。"""

    trials: int
    observations: int
    sharpe_tests: list[SharpeTest]
    pbo: PboResult | None


def sharpe_estimator_variance(sharpes: list[float], observations: int) -> float:
    """零假设下夏普估计量的方差，取「候选间离散度」与「估计量方差」的较大者。

    候选高度相关（甚至完全相同）时离散度≈ 0，会把门槛清零、让 DSR 恒过；
    估计量方差 (1 + 0.5·SR²)/T 不依赖候选之间的差异，是这个门槛的下界。
    """
    dispersion = (
        float(np.var(sharpes, ddof=1)) if len(sharpes) >= MIN_CANDIDATES else 0.0
    )
    if observations <= 1 or not sharpes:
        return dispersion
    estimator = (1 + 0.5 * float(np.mean(sharpes)) ** 2) / observations
    return max(dispersion, estimator)


def deflated_sharpe(label: str, returns: pd.Series, expected_max: float) -> SharpeTest:
    """把观测夏普与零假设下的期望最大夏普比较，输出 DSR 概率。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    sr = sharpe_ratio(arr)
    if arr.size < MIN_DSR_OBSERVATIONS:
        return SharpeTest(label, sr, expected_max, float("nan"), arr.size)

    series = pd.Series(arr)
    z_score = sharpe_z_score(
        sr, arr.size, float(series.skew()), float(series.kurt()) + 3.0, expected_max
    )
    if z_score is None:
        return SharpeTest(label, sr, expected_max, 0.0, arr.size)
    return SharpeTest(label, sr, expected_max, float(_NORMAL.cdf(z_score)), arr.size)


def _resolve_blocks(bars: int, requested: int) -> int | None:
    """分块数须为偶数，且每块不少于 MIN_BARS_PER_BLOCK 根，否则不做 PBO。"""
    limit = bars // MIN_BARS_PER_BLOCK
    blocks = min(requested, limit if limit % 2 == 0 else limit - 1)
    return blocks if blocks >= MIN_BLOCKS else None


def _is_degenerate(frame: pd.DataFrame) -> bool:
    """候选是否退化：各候选逐根取值完全相同（相关 = 1）。

    此时样本内冠军与其余候选没有差别，分位秩无意义，PBO 必须报「未定义」
    而不是 0%（0% 会被读成「没有过拟合风险」）。
    """
    values = frame.to_numpy(dtype=float)
    spread = values.max(axis=1) - values.min(axis=1)
    scale = float(np.abs(values).mean()) or 1.0
    return bool(np.all(spread <= 1e-9 * scale))


def probability_of_backtest_overfitting(
    series_map: dict[str, pd.Series],
    blocks: int = DEFAULT_BLOCKS,
    purge: int = 0,
    embargo: float = 0.0,
) -> PboResult | None:
    """CSCV：把时间轴等分 S 块，枚举全部「一半样本内 / 一半样本外」切分。

    `purge` 与 `embargo` 都用来堵住测试期信息漏进训练集，但口径不同（见
    `src/research/validation/overlap.py`）：purge 是**标签跨几个观测**——逐根采样 + h 根标签
    时 purge=h，非重叠采样（每 h 根取一个观测）时标签首尾相接、purge=0；
    embargo 是**占观测数的比例**，防的是序列相关而非标签重叠，作者取 0.01。
    不剔除会让 PBO 偏乐观。
    """
    frame = pd.DataFrame(series_map).dropna(how="any")
    if frame.shape[1] < MIN_CANDIDATES or frame.empty:
        return None
    if _is_degenerate(frame):
        return None
    effective = _resolve_blocks(len(frame), blocks)
    if effective is None:
        return None

    values = frame.to_numpy(dtype=float)
    embargo_bars = embargo_from_ratio(values.shape[0], embargo)
    bounds = np.linspace(0, values.shape[0], effective + 1).astype(int)
    rows = [np.arange(bounds[i], bounds[i + 1]) for i in range(effective)]
    half = effective // 2
    logits: list[float] = []
    winner_oos: list[float] = []

    for combo in combinations(range(effective), half):
        chosen = set(combo)
        is_test = np.zeros(values.shape[0], dtype=bool)
        for i in range(effective):
            if i not in chosen:
                is_test[rows[i]] = True
        keep = purge_mask(is_test, purge, embargo_bars)
        in_sample = np.concatenate(
            [rows[i][keep[rows[i]]] for i in range(effective) if i in chosen]
        )
        out_sample = np.flatnonzero(is_test)
        if in_sample.size < MIN_DSR_OBSERVATIONS:
            continue
        is_sharpe = np.array(
            [sharpe_ratio(values[in_sample, j]) for j in range(frame.shape[1])]
        )
        oos_sharpe = np.array(
            [sharpe_ratio(values[out_sample, j]) for j in range(frame.shape[1])]
        )
        winner = int(np.argmax(is_sharpe))
        # 平局（候选完全相同）取中位秩：直接数「≤ 冠军」会把冠军判成第一名
        less = float((oos_sharpe < oos_sharpe[winner]).sum())
        equal = float((oos_sharpe == oos_sharpe[winner]).sum())
        rank = less + (equal + 1) / 2  # 1..N 的中位秩
        percentile = rank / (frame.shape[1] + 1)
        logits.append(math.log(percentile / (1 - percentile)))
        winner_oos.append(float(oos_sharpe[winner]))

    logit_arr = np.asarray(logits)
    if logit_arr.size == 0:
        return None
    return PboResult(
        blocks=effective,
        splits=int(logit_arr.size),
        pbo=float((logit_arr < 0).mean()),
        logit_median=float(np.median(logit_arr)),
        winner_oos_sharpe_median=float(np.median(winner_oos)),
        purge=purge,
        embargo=embargo_bars,
    )


def build_overfitting_report(
    series_map: dict[str, pd.Series],
    trials: int | None = None,
    blocks: int = DEFAULT_BLOCKS,
    purge: int = 0,
    embargo: float = 0.0,
) -> OverfittingReport:
    """对一组候选（同一份数据上的全部试验）做 DSR 与 PBO。"""
    frame = pd.DataFrame(series_map).dropna(how="any")
    sharpes = [sharpe_ratio(series) for series in series_map.values()]
    variance = sharpe_estimator_variance(sharpes, len(frame))
    trial_count = max(trials or len(series_map), MIN_CANDIDATES)
    expected_max = expected_max_sharpe(variance, trial_count)
    tests = [
        deflated_sharpe(label, series, expected_max)
        for label, series in series_map.items()
    ]
    return OverfittingReport(
        trials=trial_count,
        observations=len(frame),
        sharpe_tests=tests,
        pbo=probability_of_backtest_overfitting(series_map, blocks, purge, embargo),
    )


def print_overfitting(
    symbol: str, series_map: dict[str, pd.Series], trials: int | None = None
) -> None:
    """打印多重检验校正表：每个候选的 DSR 与整组候选的 PBO。"""
    if not series_map:
        return
    report = build_overfitting_report(series_map, trials)

    print(f"\n多重检验校正（{symbol}）")
    if trials is None:
        print(
            f"试验集合 = 本符号上竞争的 {len(series_map)} 个候选；"
            "若历史上还试过其他参数/窗口，请用 --trials 声明真实试验次数。"
        )
    else:
        print(
            f"试验集合 = {len(series_map)} 个候选，声明的历史试验次数 N={report.trials}。"
        )
    print(
        f"{'策略':<26}{'夏普(非年化)':>13}{'零假设期望最大':>15}{'DSR':>9}{'判定':>7}"
    )
    print("-" * 118)
    for test in report.sharpe_tests:
        if not test.sufficient:
            verdict = "样本不足"
        else:
            verdict = "通过" if test.passed else "未通过"
        print(
            f"{test.label:<26}{test.sharpe:>12.4f} {test.expected_max:>14.4f} "
            f"{test.dsr:>8.3f} {verdict:>8}"
        )
    if any(not test.sufficient for test in report.sharpe_tests):
        print(
            f"（观测数少于 {MIN_DSR_OBSERVATIONS} 根的候选无法算 DSR，"
            "「样本不足」不等于「未通过」）"
        )

    pbo = report.pbo
    if pbo is None:
        print(
            "\nPBO：未计算（样本不足、候选过少，或候选之间完全相同——"
            "后者的分位秩无意义，0% 会被误读成「没有过拟合风险」）。"
        )
        return
    print(
        f"\nPBO (CSCV, S={pbo.blocks}, {pbo.splits} 次切分) = {pbo.pbo:.1%}"
        f" | 中位 logit = {pbo.logit_median:+.2f}"
        f" | 样本内冠军的样本外夏普中位 = {pbo.winner_oos_sharpe_median:.4f}"
    )
    print(
        f"门槛：DSR ≥ {DSR_THRESHOLD:.2f} 且 PBO < {PBO_THRESHOLD:.0%}"
        " —— 未过门槛意味着该结果无法与「试 N 次碰运气」区分。"
    )
