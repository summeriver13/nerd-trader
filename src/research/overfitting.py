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

EULER_MASCHERONI = 0.5772156649015329
DSR_THRESHOLD = 0.95  # 去膨胀夏普门槛：低于它无法与「试 N 次碰运气」区分
PBO_THRESHOLD = 0.50  # 过拟合概率门槛
DEFAULT_BLOCKS = 10  # CSCV 分块数（须为偶数）
MIN_BLOCKS = 4
MIN_BARS_PER_BLOCK = 60
MIN_OBSERVATIONS = 2
MIN_CANDIDATES = 2

_NORMAL = NormalDist()


def sharpe_ratio(returns: pd.Series | np.ndarray) -> float:
    """非年化夏普（逐根收益均值 / 标准差），DSR 与 PBO 统一用它。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_OBSERVATIONS:
        return 0.0
    sd = float(arr.std(ddof=1))
    return float(arr.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(variance_of_sharpe: float, trials: int) -> float:
    """零假设（无技能）下 N 次试验的夏普最大值期望。"""
    if variance_of_sharpe <= 0 or trials < MIN_CANDIDATES:
        return 0.0
    z_first = _NORMAL.inv_cdf(1 - 1 / trials)
    z_last = _NORMAL.inv_cdf(1 - 1 / (trials * math.e))
    weight = EULER_MASCHERONI
    return float(
        math.sqrt(variance_of_sharpe)
        * ((1 - weight) * z_first + weight * z_last)
    )


@dataclass(frozen=True)
class SharpeTest:
    """单个候选的去膨胀夏普检验结果。"""

    label: str
    sharpe: float
    expected_max: float
    dsr: float

    @property
    def passed(self) -> bool:
        return self.dsr >= DSR_THRESHOLD


@dataclass(frozen=True)
class PboResult:
    """CSCV 回测过拟合概率：样本内冠军在样本外跌到中位数以下的比例。"""

    blocks: int
    splits: int
    pbo: float
    logit_median: float
    winner_oos_sharpe_median: float

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


def deflated_sharpe(label: str, returns: pd.Series, expected_max: float) -> SharpeTest:
    """把观测夏普与零假设下的期望最大夏普比较，输出 DSR 概率。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    sr = sharpe_ratio(arr)
    if arr.size < MIN_OBSERVATIONS:
        return SharpeTest(label, sr, expected_max, 0.0)

    series = pd.Series(arr)
    skew = float(series.skew())
    kurtosis = float(series.kurt()) + 3.0  # 转成非超额峰度
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if denominator <= 0:
        return SharpeTest(label, sr, expected_max, 0.0)
    z_score = (sr - expected_max) * math.sqrt(arr.size - 1) / math.sqrt(denominator)
    return SharpeTest(label, sr, expected_max, float(_NORMAL.cdf(z_score)))


def _resolve_blocks(bars: int, requested: int) -> int | None:
    """分块数须为偶数，且每块不少于 MIN_BARS_PER_BLOCK 根，否则不做 PBO。"""
    limit = bars // MIN_BARS_PER_BLOCK
    blocks = min(requested, limit if limit % 2 == 0 else limit - 1)
    return blocks if blocks >= MIN_BLOCKS else None


def probability_of_backtest_overfitting(
    series_map: dict[str, pd.Series], blocks: int = DEFAULT_BLOCKS
) -> PboResult | None:
    """CSCV：把时间轴等分 S 块，枚举全部「一半样本内 / 一半样本外」切分。"""
    frame = pd.DataFrame(series_map).dropna(how="any")
    if frame.shape[1] < MIN_CANDIDATES or frame.empty:
        return None
    effective = _resolve_blocks(len(frame), blocks)
    if effective is None:
        return None

    values = frame.to_numpy(dtype=float)
    bounds = np.linspace(0, values.shape[0], effective + 1).astype(int)
    rows = [np.arange(bounds[i], bounds[i + 1]) for i in range(effective)]
    half = effective // 2
    logits: list[float] = []
    winner_oos: list[float] = []

    for combo in combinations(range(effective), half):
        chosen = set(combo)
        in_sample = np.concatenate([rows[i] for i in range(effective) if i in chosen])
        out_sample = np.concatenate([rows[i] for i in range(effective) if i not in chosen])
        is_sharpe = np.array([sharpe_ratio(values[in_sample, j]) for j in range(frame.shape[1])])
        oos_sharpe = np.array([sharpe_ratio(values[out_sample, j]) for j in range(frame.shape[1])])
        winner = int(np.argmax(is_sharpe))
        rank = int((oos_sharpe <= oos_sharpe[winner]).sum())  # 1..N
        percentile = rank / (frame.shape[1] + 1)
        logits.append(math.log(percentile / (1 - percentile)))
        winner_oos.append(float(oos_sharpe[winner]))

    logit_arr = np.asarray(logits)
    return PboResult(
        blocks=effective,
        splits=int(logit_arr.size),
        pbo=float((logit_arr < 0).mean()),
        logit_median=float(np.median(logit_arr)),
        winner_oos_sharpe_median=float(np.median(winner_oos)),
    )


def build_overfitting_report(
    series_map: dict[str, pd.Series],
    trials: int | None = None,
    blocks: int = DEFAULT_BLOCKS,
) -> OverfittingReport:
    """对一组候选（同一份数据上的全部试验）做 DSR 与 PBO。"""
    sharpes = [sharpe_ratio(series) for series in series_map.values()]
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) >= MIN_CANDIDATES else 0.0
    trial_count = max(trials or len(series_map), MIN_CANDIDATES)
    expected_max = expected_max_sharpe(variance, trial_count)
    tests = [
        deflated_sharpe(label, series, expected_max)
        for label, series in series_map.items()
    ]
    return OverfittingReport(
        trials=trial_count,
        observations=len(pd.DataFrame(series_map).dropna(how="any")),
        sharpe_tests=tests,
        pbo=probability_of_backtest_overfitting(series_map, blocks),
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
        print(f"试验集合 = {len(series_map)} 个候选，声明的历史试验次数 N={report.trials}。")
    print(
        f"{'策略':<26}{'夏普(非年化)':>13}{'零假设期望最大':>15}"
        f"{'DSR':>9}{'判定':>7}"
    )
    print("-" * 118)
    for test in report.sharpe_tests:
        verdict = "通过" if test.passed else "未通过"
        print(
            f"{test.label:<26}{test.sharpe:>12.4f} {test.expected_max:>14.4f} "
            f"{test.dsr:>8.3f} {verdict:>6}"
        )

    pbo = report.pbo
    if pbo is None:
        print("\nPBO：样本不足或候选过少，未计算。")
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
