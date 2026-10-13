"""
src/research/tsmom_report.py
多标的时序动量研究的报告层：组合读数、分段、逐标的命中率、多重检验校正。

与 `vol_target_report.py` 同一分工：研究逻辑在 `tsmom.py`、编排在 `scripts/`、
读数整形与打印放这里——`scripts/tsmom_study.py` 加了正交性与账本级缩放两段后
会超过 300 行上限。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    annualize,
    annualized_sharpe,
    bars_per_year,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from src.research.overfitting import (
    DSR_THRESHOLD,
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_estimator_variance,
)
from src.research.robustness import segment_table
from src.research.track_record import track_record
from src.research.tsmom import cross_section_mean, portfolio_returns

SEGMENTS = 4
LABEL_WIDTH = 22
# 声明的试验数三档：臂数（6）、臂 × 两个池子（12）、再加单速度与成本的组合（30）
TRIAL_BANDS: tuple[int, ...] = (6, 12, 30)
# 逐标的对比的样本门槛：太短的标的夏普不可比
MIN_HIT_BARS = 365


@dataclass(frozen=True)
class ArmRun:
    """一个臂的逐标的引擎输出。"""

    net: dict[str, pd.Series]
    pos: dict[str, pd.Series]
    cost: dict[str, pd.Series]


@dataclass(frozen=True)
class ArmRow:
    """一个组合级绩效读数。"""

    universe: str
    label: str
    total_ret: float
    annual_ret: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    psr: float
    min_trl: float
    exposure: float
    turnover: float
    cost: float


def summarize_stream(
    universe: str,
    label: str,
    net: pd.Series,
    *,
    exposure: float = float("nan"),
    turnover: float = float("nan"),
    cost: float = 0.0,
) -> ArmRow:
    """从逐根净收益算读数；暴露与换手由调用方给（纯组合收益流没有仓位）。

    先丢掉 NaN：带宽度门槛的组合收益会有空日，cumprod 遇到 NaN 会把整条曲线变成 NaN。
    """
    net = net.dropna()
    periods = bars_per_year(net.index)
    total = total_return(net)
    drawdown = max_drawdown(net)
    annual = annualize(total, len(net), int(round(periods)))
    record = track_record(label, net)
    return ArmRow(
        universe=universe,
        label=label,
        total_ret=total,
        annual_ret=annual,
        sharpe=annualized_sharpe(net, periods),
        sortino=sortino_ratio(net, periods),
        calmar=calmar_ratio(annual, drawdown),
        max_dd=drawdown,
        psr=record.psr,
        min_trl=record.min_track_record,
        exposure=exposure,
        turnover=turnover,
        cost=cost,
    )


def summarize(universe: str, label: str, run: ArmRun) -> ArmRow:
    """从逐标的引擎输出算组合读数：等权 1/N_t 拼成组合，再算绩效与记录体检。"""
    net = portfolio_returns(run.net)
    pos = portfolio_returns(run.pos)
    return summarize_stream(
        universe,
        label,
        net,
        exposure=float(pos.mean()),
        turnover=float(cross_section_mean(pd.DataFrame(run.pos).diff().abs()).mean()),
        cost=float(portfolio_returns(run.cost).sum()),
    )


def print_rows(title: str, rows: list[ArmRow], note: str = "") -> None:
    """打印组合表。"""
    print(f"\n{title}")
    if note:
        print(note)
    header = (
        f"{'arm':<{LABEL_WIDTH}}{'total':>10}{'annual':>9}{'sharpe':>8}{'sortino':>9}"
        f"{'calmar':>8}{'maxDD':>9}{'PSR':>7}{'MinTRL':>9}{'expo':>7}{'turn':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.label:<{LABEL_WIDTH}}{row.total_ret:>9.1%}{row.annual_ret:>9.1%}"
            f"{row.sharpe:>8.2f}{row.sortino:>9.2f}{row.calmar:>8.2f}"
            f"{row.max_dd:>9.1%}{row.psr:>7.3f}{row.min_trl:>9.0f}"
            f"{row.exposure:>7.1%}{row.turnover:>7.3f}"
        )


def print_segments(title: str, net: pd.Series, bench: pd.Series) -> None:
    """四等分全历史：策略与基准的逐段收益（不年化）。"""
    print(f"\n{title}")
    header = (
        f"{'segment':<26}{'bars':>6}{'strategy':>11}{'benchmark':>11}"
        f"{'excess':>10}{'maxDD':>9}"
    )
    print(header)
    print("-" * len(header))
    for seg in segment_table(net, bench, SEGMENTS):
        label = f"{seg.start:%Y-%m} -> {seg.end:%Y-%m}"
        print(
            f"{label:<26}{seg.bars:>6}{seg.total_ret:>10.1%}{seg.bench_ret:>11.1%}"
            f"{seg.total_ret - seg.bench_ret:>10.1%}{seg.max_drawdown:>9.1%}"
        )


def hit_rate(run: ArmRun, baseline: ArmRun) -> tuple[float, float, int]:
    """逐标的对比：策略夏普高于该标的买入持有的比例、超额中位、样本数。"""
    diffs = [
        sharpe_ratio(net) - sharpe_ratio(baseline.net[stem])
        for stem, net in run.net.items()
        if stem in baseline.net and len(net) >= MIN_HIT_BARS
    ]
    array = np.asarray(diffs, dtype=float)
    return float((array > 0).mean()), float(np.median(array)), int(array.size)


def print_hits(rows: list[tuple[str, float, float, int]]) -> None:
    """打印逐标的命中率。"""
    print("\n=== 逐标的对比（策略夏普 vs 同标的买入持有）===")
    header = f"{'arm':<{LABEL_WIDTH}}{'n':>6}{'win_rate':>10}{'median_diff':>13}"
    print(header)
    print("-" * len(header))
    for label, rate, median, count in rows:
        print(f"{label:<{LABEL_WIDTH}}{count:>6}{rate:>9.1%}{median:>13.4f}")


def print_corrections(
    streams: dict[str, pd.Series], bands: tuple[int, ...] = TRIAL_BANDS
) -> None:
    """多重检验校正：这些臂里挑最好的算不算数。"""
    labels = list(streams)
    sharpes = [sharpe_ratio(streams[label]) for label in labels]
    observations = len(streams[labels[0]])
    variance = sharpe_estimator_variance(sharpes, observations)
    print("\n=== 多重检验校正（组合级，逐根夏普口径）===")
    header = (
        f"{'arm':<{LABEL_WIDTH}}{'sharpe/bar':>11}{'obs':>7}"
        + "".join(f"{'DSR@' + str(t):>10}" for t in bands)
        + "  verdict"
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        tests = [
            deflated_sharpe(
                label, streams[label], expected_max_sharpe(variance, trials=t)
            )
            for t in bands
        ]
        cells = "".join(f"{test.dsr:>10.3f}" for test in tests)
        passed = tests[-1].dsr >= DSR_THRESHOLD
        print(
            f"{label:<{LABEL_WIDTH}}{tests[0].sharpe:>11.4f}{observations:>7}{cells}"
            f"  {'PASS' if passed else 'FAIL'}"
        )
    pbo = probability_of_backtest_overfitting(streams)
    if pbo is None:
        print("PBO: 样本不足以分块，未定义")
        return
    verdict = "PASS" if pbo.passed else "FAIL"
    print(
        f"PBO CSCV: blocks={pbo.blocks} splits={pbo.splits} pbo={pbo.pbo:.1%} "
        f"winner_oos_sharpe_median={pbo.winner_oos_sharpe_median:.2f} -> {verdict}"
    )
