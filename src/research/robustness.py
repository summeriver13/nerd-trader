"""
稳健性检验：全样本指标 + 一年滚动窗口分布 + 分段稳定性。

回答的问题是：这个策略的收益是全样本侥幸，还是在多个时间段上都成立。
单一全样本数字不能作为上真钱的依据，先看分布，再看分段一致性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

ANNUAL_DAYS = 365
DEFAULT_WINDOW = 365


@dataclass(frozen=True)
class WindowStats:
    """持有固定根数（默认一年）的滚动窗口收益与窗口内回撤分布。"""

    count: int
    positive_share: float
    p5: float
    p25: float
    median: float
    p75: float
    worst: float
    best: float
    dd_median: float
    dd_worst: float


@dataclass(frozen=True)
class SegmentStats:
    """单段样本的绩效，用于判断结果是否只在某一段时间成立。"""

    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    total_ret: float
    bench_ret: float
    max_drawdown: float
    calmar: float


@dataclass(frozen=True)
class StrategyReport:
    """一个策略在一个标的上：全样本指标 + 滚动窗口分布 + 分段结果。"""

    name: str
    total_ret: float
    annual_ret: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    trade_count: int
    exposure: float
    window: WindowStats
    segments: list[SegmentStats]


class BuyAndHoldStrategy:
    """基准：始终满仓持有，任何策略必须先打败它。"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 1.0
        return df


def segment_return(net_ret: pd.Series) -> float:
    """一段区间的复利收益。"""
    return float((1 + net_ret).prod() - 1)


def max_drawdown(net_ret: pd.Series) -> float:
    """一段区间内基于自身资金曲线的最大回撤。"""
    equity = (1 + net_ret).cumprod()
    return float((equity / equity.cummax() - 1).min())


def annualize(total_ret: float, days: int) -> float:
    """按 365 天复利折算年化。"""
    if total_ret <= -1:
        return -1.0
    return (1 + total_ret) ** (ANNUAL_DAYS / max(days, 1)) - 1


def rolling_window_stats(
    equity: pd.Series, window: int = DEFAULT_WINDOW
) -> WindowStats:
    """滚动窗口分布：第 i 个窗口 = 从第 i 根起持有 window 根。"""
    values = equity.to_numpy(dtype=float)
    rets: list[float] = []
    dds: list[float] = []
    for start in range(max(len(values) - window, 0)):
        seg = values[start : start + window + 1]
        rets.append(float(seg[-1] / seg[0] - 1))
        dds.append(float((seg / np.maximum.accumulate(seg) - 1).min()))
    if not rets:
        return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(rets)
    dd = np.asarray(dds)
    return WindowStats(
        count=int(arr.size),
        positive_share=float((arr > 0).mean()),
        p5=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        p75=float(np.percentile(arr, 75)),
        worst=float(arr.min()),
        best=float(arr.max()),
        dd_median=float(np.median(dd)),
        dd_worst=float(dd.min()),
    )


def split_segments(index: pd.DatetimeIndex, segments: int) -> list[pd.DatetimeIndex]:
    """把评估区间按根数等分成若干段。"""
    if segments < 2:
        return [index]
    bounds = np.linspace(0, len(index), segments + 1).astype(int)
    return [index[bounds[i] : bounds[i + 1]] for i in range(segments)]


def segment_table(
    net_ret: pd.Series, bench_ret: pd.Series, segments: int
) -> list[SegmentStats]:
    """逐段统计收益、基准与回撤。"""
    out: list[SegmentStats] = []
    for seg_index in split_segments(pd.DatetimeIndex(net_ret.index), segments):
        seg_net = net_ret.loc[seg_index]
        seg_days = max(int((seg_index[-1] - seg_index[0]).days), 1)
        drawdown = max_drawdown(seg_net)
        total = segment_return(seg_net)
        out.append(
            SegmentStats(
                start=cast(pd.Timestamp, seg_index[0]),
                end=cast(pd.Timestamp, seg_index[-1]),
                bars=int(seg_index.size),
                total_ret=total,
                bench_ret=segment_return(bench_ret.loc[seg_index]),
                max_drawdown=drawdown,
                calmar=annualize(total, seg_days) / abs(drawdown) if drawdown else 0.0,
            )
        )
    return out


def build_report(
    name: str,
    result: dict[str, Any],
    segments: int,
    window: int = DEFAULT_WINDOW,
) -> StrategyReport:
    """把引擎结果整理成全样本指标 + 滚动窗口分布 + 分段结果。"""
    df = result["df"]
    pos = df["pos"]
    return StrategyReport(
        name=name,
        total_ret=float(result["total_ret"]),
        annual_ret=float(result["annual_ret"]),
        max_drawdown=float(result["max_drawdown"]),
        calmar_ratio=float(result["calmar_ratio"]),
        win_rate=float(result["win_rate"]),
        trade_count=int(result["trade_count"]),
        exposure=float((pos != 0).mean()),
        window=rolling_window_stats(result["equity_curve"], window),
        segments=segment_table(df["net_ret"], df["pct_change"], segments),
    )


def print_reports(
    symbol: str,
    index: pd.DatetimeIndex,
    reports: list[StrategyReport],
    window: int = DEFAULT_WINDOW,
) -> None:
    """打印全样本、滚动窗口、分段三张表（最后一行应为买入持有基准）。"""
    span = f"{cast(pd.Timestamp, index[0]).date()} → {cast(pd.Timestamp, index[-1]).date()}"
    print(f"\n{'=' * 118}")
    print(f"{symbol} | 评估区间 {span} | {len(index)} 根 | 滚动窗口 {window} 根")
    print("=" * 118)

    print(
        f"{'策略':<26}{'总收益':>10}{'年化':>9}{'回撤':>9}{'卡尔玛':>8}{'持仓占比':>9}"
        f"{'窗口数':>8}{'窗口中位':>10}{'窗口P25':>9}{'窗口最差':>10}"
        f"{'正收益窗占比':>12}{'窗口回撤中位':>12}"
    )
    print("-" * 118)
    for report in reports:
        stats = report.window
        if stats.count == 0:
            # 评估区间不足一个滚动窗口时不再报一排 0，避免读成「收益为零」
            window_cells = f"{'n/a':>8}{'n/a':>10}{'n/a':>9}{'n/a':>10}{'n/a':>12}{'n/a':>12}"
        else:
            window_cells = (
                f"{stats.count:>7d} {stats.median:>9.1%} "
                f"{stats.p25:>8.1%} {stats.worst:>9.1%} {stats.positive_share:>11.1%} "
                f"{stats.dd_median:>11.1%}"
            )
        print(
            f"{report.name:<26}{report.total_ret:>9.1%} {report.annual_ret:>8.1%} "
            f"{report.max_drawdown:>8.1%} {report.calmar_ratio:>7.2f} "
            f"{report.exposure:>8.1%} {window_cells}"
        )

    if not reports or not reports[0].segments:
        return

    print(f"\n分段稳定性（{len(reports[0].segments)} 段）")
    print(f"{'策略':<26}{'正收益段':>10}{'跑赢基准段':>12}{'最差段':>10}{'最好段':>10}")
    print("-" * 118)
    for report in reports:
        segs = report.segments
        positive = sum(1 for s in segs if s.total_ret > 0)
        beaten = sum(1 for s in segs if s.total_ret > s.bench_ret)
        worst = min(s.total_ret for s in segs)
        best = max(s.total_ret for s in segs)
        print(
            f"{report.name:<26}{f'{positive}/{len(segs)}':>10}"
            f"{f'{beaten}/{len(segs)}':>12}{worst:>9.1%} {best:>9.1%}"
        )

    print("\n分段收益明细（策略 / 同期基准）")
    print(f"{'策略':<26}" + "".join(f"{s.start.date()!s:>19}" for s in reports[0].segments))
    print("-" * 118)
    for report in reports:
        cells = "".join(
            f"{s.total_ret:>8.1%}/{s.bench_ret:>8.1%}" for s in report.segments
        )
        print(f"{report.name:<26}{cells}")
