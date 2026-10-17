"""
稳健性检验：全样本指标 + 一年滚动窗口分布 + 分段稳定性。

回答的问题是：这个策略的收益是全样本侥幸，还是在多个时间段上都成立。
单一全样本数字不能作为上真钱的依据，先看分布，再看分段一致性。

口径约定：
- 滚动窗口是「一年」，按评估区间的实际周期折算根数（1h 下 365 根只有 15 天）；
- 窗口之间高度重叠，分位数是同一序列的自相关函数，不是独立样本；
- 分段只报区间收益与段内回撤，不做年化（不足一天的分段年化会放大成天文数字）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from src.analytics.metrics import bars_per_year, max_drawdown, total_return
from src.utils.console import pad

LINE_WIDTH = 130
W_NAME = 34  # 策略名列宽：最长策略名「RSRS 阻力支撑相对强度 (RSRS Trend)」占 34 显示列
W_SEG = 17  # 分段明细单元格：8 + 1 + 8 个字符的「策略/基准」

# 主表列宽：表头与数据共用同一组常量，改一处即可
W_RET, W_ANN, W_DD, W_CALMAR, W_EXPO = 9, 9, 8, 8, 7
W_COUNT, W_WIN, W_WORST, W_DDM, W_DDW, W_SHARE = 7, 9, 9, 9, 9, 9
WINDOW_WIDTHS = (W_COUNT, W_WIN, W_WIN, W_WORST, W_DDM, W_DDW, W_SHARE)


@dataclass(frozen=True)
class WindowStats:
    """持有固定根数（按周期折算的一年）的滚动窗口收益与窗口内回撤分布。"""

    count: int
    positive_share: float
    p5: float
    median: float
    worst: float
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


def rolling_window_stats(equity: pd.Series, window: int) -> WindowStats:
    """滚动窗口分布：第 i 个窗口 = 从第 i 根起持有 window 根。"""
    values = equity.to_numpy(dtype=float)
    rets: list[float] = []
    dds: list[float] = []
    for start in range(max(len(values) - window, 0)):
        seg = values[start : start + window + 1]
        rets.append(float(seg[-1] / seg[0] - 1))
        dds.append(float((seg / np.maximum.accumulate(seg) - 1).min()))
    if not rets:
        return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(rets)
    dd = np.asarray(dds)
    return WindowStats(
        count=int(arr.size),
        positive_share=float((arr > 0).mean()),
        p5=float(np.percentile(arr, 5)),
        median=float(np.median(arr)),
        worst=float(arr.min()),
        dd_median=float(np.median(dd)),
        dd_worst=float(dd.min()),
    )


def split_segments(index: pd.DatetimeIndex, segments: int) -> list[pd.DatetimeIndex]:
    """把评估区间按根数等分成若干段。"""
    if segments < 2:
        return [index]
    bounds = np.linspace(0, len(index), segments + 1).astype(int)
    return [pd.DatetimeIndex(index[bounds[i] : bounds[i + 1]]) for i in range(segments)]


def segment_table(
    net_ret: pd.Series, bench_ret: pd.Series, segments: int
) -> list[SegmentStats]:
    """逐段统计收益、基准与回撤（不年化：不足一天的分段年化会放大成天文数字）。"""
    out: list[SegmentStats] = []
    for seg_index in split_segments(pd.DatetimeIndex(net_ret.index), segments):
        seg_net = net_ret.loc[seg_index]
        out.append(
            SegmentStats(
                start=cast(pd.Timestamp, seg_index[0]),
                end=cast(pd.Timestamp, seg_index[-1]),
                bars=int(seg_index.size),
                total_ret=total_return(seg_net),
                bench_ret=total_return(bench_ret.loc[seg_index]),
                max_drawdown=max_drawdown(seg_net),
            )
        )
    return out


def build_report(
    name: str, result: dict[str, Any], segments: int, window: int
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


def _window_cells(stats: WindowStats) -> str:
    """滚动窗口各列；评估区间不足一个窗口时打 n/a 而不是一排 0（避免读成收益为零）。"""
    if stats.count == 0:
        return "".join(pad("n/a", width, "right") for width in WINDOW_WIDTHS)
    values = (
        f"{stats.count:d}",
        f"{stats.median:.1%}",
        f"{stats.p5:.1%}",
        f"{stats.worst:.1%}",
        f"{stats.dd_median:.1%}",
        f"{stats.dd_worst:.1%}",
        f"{stats.positive_share:.1%}",
    )
    return "".join(
        pad(v, w, "right") for v, w in zip(values, WINDOW_WIDTHS, strict=True)
    )


def _segment_label(start: pd.Timestamp, intraday: bool) -> str:
    """分段起点标签：intraday 下带时刻，否则同一天的多段会打印成同一个日期。"""
    return start.strftime("%m-%d %H:%M") if intraday else start.strftime("%Y-%m-%d")


def print_reports(
    symbol: str, index: pd.DatetimeIndex, reports: list[StrategyReport], window: int
) -> None:
    """打印全样本、滚动窗口、分段三张表（最后一行应为买入持有基准）。"""
    span = f"{cast(pd.Timestamp, index[0]).date()} → {cast(pd.Timestamp, index[-1]).date()}"
    window_days = window / bars_per_year(index) * 365
    print(f"\n{'=' * LINE_WIDTH}")
    print(
        f"{symbol} | 评估区间 {span} | {len(index)} 根"
        f" | 滚动窗口 {window} 根 ≈ {window_days:.0f} 天（重叠窗口，非独立样本）"
    )
    print("=" * LINE_WIDTH)

    print(
        pad("策略", W_NAME)
        + pad("总收益", W_RET, "right")
        + pad("年化", W_ANN, "right")
        + pad("回撤", W_DD, "right")
        + pad("卡尔玛", W_CALMAR, "right")
        + pad("持仓", W_EXPO, "right")
        + pad("窗口数", W_COUNT, "right")
        + pad("窗口中位", W_WIN, "right")
        + pad("窗口P5", W_WIN, "right")
        + pad("窗口最差", W_WORST, "right")
        + pad("回撤中位", W_DDM, "right")
        + pad("回撤最差", W_DDW, "right")
        + pad("正收益窗", W_SHARE, "right")
    )
    print("-" * LINE_WIDTH)
    for report in reports:
        print(
            pad(report.name, W_NAME)
            + pad(f"{report.total_ret:.1%}", W_RET, "right")
            + pad(f"{report.annual_ret:.1%}", W_ANN, "right")
            + pad(f"{report.max_drawdown:.1%}", W_DD, "right")
            + pad(f"{report.calmar_ratio:.2f}", W_CALMAR, "right")
            + pad(f"{report.exposure:.1%}", W_EXPO, "right")
            + _window_cells(report.window)
        )

    if not reports or not reports[0].segments:
        return

    print(f"\n分段稳定性（{len(reports[0].segments)} 段）")
    print(
        pad("策略", W_NAME)
        + pad("正收益段", 10, "right")
        + pad("跑赢基准段", 12, "right")
        + pad("最差段", 9, "right")
        + pad("最好段", 9, "right")
        + pad("最差段回撤", 11, "right")
    )
    print("-" * LINE_WIDTH)
    for report in reports:
        segs = report.segments
        positive = sum(1 for s in segs if s.total_ret > 0)
        beaten = sum(1 for s in segs if s.total_ret > s.bench_ret)
        print(
            pad(report.name, W_NAME)
            + pad(f"{positive}/{len(segs)}", 10, "right")
            + pad(f"{beaten}/{len(segs)}", 12, "right")
            + pad(f"{min(s.total_ret for s in segs):.1%}", 9, "right")
            + pad(f"{max(s.total_ret for s in segs):.1%}", 9, "right")
            + pad(f"{min(s.max_drawdown for s in segs):.1%}", 11, "right")
        )

    first = reports[0].segments
    intraday = any((s.end - s.start).total_seconds() < 86400 for s in first)
    print("\n分段收益明细（策略 / 同期基准）")
    print(
        pad("策略", W_NAME)
        + "".join(pad(_segment_label(s.start, intraday), W_SEG, "right") for s in first)
    )
    print("-" * LINE_WIDTH)
    for report in reports:
        cells = "".join(
            pad(f"{s.total_ret:>8.1%}/{s.bench_ret:>8.1%}", W_SEG, "right")
            for s in report.segments
        )
        print(pad(report.name, W_NAME) + cells)
