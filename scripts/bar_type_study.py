"""
bar 类型对照：时间棒 / 美元棒 / 量棒 / tick 棒 / 失衡棒 / run 棒（de Prado Ch 2）。

de Prado 的原命题是「等量棒比时间棒更接近正态、自相关更低」，理由是不按时间采样会让
每根棒的信息含量大致相等。本脚本把它拆成两组可测的对照：

- **A 组（OKX 永续，1m 源，30 只）**：日历日线 vs 美元棒 vs 量棒。三种都按「每天一根」
  折算，平均跨度天然可比，差异只来自「按什么累计」。
- **B 组（Binance 现货，1h 源）**：时间棒 vs tick 棒 vs 美元失衡棒 vs run 棒。只有这个源
  带 `trades` 与 taker 分列（见 `scripts/fetch_binance_klines.py`），但粒度是 1 小时，
  所以「一笔」= 1 小时棒、run 长度以小时计，这是降级版（de Prado 跑在逐笔上）。
  信息棒的平均跨度事先不可控，故另配一条**同平均跨度的等量时间棒**做对照——否则
  「失衡棒自相关更低」很可能只是「棒更长」。

读数两类：诊断（棒数、跨度、对数收益的偏度/超额峰度/自相关）与反转 IC（同前向收益、
同掩码、同窗口号，只换尺度，复用 `src/research/sampling/bar_factors.py`）。

用法：
    uv run python scripts/bar_type_study.py [--market swap] [--horizon 1] [--no-save]

产物落 data/results/bars/bar_types.parquet。只读本地缓存，不联网。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import CRYPTO_ROOT, RESULTS_ROOT, crypto_dataset_dir
from src.research.factors.evaluation import forward_returns
from src.research.factors.panel import UniverseRule, load_panel, tradable_mask
from src.research.sampling.bar_factors import (
    calendar_reversal,
    ic_row,
    keep_symbols,
    load_bar_closes,
    rescaled_reversal,
)
from src.research.sampling.bars import (
    BarOptions,
    aggregate_groups,
    group_bounds,
    threshold_for_bars,
    tick_bars,
)
from src.research.sampling.imbalance_bars import (
    SIGN_NOTIONAL,
    FlowBarOptions,
    imbalance_bars,
    run_bars,
)
from src.utils.study_cli import add_market_argument, add_no_save

DEFAULT_WINDOWS = (1, 3, 5, 10, 20)
DOLLAR_DIR = "dollar"
VOLUME_DIR = "volume"
BINANCE_1H = crypto_dataset_dir("binance", "spot", "1h")
# B 组的门槛：至少两年 1 小时历史，否则偏度/自相关这些统计量没有意义
MIN_HOURS = 2 * 365 * 24
# 失衡/run 棒的目标跨度（小时）：E[T] 初值取它，实测跨度再用于配对时间棒
TARGET_HOURS = 6.0
MIN_RETURNS = 50
# 取中位的列：逐个 Series 取均值，避免 DataFrame.median() 的类型桩摩擦
MEDIAN_COLUMNS = (
    "bars",
    "mean_hours",
    "span_ratio",
    "skew",
    "ex_kurtosis",
    "ar1",
    "abs_ar1",
)


@dataclass(frozen=True)
class BarStats:
    """一种棒在整池上的诊断（逐标的算完取中位）。"""

    label: str
    symbols: int
    bars: float
    mean_hours: float
    span_ratio: float
    skew: float
    ex_kurtosis: float
    ar1: float
    abs_ar1: float


def symbol_stats(close: pd.Series) -> dict[str, float]:
    """单标的诊断：跨度统计与对数收益的分布/自相关。

    偏度与超额峰度用有偏（总体）口径，与 scipy 的默认略有差别——同一批读数里所有棒种
    用同一口径，比较才是有效的。跨度 p95/p5 度量「信息对齐」：时间棒恒为 1，等量棒越大
    说明冷热时段的棒长差异越大。
    """
    stamps = np.asarray(close.index, dtype="datetime64[s]").astype("int64")
    gaps = np.diff(stamps) / 3600.0
    log_ret = np.diff(np.log(close.to_numpy(dtype=float)))
    empty = {
        "bars": float(len(close)),
        "mean_hours": float(np.mean(gaps)) if gaps.size else float("nan"),
        "span_ratio": float("nan"),
        "skew": float("nan"),
        "ex_kurtosis": float("nan"),
        "ar1": float("nan"),
        "abs_ar1": float("nan"),
    }
    if log_ret.size < MIN_RETURNS or not gaps.size:
        return empty
    centered = log_ret - log_ret.mean()
    std = float(centered.std())
    if std <= 0.0:
        return empty
    low, high = np.percentile(gaps, [5, 95])
    return {
        "bars": float(len(close)),
        "mean_hours": float(gaps.mean()),
        "span_ratio": float(high / max(low, 1e-9)),
        "skew": float((centered**3).mean() / std**3),
        "ex_kurtosis": float((centered**4).mean() / std**4 - 3.0),
        "ar1": float(np.corrcoef(log_ret[1:], log_ret[:-1])[0, 1]),
        "abs_ar1": float(np.corrcoef(np.abs(log_ret[1:]), np.abs(log_ret[:-1]))[0, 1]),
    }


def bar_stats(label: str, closes: dict[str, pd.Series]) -> BarStats:
    """整池诊断：逐标的算完取中位——各标的棒数差几个数量级，直接池化会被大币主导。"""
    rows = [symbol_stats(close) for close in closes.values()]
    frame = pd.DataFrame(rows)
    usable = frame[frame["mean_hours"].notna()]
    source = usable if not usable.empty else frame
    median = {
        name: float(cast(pd.Series, source[name]).median()) for name in MEDIAN_COLUMNS
    }
    return BarStats(
        label=label,
        symbols=len(closes),
        bars=median["bars"],
        mean_hours=median["mean_hours"],
        span_ratio=median["span_ratio"],
        skew=median["skew"],
        ex_kurtosis=median["ex_kurtosis"],
        ar1=median["ar1"],
        abs_ar1=median["abs_ar1"],
    )


def spaced_bars(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    """每 hours 根源棒归一根的等量时间棒——与信息棒做跨度配对用。"""
    group = np.arange(len(frame), dtype=np.int64) // max(hours, 1)
    starts, ends = group_bounds(group)
    return aggregate_groups(frame, starts, ends)


def mean_hours(close: pd.Series) -> float:
    """棒的平均跨度（小时）。"""
    stamps = np.asarray(close.index, dtype="datetime64[s]").astype("int64")
    gaps = np.diff(stamps) / 3600.0
    return float(gaps.mean()) if gaps.size else float("nan")


def load_binance_1h(min_hours: int) -> dict[str, pd.DataFrame]:
    """读 Binance 1 小时源棒（带 trades 与 taker 分列），按历史长度过滤。"""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(BINANCE_1H.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame) >= min_hours:
            frames[path.stem] = frame
    return frames


def print_stats(rows: list[BarStats]) -> None:
    """诊断表：跨度与收益分布。"""
    print("\n=== 棒类型诊断（逐标的取中位）===")
    print(
        f"{'棒种':<22}{'只数':>5}{'棒数':>9}{'平均跨度(h)':>12}"
        f"{'跨度p95/p5':>11}{'偏度':>8}{'超额峰度':>10}{'AR(1)':>8}{'|r|AR(1)':>10}"
    )
    print("-" * 103)
    for row in rows:
        print(
            f"{row.label:<22}{row.symbols:>5}{row.bars:>9,.0f}{row.mean_hours:>12.2f}"
            f"{row.span_ratio:>11.2f}{row.skew:>8.2f}{row.ex_kurtosis:>10.2f}"
            f"{row.ar1:>8.3f}{row.abs_ar1:>10.3f}"
        )


def print_ic(rows: list[dict[str, object]]) -> None:
    """IC 表：每个窗口一行、每个尺度一列。"""
    table = pd.DataFrame(rows)
    if table.empty:
        return
    horizon = int(cast(int, table["horizon"].iloc[0]))
    pivot = table.pivot_table(index="window", columns="scale", values="mean_ic")
    print(f"\n=== 反转因子 IC（前向 {horizon} 日，同前向收益/同掩码，只换尺度）===")
    header = f"{'窗口':>6}" + "".join(f"{str(col):>14}" for col in pivot.columns)
    print(header)
    print("-" * len(header))
    for window in pivot.index:
        line = f"{int(window):>6}"
        for col in pivot.columns:
            line += f"{float(pivot.loc[window, col]):>14.4f}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="bar 类型对照（诊断 + 反转 IC）")
    add_market_argument(parser, default="swap")
    parser.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--horizon", type=int, default=1, help="IC 的前向期限（日）")
    add_no_save(parser)
    args = parser.parse_args()
    windows = [int(part) for part in args.windows.split(",") if part]

    started = time.time()
    panel = load_panel("1d", market=args.market)
    mask = tradable_mask(panel, UniverseRule())
    dollar = load_bar_closes(CRYPTO_ROOT / "okx" / args.market / DOLLAR_DIR)
    volume = load_bar_closes(CRYPTO_ROOT / "okx" / args.market / VOLUME_DIR)
    shared = [sym for sym in panel.symbols if sym in dollar and sym in volume]
    print(
        f"A 组 | OKX {args.market} | 日线 {len(panel.symbols)} 只 · 交集 {len(shared)} 只 | "
        f"美元棒 {len(dollar)} 只 · 量棒 {len(volume)} 只"
    )

    stats: list[BarStats] = [
        bar_stats(
            "A 日历日线", {sym: cast(pd.Series, panel.close[sym]) for sym in shared}
        ),
        bar_stats("A 美元棒", {sym: dollar[sym] for sym in shared}),
        bar_stats("A 量棒", {sym: volume[sym] for sym in shared}),
    ]

    ic_rows: list[dict[str, object]] = []
    forward = forward_returns(panel, args.horizon)
    subset_mask = keep_symbols(mask, shared)
    for window in windows:
        for label, values, scale_mask in (
            ("全池日历", calendar_reversal(panel.close, window), mask),
            (
                "同池日历",
                keep_symbols(calendar_reversal(panel.close, window), shared),
                subset_mask,
            ),
            ("美元棒", rescaled_reversal(dollar, panel.index, window), subset_mask),
            ("量棒", rescaled_reversal(volume, panel.index, window), subset_mask),
        ):
            ic_rows.append(
                ic_row(label, window, args.horizon, values, forward, scale_mask)
            )

    stats.extend(binance_stats())
    print_stats(stats)
    print_ic(ic_rows)
    if not args.no_save:
        target = RESULTS_ROOT / "bars"
        target.mkdir(parents=True, exist_ok=True)
        stats_frame = pd.DataFrame([vars(row) for row in stats])
        stats_frame.to_parquet(target / "bar_types.parquet")
        print(f"\n落盘 {target / 'bar_types.parquet'}")
    print(f"耗时 {time.time() - started:.0f} 秒")
    return 0


def binance_stats() -> list[BarStats]:
    """B 组：Binance 1 小时源上的时间棒 vs tick/失衡/run 棒（含跨度配对时间棒）。"""
    frames = load_binance_1h(MIN_HOURS)
    if not frames:
        print(f"B 组跳过：{BINANCE_1H} 里没有 ≥{MIN_HOURS:,} 小时历史的数据")
        return []
    print(f"B 组 | Binance 现货 1h | {len(frames)} 只 ≥{MIN_HOURS // 24} 天")
    time_bars_map: dict[str, pd.Series] = {}
    info_maps: dict[str, dict[str, pd.Series]] = {"tick": {}, "失衡": {}, "run": {}}
    matched_maps: dict[str, dict[str, pd.Series]] = {"tick": {}, "失衡": {}, "run": {}}
    for symbol, frame in frames.items():
        target_bars = max(int(round(len(frame) / TARGET_HOURS)), 1)
        threshold = threshold_for_bars(frame, target_bars, "trades")
        options = FlowBarOptions(expected_bars=TARGET_HOURS)
        series = {
            "tick": tick_bars(frame, BarOptions(threshold=threshold))["close"],
            "失衡": imbalance_bars(frame, options, SIGN_NOTIONAL)["close"],
            "run": run_bars(frame, options, SIGN_NOTIONAL)["close"],
        }
        time_bars_map[symbol] = cast(pd.Series, frame["close"])
        for name, values in series.items():
            close = cast(pd.Series, values)
            info_maps[name][symbol] = close
            # 跨度配对：每种信息棒配一条同平均跨度的等量时间棒（至少 1 小时）
            hours = max(int(round(mean_hours(close))), 1)
            matched_maps[name][symbol] = cast(
                pd.Series, spaced_bars(frame, hours)["close"]
            )
    rows = [bar_stats("B 时间棒 1h（参考）", time_bars_map)]
    for name in ("tick", "失衡", "run"):
        rows.append(bar_stats(f"B {name}棒", info_maps[name]))
        rows.append(bar_stats(f"B 时间棒（配 {name}）", matched_maps[name]))
    return rows


if __name__ == "__main__":
    sys.exit(main())
