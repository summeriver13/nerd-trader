"""
Ch 4.5 顺序 bootstrap 在什么条件下有效？（实测：只有低并发时）

de Prado 的动因是「标签重叠时，均匀 bootstrap 反复抽到重叠观测」。但顺序采样能修的只是
**袋内重复**，修不了**标签自身的重叠**：并发数一高，候选的抽签权重就趋同
（1/(c+1) 在 c≈30 时几乎与 c 无关），采样退化成均匀。所以本脚本量的不是「顺序比均匀好
多少」，而是**仓库的真实标签落在哪个区间**。

口径：

- 每只标的**在自己的坐标里**算（唯一性必须在标的内部算——长表行号是另一套坐标，
  混用会静默错位，见 `src/research/study/observations.py` 的踩坑记录）。
- 两套标签对照：三重障碍（跨度可变，Ch 3——唯一性只有在这里才有内容）与固定期限
  （跨度恒为 h，唯一性退化成 1/h）。
- 袋子长度 = 标签数（有放回），每套口径跑 `--seeds` 颗种子取中位。
- 「抽签权重分散度」取「均匀袋装了半数后再看候选权重」的 max/min：并发越低这个数越大，
  越接近 1 说明采样器没有可用的杠杆。它与「顺序袋赢多少」是同一件事的两种读法。
- 「非重叠抽稀」是 de Prado 拿来对照的另一条路：唯一性恒为 1，但只留下 1/跨度 的样本。

用法：uv run python scripts/bootstrap_study.py [--market swap] [--vertical 5] [--seeds 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.research.factors.panel import Panel, load_panel
from src.research.labeling import labels
from src.research.validation import bootstrap
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

OUTPUT_DIR = Path("data/results/breaks")
MIN_BARS = 400
FIXED_NAME = "固定期限"
BARRIER_NAME = "三重障碍"
FAMILIES = (BARRIER_NAME, FIXED_NAME)


def label_spans(close: pd.Series, vertical: int) -> tuple[int, np.ndarray, np.ndarray]:
    """三重障碍的标签跨度（整数根序号，落在该标的自己的坐标里）。"""
    size = int(close.size)
    frame = labels.triple_barrier(
        close, labels.TripleBarrierOptions(vertical=vertical), side=1
    )
    if frame.empty:
        return size, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.asarray(close.index.get_indexer(frame.index), dtype=np.int64)
    ends = frame["t1"].to_numpy(dtype=np.int64)
    return size, starts, ends


def fixed_spans(size: int, vertical: int) -> tuple[np.ndarray, np.ndarray]:
    """固定期限标签的跨度：起点逐根、长度恒为 vertical。"""
    starts = np.arange(size - vertical, dtype=np.int64)
    return starts, starts + vertical


def _bag_stats(
    size: int,
    starts: np.ndarray,
    ends: np.ndarray,
    forward: np.ndarray,
    seeds: int,
) -> dict[str, float]:
    """两种采样各 `seeds` 颗种子：袋唯一性、去重占比、袋估计量。"""
    length = int(starts.size)
    sequential, uniform, distinct_seq, distinct_uni, seq_est, uni_est = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed in range(seeds):
        drawn = bootstrap.sequential_bootstrap(size, starts, ends, length, seed=seed)
        sequential.append(bootstrap.bag_uniqueness(size, starts, ends, drawn))
        distinct_seq.append(float(np.unique(drawn).size) / length)
        seq_est.append(float(np.nanmean(forward[starts[drawn]])))
        other = bootstrap.uniform_bootstrap(length, length, seed=seed)
        uniform.append(bootstrap.bag_uniqueness(size, starts, ends, other))
        distinct_uni.append(float(np.unique(other).size) / length)
        uni_est.append(float(np.nanmean(forward[starts[other]])))
    target = float(np.nanmean(forward[starts]))
    return {
        "target_forward": target,
        "unique_sequential": float(np.median(sequential)),
        "unique_uniform": float(np.median(uniform)),
        "sequential_wins": float(np.mean(np.asarray(sequential) > np.asarray(uniform))),
        "distinct_sequential": float(np.median(distinct_seq)),
        "distinct_uniform": float(np.median(distinct_uni)),
        "bias_sequential": float(np.nanmean(seq_est) - target),
        "bias_uniform": float(np.nanmean(uni_est) - target),
        "std_sequential": float(np.nanstd(seq_est, ddof=1)),
        "std_uniform": float(np.nanstd(uni_est, ddof=1)),
    }


def _family_row(
    name: str,
    symbol: str,
    size: int,
    starts: np.ndarray,
    ends: np.ndarray,
    close: pd.Series,
    vertical: int,
    seeds: int,
) -> dict[str, object]:
    """一套标签的全部读数。"""
    length = int(starts.size)
    concurrency = bootstrap.bag_concurrency(size, starts, ends, np.arange(length))
    forward = (close.shift(-vertical).div(close) - 1.0).to_numpy(dtype=float)
    half = max(length // 2, 1)
    warm = bootstrap.uniform_bootstrap(length, half, seed=0)
    weights = bootstrap.candidate_uniqueness(
        size, starts, ends, bootstrap.bag_concurrency(size, starts, ends, warm)
    )
    span_median = float(np.median(ends - starts))
    strided = np.arange(0, length, max(vertical, 1))
    thinned = bootstrap.bag_uniqueness(
        size, starts[strided], ends[strided], np.arange(strided.size)
    )
    return {
        "symbol": symbol,
        "family": name,
        "labels": length,
        "span_median": span_median,
        "concurrency_median": float(np.median(concurrency)),
        "concurrency_p95": float(np.quantile(concurrency, 0.95)),
        "concurrency_max": float(concurrency.max()),
        "unique_full": bootstrap.bag_uniqueness(size, starts, ends, np.arange(length)),
        "weight_spread": float(weights.max() / max(weights.min(), 1e-12)),
        "thinned_labels": float(strided.size),
        "thinned_unique": thinned,
        **_bag_stats(size, starts, ends, forward, seeds),
    }


def symbol_rows(
    panel: Panel, symbol: str, vertical: int, seeds: int
) -> list[dict[str, object]]:
    """一只标的的两套标签。"""
    close = cast(pd.Series, panel.close[symbol]).dropna()
    if close.size < max(MIN_BARS, 4 * vertical):
        return []
    size, barrier_starts, barrier_ends = label_spans(close, vertical)
    fixed_starts, fixed_ends = fixed_spans(int(close.size), vertical)
    rows = []
    for name, space, starts, ends in (
        (BARRIER_NAME, size, barrier_starts, barrier_ends),
        (FIXED_NAME, int(close.size), fixed_starts, fixed_ends),
    ):
        if starts.size >= 2:
            rows.append(
                _family_row(name, symbol, space, starts, ends, close, vertical, seeds)
            )
    return rows


def report(frame: pd.DataFrame) -> None:
    """四张表：并发剖面、袋唯一性、抽签杠杆、袋估计量偏差。"""
    med = frame.groupby("family", observed=True).median(numeric_only=True)
    counts = {
        str(name): int(value) for name, value in frame["family"].value_counts().items()
    }
    print("\n=== 1. 标签的并发剖面（逐标的取中位）===")
    header = f"{'标签口径':<10}{'标的数':>7}{'标签数':>8}{'跨度':>7}{'并发中位':>9}{'并发p95':>9}{'并发max':>9}{'全样本唯一性':>13}"
    print(header)
    print("-" * 96)
    for name in FAMILIES:
        if name not in med.index:
            continue
        row = med.loc[name]
        print(
            f"{name:<10}{counts.get(name, 0):>7}{row['labels']:>8.0f}{row['span_median']:>7.1f}"
            f"{row['concurrency_median']:>9.2f}{row['concurrency_p95']:>9.1f}"
            f"{row['concurrency_max']:>9.1f}{row['unique_full']:>13.4f}"
        )
    print("\n=== 2. 袋的平均唯一性（袋长 = 标签数，逐标的取中位）===")
    print(
        f"{'标签口径':<10}{'均匀袋':>9}{'顺序袋':>9}{'全样本':>9}{'顺序>均匀的标的占比':>20}{'顺序袋去重占比':>15}"
    )
    print("-" * 96)
    for name in FAMILIES:
        if name not in med.index:
            continue
        row = med.loc[name]
        print(
            f"{name:<10}{row['unique_uniform']:>9.4f}{row['unique_sequential']:>9.4f}"
            f"{row['unique_full']:>9.4f}{row['sequential_wins']:>20.2f}"
            f"{row['distinct_sequential']:>15.3f}"
        )
    print("\n=== 3. 抽签杠杆与「每 vertical 根取一个」的抽稀对照 ===")
    print(
        f"{'标签口径':<10}{'权重max/min':>12}{'抽稀后标签数':>13}{'抽稀保留占比':>13}"
        f"{'抽稀后唯一性':>13}"
    )
    print("-" * 96)
    for name in FAMILIES:
        if name not in med.index:
            continue
        row = med.loc[name]
        print(
            f"{name:<10}{row['weight_spread']:>12.3f}{row['thinned_labels']:>13.0f}"
            f"{row['thinned_labels'] / max(row['labels'], 1.0):>13.3f}"
            f"{row['thinned_unique']:>13.4f}"
        )
    print("\n=== 4. 袋估计量（均值前向收益，bp）相对全样本 ===")
    print(
        f"{'标签口径':<10}{'全样本':>10}{'均匀袋偏差':>12}{'顺序袋偏差':>12}"
        f"{'均匀袋标准差':>14}{'顺序袋标准差':>14}"
    )
    print("-" * 96)
    for name in FAMILIES:
        if name not in med.index:
            continue
        row = med.loc[name]
        print(
            f"{name:<10}{1e4 * row['target_forward']:>10.2f}"
            f"{1e4 * row['bias_uniform']:>12.2f}{1e4 * row['bias_sequential']:>12.2f}"
            f"{1e4 * row['std_uniform']:>14.2f}{1e4 * row['std_sequential']:>14.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch 4.5 顺序 bootstrap 的适用条件")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser)
    parser.add_argument("--vertical", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--symbols", type=int, default=60)
    add_no_save(parser)
    args = parser.parse_args(argv)

    panel = load_panel(args.timeframe, market=args.market)
    symbols = sorted(panel.close.columns)[: args.symbols]
    print(
        f"{args.market} {args.timeframe}：{len(symbols)} 只标的、垂直障碍 {args.vertical} 根、"
        f"{args.seeds} 颗种子、袋长 = 标签数"
    )
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(symbol_rows(panel, symbol, args.vertical, args.seeds))
    if not rows:
        print("没有可用的标的", file=sys.stderr)
        return 1
    frame = pd.DataFrame(rows)
    report(frame)
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"bootstrap_study_{args.market}_{args.timeframe}.parquet"
        frame.to_parquet(path)
        print(f"\n落盘 {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
