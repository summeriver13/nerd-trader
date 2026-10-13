"""
scripts/cpcv_study.py
第 12 章 CPCV 实证：把「三种配权方法有没有差别」从一个数换成一个分布。

第 16 章用一次 50/50 分割读到等权 / 逆方差 / HRP 的夏普差只有 0.002~0.019，结论只能是
「差异在噪声内」——但那是**单次切分**给出的一个数。CPCV 把时间轴切成 N 组、每次拿 k 组做
测试（共 C(N,k) 次切分），再按原书规则拼成 φ = C(N,k)·k/N 条覆盖全时段的**回测路径**：
于是每个方案有 φ 个夏普，两条方案之差也有 φ 个，可以报中位数、极差与符号一致率。

口径：
- 协方差只从每次切分的**训练组**估，测试组只吃收益。CPCV 的训练组含测试组之后的日期，
  这是原书 K 折口径（不是滚动窗），也是「路径能覆盖全时段」的前提。
- 同一路径上各方案走同一批日期，所以差值是**配对读数**——这是它强于「换个分割重跑一遍」
  的地方：日历相位噪声被配对消掉。
- 第 16 章的 NCO 也在这里比：五臂 = 等权 / 逆方差 / HRP / NCO（簇数 3）/ 全局最小方差，
  另扫 NCO 的簇数 2~6 报**整个平台**（挑最好的那个就是又一次搜索）。NCO 与全局最小方差
  两层次都解无约束最小方差（原文 optPort），允许做空策略。
- purge/embargo 以天计，默认各 1 天（日频 h=1 的价差，只防序列相关）。
- 报的是**年化**夏普（逐根 × √365）；第 16 章的表是逐根口径（0.045 一类），别直接比数值。
- 对照组（单次 50% 切分）必须复现第 16 章：swap 逐根 0.0451/0.0475/0.0444、spot
  0.0493/0.0409/0.0630——这条是接线自检，跑一次就能看出接错没有。
- 不记新试验：策略与第 16 章同一批，只换了评估口径与配权方案；NCO 不生成收益信号，
  簇数扫描又是「报平台、不挑选」，故都不构成新试验（同「预算扫描不计新试验」例）。

用法：uv run python scripts/cpcv_study.py [--market swap] [--no-save]
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.analytics.metrics import annualized_sharpe
from src.config.paths import RESULTS_ROOT
from src.research.allocation import (
    DEFAULT_CLUSTERS,
    hrp_weights,
    inverse_variance_weights,
    minimum_variance_weights,
    nco_weights,
)
from src.research.cpcv import CombinatorialPurgedCv, CpcvSplit, path_series
from src.research.factors.panel import UniverseRule, load_panel, tradable_mask
from src.research.observations import strategy_spreads

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "breaks"
# 与第 16 章同一批策略，保证两节的读数可比
DEFAULT_STRATEGIES: Final[tuple[str, ...]] = (
    "vol_30",
    "mom_7",
    "mom_30",
    "rev_3",
    "dth_60",
    "ffd_d0p4",
    "liq_30",
)
SCHEMES: Final[tuple[str, ...]] = ("等权", "逆方差", "HRP", "NCO", "最小方差")
# NCO 簇数扫描：报整个平台而不是挑最好的那个（挑就是又一次搜索）
NCO_CLUSTER_SCAN: Final[tuple[int, ...]] = (2, 3, 4, 5, 6)
MIN_SERIES: Final[int] = 400
MIN_DATES_PER_GROUP: Final[int] = 30
PERIODS_PER_YEAR: Final[int] = 365
SINGLE_SPLIT: Final[float] = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPCV：配权方案的差值分布")
    parser.add_argument("--market", default="swap", help="市场层，默认 swap")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument("--horizon", type=int, default=1, help="统一期限，默认 1")
    parser.add_argument("--groups", type=int, default=10, help="时间轴切几组，默认 10")
    parser.add_argument(
        "--test-groups", type=int, default=2, help="每次拿几组做测试，默认 2"
    )
    parser.add_argument(
        "--purge", type=int, default=1, help="测试块之前挖掉几天，默认 1"
    )
    parser.add_argument(
        "--embargo", type=int, default=1, help="测试块之后挖掉几天，默认 1"
    )
    parser.add_argument("--strategies", default=None, help="逗号分隔的因子键")
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def scheme_weights(
    scheme: str, train: pd.DataFrame, clusters: int = DEFAULT_CLUSTERS
) -> np.ndarray:
    """一个配权方案在训练段上的权重（NCO 的簇数由 `clusters` 给）。"""
    if scheme == "等权":
        return np.full(train.shape[1], 1.0 / train.shape[1])
    covariance = train.cov().to_numpy(dtype=float)
    if scheme == "逆方差":
        return inverse_variance_weights(covariance)
    if scheme == "最小方差":
        return minimum_variance_weights(covariance)
    if scheme == "NCO":
        return nco_weights(train, clusters=clusters).to_numpy(dtype=float)
    return hrp_weights(train).to_numpy(dtype=float)


def path_table(
    frame: pd.DataFrame,
    splits: list[CpcvSplit],
    index: pd.Index,
    *,
    paths: np.ndarray,
    groups: tuple[np.ndarray, ...],
    schemes: Sequence[str] = SCHEMES,
    clusters: int = DEFAULT_CLUSTERS,
) -> pd.DataFrame:
    """逐切分算各方案在**测试段**上的加权收益（对齐到全时段，测试段之外为 NaN），
    拼成路径后逐路径算年化夏普。
    """
    values = frame.to_numpy(dtype=float)
    readings: dict[str, list[pd.Series]] = {scheme: [] for scheme in schemes}
    for split in splits:
        train = frame.iloc[split.train_dates]
        for scheme in schemes:
            weights = scheme_weights(scheme, train, clusters=clusters)
            series = pd.Series(np.nan, index=index, dtype=float)
            series.iloc[split.test_dates] = values[split.test_dates] @ weights
            readings[scheme].append(series)
    return pd.DataFrame(
        {scheme: per_path_sharpes(readings[scheme], paths, groups) for scheme in schemes}
    )


def cluster_scan(
    frame: pd.DataFrame,
    splits: list[CpcvSplit],
    index: pd.Index,
    paths: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> pd.DataFrame:
    """NCO 的簇数扫描：每个 K 一列逐路径夏普。"""
    columns: dict[str, pd.Series] = {}
    for clusters in NCO_CLUSTER_SCAN:
        table = path_table(
            frame,
            splits,
            index,
            paths=paths,
            groups=groups,
            schemes=("NCO",),
            clusters=clusters,
        )
        columns[f"NCO@{clusters}"] = pd.Series(
            table["NCO"].to_numpy(dtype=float), index=table.index, dtype=float
        )
    return pd.DataFrame(columns)


def weight_diagnostics(
    frame: pd.DataFrame, splits: list[CpcvSplit], schemes: Sequence[str] = SCHEMES
) -> pd.DataFrame:
    """各方案在训练段上的权重形态：最大绝对权重、集中度（平方和）、负权重个数。"""
    rows: list[dict[str, object]] = []
    for scheme in schemes:
        maxima: list[float] = []
        concentrations: list[float] = []
        negatives: list[float] = []
        for split in splits:
            weights = scheme_weights(scheme, frame.iloc[split.train_dates])
            maxima.append(float(np.abs(weights).max()))
            concentrations.append(float((weights**2).sum()))
            negatives.append(float((weights < 0).sum()))
        rows.append(
            {
                "方案": scheme,
                "最大|权重|中位": float(np.median(maxima)),
                "集中度中位": float(np.median(concentrations)),
                "负权重数中位": float(np.median(negatives)),
            }
        )
    return pd.DataFrame(rows)


def per_path_sharpes(
    readings: list[pd.Series], paths: np.ndarray, groups: tuple[np.ndarray, ...]
) -> pd.Series:
    """把一次方案的各切分读数拼成路径，逐路径算年化夏普。"""
    frame = path_series(readings, paths=paths, group_dates=groups)
    values = [
        annualized_sharpe(frame.iloc[:, column].to_numpy(), PERIODS_PER_YEAR)
        for column in range(frame.shape[1])
    ]
    return pd.Series(values, index=frame.columns)


def single_split_sharpes(frame: pd.DataFrame, cutoff: int) -> dict[str, float]:
    """对照组：第 16 章那套「一次切分」的读法（训练前半段、吃后半段收益）。"""
    train, test = frame.iloc[:cutoff], frame.iloc[cutoff:]
    values = test.to_numpy(dtype=float)
    return {
        scheme: annualized_sharpe(
            values @ scheme_weights(scheme, train), PERIODS_PER_YEAR
        )
        for scheme in SCHEMES
    }


def describe(differences: pd.Series) -> str:
    positive = int((differences > 0).sum())
    return (
        f"中位 {differences.median():+.4f}｜"
        f"区间 [{differences.min():+.4f}, {differences.max():+.4f}]｜"
        f"正号 {positive}/{differences.size}"
    )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    keys = (
        tuple(item.strip() for item in args.strategies.split(","))
        if args.strategies
        else DEFAULT_STRATEGIES
    )
    rule = UniverseRule()
    panel = load_panel(args.timeframe, rule, market=args.market)
    mask = tradable_mask(panel, rule)
    frame = strategy_spreads(
        keys, panel, mask, horizon=args.horizon, min_series=MIN_SERIES
    ).dropna()
    dates = frame.shape[0]
    cv = CombinatorialPurgedCv(
        n_groups=args.groups,
        k_test=args.test_groups,
        purge=args.purge,
        embargo=args.embargo,
    )
    print(
        f"市场 {args.market} | 周期 {args.timeframe} | 期限 h={args.horizon} | "
        f"策略 {list(frame.columns)}"
    )
    print(
        f"期数 {dates} | CPCV：{cv.n_groups} 组 × 每次 {cv.k_test} 组做测试 = "
        f"{cv.n_splits} 次切分（purge={cv.purge}、embargo={cv.embargo}）→ "
        f"{cv.n_paths} 条回测路径"
    )
    if frame.shape[1] < 2 or dates < cv.n_groups * MIN_DATES_PER_GROUP:
        print("样本不足：要么策略太少，要么每组期数不够。")
        return 1

    splits = cv.split(dates)
    paths = cv.paths()
    groups = cv.group_dates(dates)
    table = path_table(frame, splits, frame.index, paths=paths, groups=groups)

    print("\n=== 逐路径年化夏普（同一行是同一批日期，可比）===")
    print(table.round(4).to_string())
    print("\n各方案：")
    for scheme in SCHEMES:
        column = table[scheme]
        print(
            f"  {scheme:<4}中位 {column.median():+.4f}｜"
            f"区间 [{column.min():+.4f}, {column.max():+.4f}]"
        )
    print("\n配对差值（同一路径上两方案相减）：")
    for left, right in combinations(SCHEMES, 2):
        print(f"  {left} − {right}：{describe(table[left] - table[right])}")

    scan = cluster_scan(frame, splits, frame.index, paths, groups)
    print("\nNCO 簇数扫描（逐路径年化夏普；报平台，不挑最好的那个）：")
    for name in scan.columns:
        column = scan[name]
        print(
            f"  {name:<7}中位 {column.median():+.4f}｜"
            f"区间 [{column.min():+.4f}, {column.max():+.4f}]"
        )
    print("\n权重形态（训练段上的中位数）：")
    print(weight_diagnostics(frame, splits).round(4).to_string(index=False))

    cutoff = int(dates * SINGLE_SPLIT)
    single = single_split_sharpes(frame, cutoff)
    print(f"\n对照组（第 16 章口径，单次 {SINGLE_SPLIT:.0%} 切分）：")
    for scheme in SCHEMES:
        print(f"  {scheme:<4}夏普 {single[scheme]:+.4f}")
    for left, right in combinations(SCHEMES, 2):
        print(f"  {left} − {right}：{single[left] - single[right]:+.4f}")

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"cpcv_study_{args.market}_{args.timeframe}.parquet"
        table.to_parquet(target)
        print(f"\n落盘：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
