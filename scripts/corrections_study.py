"""
scripts/corrections_study.py
多重检验校正第三族的实证：先在合成零假设床上量尺寸与功效，再对真实候选家族实测
「从一堆候选里挑最好的那个」校正后是否还显著。

背景：本仓文档已判定 FDR 那条腿在搜索-选择过程下不可识别（de Prado & Fabozzi 2026），
故 BH 只作对照臂；主检验是 bootstrap 型的 White RC 与 Hansen SPA，它们不依赖样本内
统计量识别搜索过程。

三个阶段：

- `bed`：合成数据（无信号 / 种信号），量四种做法的拒绝率。这是 Ch 13 里对本仓唯一有用
  的部分——真值已知的零假设床：不校正（取最小 p）、BH、White RC、Hansen SPA。
- `factors`：7 个主池因子 × 周期 × 市场，用**逐期多空价差**（钱的口径，不是 IC）。
- `sweep`：扫参 IC 矩阵（5 族 × 窗口 × 期限 ≈ 3276 个候选，与试验账本同规模），问
  「挑出最好的那个窗口」校正后是否还显著。IC 矩阵落缓存，二次运行直接读。

用法：uv run python scripts/corrections_study.py [--stage all] [--market swap,spot]
     [--timeframe 1d] [--replications 499]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.corrections import (
    DEFAULT_ALPHA,
    benjamini_hochberg,
    hansen_spa,
    naive_p_values,
    reality_check,
)
from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.panel import Panel, load_panel, tradable_mask
from src.research.factors.portfolio import PortfolioOptions, portfolio_stats
from src.research.factors.sweep import (
    DEFAULT_HORIZONS,
    resolve_families,
    sweep_ic_table,
)

RESULTS_DIR: Path = Path("data/results/breaks")
CACHE_DIR: Path = RESULTS_DIR / "corrections_cache"
MAIN_POOL_KEYS: tuple[str, ...] = (
    "rev_3",
    "mom_7",
    "mom_30",
    "vol_30",
    "liq_30",
    "dth_60",
    "ffd_d0p4",
)
FACTOR_HORIZONS: tuple[int, ...] = (1, 7)
# 家族级检验必须钉同一相位：非重叠网格的起点默认取各自 warmup，相位会错开，拼成的
# 矩阵每行只剩一个非 NaN、被对齐逻辑整行丢掉（README 第十六节早有这个坑）
FAMILY_ANCHOR: int = 0
METHODS: tuple[str, ...] = ("naive", "bh", "rc", "spa")
# 合成床：候选数、观测数、序列自相关、截面相关
BED_GRID: tuple[tuple[int, int, float, float], ...] = (
    (15, 300, 0.0, 0.0),
    (100, 300, 0.0, 0.0),
    (1000, 300, 0.0, 0.0),
    (100, 300, 0.3, 0.5),
    (1000, 300, 0.3, 0.5),
    (1000, 1000, 0.3, 0.5),
)
BED_EXPERIMENTS: int = 100
BED_SEED_BASE: int = 10_000
PLANTED_MEAN: float = 0.40
PLANTED_CANDIDATES: int = 3
BED_STAGES: tuple[float, ...] = (0.0, PLANTED_MEAN)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="多重检验校正族实证")
    parser.add_argument("--stage", default="all", help="bed / factors / sweep / all")
    parser.add_argument("--market", default="swap,spot", help="市场（逗号分隔）")
    parser.add_argument("--timeframe", default="1d", help="周期")
    parser.add_argument("--replications", type=int, default=499, help="bootstrap 次数")
    parser.add_argument("--no-save", action="store_true", help="不落盘")
    return parser.parse_args()


def _ar1(
    rng: np.random.Generator, observations: int, columns: int, phi: float
) -> np.ndarray:
    """AR(1) 噪声矩阵（列 = 候选）：phi=0 即 iid。"""
    noise = rng.standard_normal((observations, columns))
    values = np.empty_like(noise)
    values[0] = noise[0]
    scale = float(np.sqrt(1.0 - phi * phi))
    for index in range(1, observations):
        values[index] = phi * values[index - 1] + scale * noise[index]
    return values


def simulate(
    rng: np.random.Generator,
    observations: int,
    candidates: int,
    phi: float,
    rho: float,
    planted_mean: float,
) -> pd.DataFrame:
    """零假设床：共同因子 + 个体噪声；种信号时给前几个候选加同一均值。"""
    common = _ar1(rng, observations, 1, phi)
    idiosyncratic = _ar1(rng, observations, candidates, phi)
    values = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic
    if planted_mean:
        values[:, :PLANTED_CANDIDATES] += planted_mean
    columns = {f"c{index}": values[:, index] for index in range(candidates)}
    return pd.DataFrame(columns)


def rejections(frame: pd.DataFrame, replications: int, seed: int) -> dict[str, float]:
    """一次实验里四种做法是否拒绝（1/0）。"""
    p_values = naive_p_values(frame)
    bh = benjamini_hochberg(p_values, alpha=DEFAULT_ALPHA)
    check = reality_check(frame, replications=replications, seed=seed)
    spa = hansen_spa(frame, replications=replications, seed=seed)
    return {
        "naive": float(bool(float(p_values.min()) <= DEFAULT_ALPHA)),
        "bh": float(len(bh.labels) > 0),
        "rc": float(bool(check.p_value <= DEFAULT_ALPHA)),
        "spa": float(bool(spa.p_consistent <= DEFAULT_ALPHA)),
    }


def bed_table(replications: int) -> pd.DataFrame:
    """合成床：每个格子上四种做法的拒绝率（无信号 = 尺寸，种信号 = 功效）。"""
    rows: list[dict[str, object]] = []
    for candidates, observations, phi, rho in BED_GRID:
        for planted in BED_STAGES:
            rates: dict[str, list[float]] = {name: [] for name in METHODS}
            for experiment in range(BED_EXPERIMENTS):
                rng = np.random.default_rng(BED_SEED_BASE + experiment)
                frame = simulate(rng, observations, candidates, phi, rho, planted)
                outcome = rejections(
                    frame, replications, seed=BED_SEED_BASE + experiment
                )
                for name in METHODS:
                    rates[name].append(outcome[name])
            rows.append(
                {
                    "候选数": float(candidates),
                    "观测数": float(observations),
                    "序列自相关": phi,
                    "截面相关": rho,
                    "种信号": bool(planted),
                    **{name: float(np.mean(rates[name])) for name in METHODS},
                }
            )
    return pd.DataFrame(rows)


def factor_matrix(
    panel: Panel, mask: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """7 个主池因子在同一网格上的逐期多空价差（钱的口径）与汇总表。"""
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for key in MAIN_POOL_KEYS:
        spec = FACTOR_LIBRARY[key]
        stats = portfolio_stats(
            spec,
            panel,
            mask,
            PortfolioOptions(horizon=horizon, quantiles=5, anchor=FAMILY_ANCHOR),
        )
        if stats.periods == 0:
            continue
        frames.append(pd.DataFrame({spec.label: stats.spread_returns}))
        rows.append(
            {
                "候选": spec.label,
                "期数": float(stats.periods),
                "均值(bp)": stats.gross_spread * 1e4,
                "平衡成本(bp)": stats.break_even_cost * 1e4,
            }
        )
    return pd.concat(frames, axis=1), pd.DataFrame(rows)


def verdict(
    matrix: pd.DataFrame, replications: int, label: str, *, scale: float = 1e4
) -> dict[str, object]:
    """一个候选家族上四种做法的裁决（家族 = 同一网格上的 K 个候选）。

    `scale` 是均值的单位换算：钱口径（逐期价差）乘 1e4 得 bp，IC 口径乘 1。
    """
    p_values = naive_p_values(matrix)
    best = str(p_values.idxmin())
    bh = benjamini_hochberg(p_values, alpha=DEFAULT_ALPHA)
    check = reality_check(matrix, replications=replications)
    spa = hansen_spa(matrix, replications=replications)
    return {
        "家族": label,
        "候选数": float(matrix.shape[1]),
        "观测数": float(matrix.shape[0]),
        "最好候选": best,
        "最好候选均值": float(matrix[best].mean()) * scale,
        "naive p": float(p_values.min()),
        "BH 拒几个": float(len(bh.labels)),
        "RC p": check.p_value,
        "SPA p(一致)": spa.p_consistent,
        "SPA p(上界)": spa.p_upper,
        "RC 显著": bool(check.p_value <= DEFAULT_ALPHA),
        "SPA 显著": bool(spa.p_consistent <= DEFAULT_ALPHA),
    }


def sweep_matrix(
    panel: Panel, mask: pd.DataFrame, market: str, timeframe: str
) -> pd.DataFrame:
    """扫参 IC 矩阵：有缓存读缓存，没有就算一遍并落盘（3276 个候选，约 6 分钟）。"""
    cache = CACHE_DIR / f"sweep_ic_{market}_{timeframe}.parquet"
    if cache.exists():
        print(f"读缓存 {cache}")
        return pd.read_parquet(cache)
    families = resolve_families()
    _, series = sweep_ic_table(panel, mask, DEFAULT_HORIZONS, families)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_parquet(cache)
    print(
        f"扫参 IC 矩阵已落盘 {cache}（{series.shape[1]} 个候选 × {series.shape[0]} 期）"
    )
    return series


def main() -> None:
    """入口：三阶段各自出表。"""
    args = parse_args()
    markets = [item.strip() for item in args.market.split(",") if item.strip()]
    print(f"bootstrap 次数 {args.replications}｜α {DEFAULT_ALPHA}｜市场 {markets}")

    if args.stage in ("bed", "all"):
        table = bed_table(args.replications)
        print("\n### 合成零假设床：拒绝率（无信号应为 α，种信号应为功效）")
        print(table.round(3).to_string(index=False))
        if not args.no_save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            table.to_parquet(RESULTS_DIR / "corrections_bed.parquet")
            print("读数落盘：data/results/breaks/corrections_bed.parquet")

    if args.stage in ("factors", "sweep", "all"):
        rows: list[dict[str, object]] = []
        for market in markets:
            panel = load_panel(args.timeframe, market=market)
            mask = tradable_mask(panel)
            print(f"\n池子：{market} {args.timeframe}，{len(panel.symbols)} 标的")
            if args.stage in ("factors", "all"):
                for horizon in FACTOR_HORIZONS:
                    matrix, summary = factor_matrix(panel, mask, horizon)
                    print(f"\n### 主池因子家族（{market}，h={horizon}，钱的口径）")
                    print(summary.round(3).to_string(index=False))
                    row = verdict(
                        matrix, args.replications, f"factors_{market}_h{horizon}"
                    )
                    rows.append(row)
                    print(pd.DataFrame([row]).round(4).to_string(index=False))
            if args.stage in ("sweep", "all"):
                matrix = sweep_matrix(panel, mask, market, args.timeframe)
                row = verdict(matrix, args.replications, f"sweep_{market}", scale=1.0)
                rows.append(row)
                print(f"\n### 扫参家族（{market}，IC 口径）")
                print(pd.DataFrame([row]).round(4).to_string(index=False))
        if rows and not args.no_save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = RESULTS_DIR / "corrections_families.parquet"
            frame = pd.DataFrame(rows)
            if path.exists():
                # 分阶段跑时合并而不是覆写（同一家族重跑则取新值）
                frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
                frame = frame.drop_duplicates(subset=["家族"], keep="last")
            frame.to_parquet(path)
            print(f"\n读数落盘：{path}（{len(frame)} 个家族）")


if __name__ == "__main__":
    main()
