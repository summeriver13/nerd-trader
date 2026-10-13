"""
scripts/cost_sensitivity_study.py
第 19 章收尾：把「成本」从常数换成实测值后，哪些结论会翻。

本仓所有「成本不允许」的结论都建在 10bp/边（5bp 手续费 + 5bp **假设**的滑点）上，而
第 22 节的 1 分钟实测给出单边 5.0~5.4bp（价差只占 0.2~0.4bp，其余是手续费）。这里把
成本当自变量扫一遍：

- 每个因子的净收益 = 毛收益 − 4 × 换手 × 单边成本（成本线性，故一格读取全格）；
- 平衡成本（毛收益归零的那一点）与实测成本并排，看差距是「差一点」还是「差一个数量级」；
- 实测 5.4bp 取自最不流动的那 10 个已测标的（$14.8M/日），**池尾只会更贵**，所以 5.4bp
  是乐观下界：若在这个下界上结论仍不翻，该结论就是稳的。

用法：uv run python scripts/cost_sensitivity_study.py [--market swap] [--no-save]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.panel import Panel, load_panel, tradable_mask
from src.research.factors.portfolio import PortfolioOptions, portfolio_stats

RESULTS_DIR: Path = Path("data/results/breaks")
MAIN_POOL_KEYS: tuple[str, ...] = (
    "rev_3",
    "mom_7",
    "mom_30",
    "vol_30",
    "liq_30",
    "dth_60",
    "ffd_d0p4",
)
HORIZONS: tuple[int, ...] = (1, 7)
COST_GRID_BP: tuple[float, ...] = (0.0, 2.0, 5.4, 8.0, 10.0)
MEASURED_COST_BP: float = 5.4
REPO_COST_BP: float = 10.0
REBALANCE_TRADES: int = 4


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="成本敏感度：实测成本 vs 假设成本")
    parser.add_argument("--market", default="swap", help="市场（swap / spot）")
    parser.add_argument("--timeframe", default="1d", help="周期")
    parser.add_argument("--no-save", action="store_true", help="不落盘")
    return parser.parse_args()


def sweep(panel: Panel, mask: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """一个调仓周期下，各因子在整条成本网格上的净值（bp/期）。"""
    rows: list[dict[str, float | str]] = []
    for key in MAIN_POOL_KEYS:
        spec = FACTOR_LIBRARY[key]
        stats = portfolio_stats(
            spec, panel, mask, PortfolioOptions(horizon=horizon, quantiles=5)
        )
        if stats.periods == 0:
            continue
        row: dict[str, float | str] = {
            "因子": spec.label,
            "期数": float(stats.periods),
            "毛(bp/期)": stats.gross_spread * 1e4,
            "换手": stats.turnover,
            "平衡成本(bp)": stats.break_even_cost * 1e4,
        }
        for cost_bp in COST_GRID_BP:
            row[f"净@{cost_bp:g}bp"] = (
                stats.gross_spread - REBALANCE_TRADES * stats.turnover * cost_bp / 1e4
            ) * 1e4
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("因子")
    return frame.assign(周期=f"h={horizon}")


def verdict(frame: pd.DataFrame, horizon: int) -> list[str]:
    """逐个因子给出「换成本后翻不翻」的判词。"""
    lines: list[str] = []
    for label, row in frame.iterrows():
        gross = float(row["毛(bp/期)"])
        net_measured = float(row[f"净@{MEASURED_COST_BP:g}bp"])
        net_repo = float(row[f"净@{REPO_COST_BP:g}bp"])
        be = float(row["平衡成本(bp)"])
        if gross <= 0:
            # 毛收益本来就负：与成本无关（换成本救不回来）
            lines.append(f"- {label}: 毛收益 {gross:+.2f}bp 本就为负，与成本无关")
        elif net_repo <= 0 < net_measured:
            lines.append(
                f"- **{label}: 翻正**（10bp 下 {net_repo:+.2f} → 5.4bp 下 {net_measured:+.2f}bp，"
                f"平衡成本 {be:.2f}bp）"
            )
        elif net_measured > 0:
            lines.append(
                f"- {label}: 两种成本下都为正（{net_repo:+.2f} → {net_measured:+.2f}bp，"
                f"平衡成本 {be:.2f}bp）"
            )
        else:
            lines.append(
                f"- {label}: 两种成本下都为负（{net_repo:+.2f} → {net_measured:+.2f}bp，"
                f"平衡成本 {be:.2f}bp）"
            )
    del horizon
    return lines


def main() -> None:
    """入口：两档周期 × 一条成本网格。"""
    args = parse_args()
    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel)
    print(
        f"池子：{args.market} {args.timeframe}，{len(panel.symbols)} 标的名，可交易观测 {int(mask.sum().sum())}"
    )
    print(
        f"成本网格：{[f'{c:g}' for c in COST_GRID_BP]} bp/边｜实测下界 {MEASURED_COST_BP}bp"
    )

    frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        frame = sweep(panel, mask, horizon)
        print(f"\n### h={horizon}（净收益 bp/期，成本线性故一次算全格）")
        print(frame.round(3).to_string())
        print("判词：")
        for line in verdict(frame, horizon):
            print(line)
        frames.append(frame)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"cost_sensitivity_{args.market}_{args.timeframe}.parquet"
        pd.concat(frames).to_parquet(path)
        print(f"\n读数落盘：{path}")


if __name__ == "__main__":
    main()
