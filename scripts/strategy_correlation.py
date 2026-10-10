"""策略重合度与组合：收益相关性、仓位重叠、等权组合对比。

阶段 2（本脚本）：读阶段 1（scripts/strategy_suite.py）落盘的净收益与持仓序列，
回答三个问题：
1. 各策略赚的是不是同一笔钱（收益相关性矩阵）
2. 它们的仓位有多重叠（Jaccard 矩阵）
3. 等权组合能否改善单策略的收益与回撤（组合对比）

用法：
    uv run python scripts/strategy_correlation.py
    uv run python scripts/strategy_correlation.py --symbols BTC/USDT,ETH/USDT

口径：组合为等权（资金在成分间等分），逐根取成分净收益的算术平均，不重新计费
（成分收益已各自扣过手续费与滑点）。表头用 ASCII，避免 Windows GBK 列宽错位。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.analytics.metrics import (
    annualize,
    annualized_sharpe,
    bars_per_year,
    calmar_ratio,
    max_drawdown,
    sortino_ratio,
    total_return,
)
from src.research.registry import STRATEGY_SHORT_NAMES, resolve_symbols

INPUT_DIR = Path("data/results/strategy_suite")
ALL_KEYS: tuple[str, ...] = ("buy-and-hold", "1", "2", "3", "4", "5")
# 组合候选：分母是「策略族」还是「纳入基准」，会直接改变相关性结论
COMBOS: dict[str, tuple[str, ...]] = {
    "All 6 equal weight": ALL_KEYS,
    "5 strategies (no buy-hold)": ("1", "2", "3", "4", "5"),
    "Trend family (2/4/5)": ("2", "4", "5"),
    "TrendRev + Donchian (per-symbol champs)": ("2", "5"),
    "TrendRev + Bollinger (opposite sign)": ("2", "3"),
}
LABEL_WIDTH = 40
CELL_WIDTH = 11


@dataclass(frozen=True)
class ComboRow:
    """一个组合（或单策略）的绩效读数。"""

    label: str
    members: int
    total_ret: float
    annual_ret: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float


def short_name(key: object) -> str:
    """取策略短名。"""
    return STRATEGY_SHORT_NAMES.get(str(key), str(key))


def load(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读阶段 1 落盘的净收益与持仓序列。"""
    stem = symbol.replace("/", "-")
    returns = pd.read_parquet(INPUT_DIR / f"{stem}_net_ret.parquet")
    positions = pd.read_parquet(INPUT_DIR / f"{stem}_pos.parquet")
    returns.columns = [short_name(c) for c in returns.columns]
    positions.columns = [short_name(c) for c in positions.columns]
    return returns, positions


def jaccard_matrix(positions: pd.DataFrame) -> pd.DataFrame:
    """仓位重合度：|A∩B| / |A∪B|，只区分「有仓位 / 无仓位」。"""
    active = positions.abs() > 0
    keys = pd.Index([str(c) for c in active.columns])
    out = pd.DataFrame(1.0, index=keys, columns=keys)
    for a in keys:
        for b in keys:
            both = int((active[a] & active[b]).sum())
            either = int((active[a] | active[b]).sum())
            out.loc[a, b] = both / either if either else float("nan")
    return out


def print_matrix(title: str, matrix: pd.DataFrame, note: str = "") -> None:
    """打印方阵。"""
    labels = [str(c) for c in matrix.columns]
    print(f"\n{title}")
    if note:
        print(note)
    header = f"{'':<{LABEL_WIDTH}}" + "".join(f"{x:>{CELL_WIDTH}}" for x in labels)
    print(header)
    print("-" * len(header))
    for label, (_, row) in zip(labels, matrix.iterrows(), strict=True):
        cells = "".join(f"{float(v):>{CELL_WIDTH}.2f}" for v in row)
        print(f"{label:<{LABEL_WIDTH}}{cells}")


def combo_row(label: str, returns: pd.DataFrame, members: tuple[str, ...]) -> ComboRow:
    """等权组合：逐根取成分净收益的算术平均。"""
    columns = [short_name(k) for k in members]
    ret = cast(pd.Series, returns[columns].mean(axis=1))
    periods = bars_per_year(ret.index)
    total = total_return(ret)
    dd = max_drawdown(ret)
    annual = annualize(total, len(ret), int(round(periods)))
    return ComboRow(
        label=label,
        members=len(members),
        total_ret=total,
        annual_ret=annual,
        sharpe=annualized_sharpe(ret, periods),
        sortino=sortino_ratio(ret, periods),
        calmar=calmar_ratio(annual, dd),
        max_dd=dd,
    )


def print_combos(title: str, combos: list[ComboRow], singles: list[ComboRow]) -> None:
    """打印组合表，末尾附单策略对照。"""
    print(f"\n{title}")
    header = (
        f"{'portfolio':<{LABEL_WIDTH}}{'n':>4}{'total':>11}{'annual':>9}"
        f"{'sharpe':>8}{'sortino':>9}{'calmar':>8}{'maxDD':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in combos:
        print(
            f"{row.label:<{LABEL_WIDTH}}{row.members:>4}{row.total_ret:>10.1%}"
            f"{row.annual_ret:>9.1%}{row.sharpe:>8.2f}{row.sortino:>9.2f}"
            f"{row.calmar:>8.2f}{row.max_dd:>9.1%}"
        )
    print("-" * len(header))
    for row in singles:
        print(
            f"{row.label:<{LABEL_WIDTH}}{row.members:>4}{row.total_ret:>10.1%}"
            f"{row.annual_ret:>9.1%}{row.sharpe:>8.2f}{row.sortino:>9.2f}"
            f"{row.calmar:>8.2f}{row.max_dd:>9.1%}"
        )


def analyze(symbol: str) -> None:
    """跑一个标的的相关性与组合分析。"""
    returns, positions = load(symbol)
    print(
        f"\n{'=' * 78}\n标的 {symbol} | {len(returns)} 根 | "
        f"{returns.index[0]:%Y-%m-%d} -> {returns.index[-1]:%Y-%m-%d}\n{'=' * 78}"
    )

    print_matrix(
        "=== 收益相关性（Pearson，逐根净收益）===",
        returns.corr(),
        ">0.7 视为赚同一笔钱；组合只在该值明显低于 1 时才有分散价值",
    )
    print_matrix(
        "=== 仓位重合度（Jaccard = |A∩B| / |A∪B|）===",
        jaccard_matrix(positions),
        "持仓时间越短的策略该值天然越小，横向比较时注意各列 expo",
    )

    combos = [combo_row(label, returns, keys) for label, keys in COMBOS.items()]
    singles = [combo_row(short_name(k), returns, (k,)) for k in ALL_KEYS]
    print_combos("=== 等权组合 vs 单策略 ===", combos, singles)

    best = max(singles, key=lambda r: r.sharpe)
    avg_sharpe = sum(r.sharpe for r in singles) / len(singles)
    best_combo = max(combos, key=lambda r: r.sharpe)
    print(
        f"\n分散效应：最好单策略 {best.label} 夏普 {best.sharpe:.2f}，"
        f"单策略平均 {avg_sharpe:.2f}，最好组合 {best_combo.label} "
        f"夏普 {best_combo.sharpe:.2f}，回撤 {best_combo.max_dd:.1%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="策略重合度与等权组合分析（只读阶段 1 落盘的收益流）"
    )
    parser.add_argument(
        "--symbols", default=None, help="逗号分隔，默认 BTC/USDT 与 ETH/USDT"
    )
    args = parser.parse_args()
    for symbol in resolve_symbols(args.symbols):
        analyze(symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
