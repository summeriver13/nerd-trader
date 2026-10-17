"""carry 的风险侧：退出规则回测与爆仓情景。

回答两个「无风险」最该被检验的问题：
1. 费率转负就退出，值不值——每次切换要赚回双腿往返成本才划算；
2. 永续空头会不会被强平——1.5x 资本 = 永续 2x 杠杆，涨约 49% 触发。

分析逻辑在 src/research/funding/carry_risk.py，本脚本只负责打印与落盘。
输出：data/results/carry/carry_risk.parquet（爆仓情景表）

用法：
    uv run python scripts/carry_risk.py
    uv run python scripts/carry_risk.py --days 2900

只读本地 Parquet 缓存，不联网。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.research.funding.carry import CAPITAL_MULTIPLE, ROUND_TRIP_COST
from src.research.funding.carry_ranking import OUTPUT_DIR, list_stems
from src.research.funding.carry_risk import (
    LIQUIDATION_HORIZONS,
    RULE_WINDOWS,
    LiquidationRow,
    RuleResult,
    liquidation_rows,
    load_daily,
    rule_results,
)

FUNDING_EXCHANGE = "binance"
NAME_WIDTH = 14
DEFAULT_DAYS = 3300


def _rule(width: int) -> str:
    return "-" * width


def collect(stems: list[str], days: int) -> dict[str, dict[str, RuleResult]]:
    """每只标的 × 每条规则的结果；样本短于最长窗口两倍的标的跳过。"""
    out: dict[str, dict[str, RuleResult]] = {}
    for stem in stems:
        daily = load_daily(stem, days)
        if len(daily) < max(RULE_WINDOWS) * 2:
            continue
        out[stem] = rule_results(daily)
    return out


def print_rule_summary(by_symbol: dict[str, dict[str, RuleResult]]) -> None:
    """每条规则在全部标的上的汇总。"""
    print(f"\n=== 退出规则：费率转负就退出，值不值（{len(by_symbol)} 只标的）===")
    print(
        f"每次仓位变化算半次双腿往返（{ROUND_TRIP_COST / 2:.2%} 名义）；"
        "always 基准恰好一次完整往返"
    )
    labels = ["always", *[f"trail{window}" for window in RULE_WINDOWS]]
    header = (
        f"{'rule':<10}{'median_ret':>12}{'mean_ret':>11}{'beats':>10}"
        f"{'avg_switch':>12}{'avg_cost':>10}{'in_market':>11}"
    )
    print(header)
    print(_rule(len(header)))
    for label in labels:
        rows = [by_symbol[stem][label] for stem in by_symbol]
        returns = [row.total_return for row in rows]
        beats = sum(
            1
            for stem in by_symbol
            if by_symbol[stem][label].total_return
            > by_symbol[stem]["always"].total_return
        )
        switches = float(np.mean([row.changes for row in rows]))
        costs = float(np.mean([row.cost for row in rows]))
        in_market = float(
            np.mean([row.days_in_market / row.days_total for row in rows])
        )
        print(
            f"{label:<10}{float(np.median(returns)):>11.2%}"
            f"{float(np.mean(returns)):>11.2%}{f'{beats}/{len(rows)}':>10}"
            f"{switches:>12.1f}{costs:>10.2%}{in_market:>11.1%}"
        )
    print(
        f"\nbeats = 该规则的资本口径净收益高于 always 的标的数；"
        f"成本按资本占用 {CAPITAL_MULTIPLE:.1f}x 前的名义口径。"
    )


def print_rule_detail(by_symbol: dict[str, dict[str, RuleResult]]) -> None:
    """每只标的：always 与最优规则的对比，按改善幅度降序。"""
    entries: list[tuple[str, RuleResult, str, RuleResult]] = []
    for stem, results in by_symbol.items():
        base = results["always"]
        best_label, best = max(
            ((label, row) for label, row in results.items() if label != "always"),
            key=lambda item: item[1].total_return,
        )
        entries.append((stem, base, best_label, best))
    entries.sort(
        key=lambda item: item[3].total_return - item[1].total_return, reverse=True
    )

    print("\n每只标的的最优规则（按相对 always 的改善降序）")
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'always':>10}{'best':>10}"
        f"{'best_ret':>10}{'improve':>10}{'switch':>8}"
    )
    print(header)
    print(_rule(len(header)))
    for stem, base, best_label, best in entries:
        print(
            f"{stem:<{NAME_WIDTH}}{base.days_total:>6}{base.total_return:>10.2%}"
            f"{best_label:>10}{best.total_return:>10.2%}"
            f"{best.total_return - base.total_return:>10.2%}{best.changes:>8}"
        )


def print_liquidation_table(rows: list[LiquidationRow]) -> None:
    """爆仓情景：不同持仓期下的最坏涨幅与强平判定。"""
    if not rows:
        print("\n（无永续日线数据）")
        return
    threshold = rows[0].threshold
    print("\n=== 爆仓情景：永续空头会不会被强平 ===")
    print(
        f"1.5x 资本 = 现货 1.0 + 保证金 0.5 → 永续 2x 杠杆；维持保证金率 0.5% 下，"
        f"永续涨 {threshold:.2%} 触发强平"
    )
    print(
        "worst_Nd = 任一收盘建仓、其后 N 天内盘中高点相对建仓价的最大涨幅"
        "（已剔除零成交冻结日）"
    )
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'worst_1d':>11}{'worst_7d':>11}"
        f"{'worst_30d':>12}{'breached':>10}"
    )
    print(header)
    print(_rule(len(header)))
    for row in sorted(rows, key=lambda item: item.worst_30d, reverse=True):
        print(
            f"{row.stem:<{NAME_WIDTH}}{row.days:>6}{row.worst_1d:>11.1%}"
            f"{row.worst_7d:>11.1%}{row.worst_30d:>12.1%}"
            f"{row.breached or 'ok':>10}"
        )
    print(_rule(len(header)))
    for horizon in LIQUIDATION_HORIZONS:
        hit = sum(1 for row in rows if getattr(row, f"worst_{horizon}d") > threshold)
        print(f"持有 {horizon:>2} 天口径触及强平线：{hit}/{len(rows)}")
    print(
        "\n被强平后现货腿仍在，组合变成裸露多头——这正是「无风险」最贵的一课；"
        "且强平风险随持仓期放大，与「长期持有收息」的直觉相反。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="carry 的退出规则回测与爆仓情景")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"资金费回看天数，默认 {DEFAULT_DAYS}",
    )
    args = parser.parse_args()

    stems = list_stems(FUNDING_EXCHANGE)
    if not stems:
        print(
            f"未找到 {FUNDING_EXCHANGE} 资金费率缓存，先跑 scripts/fetch_funding_history.py"
        )
        return 1
    print(f"\n资金费源 {FUNDING_EXCHANGE}：{len(stems)} 个标的，回看 {args.days} 天")

    by_symbol = collect(stems, args.days)
    print_rule_summary(by_symbol)
    print_rule_detail(by_symbol)

    liq_rows = liquidation_rows(stems)
    print_liquidation_table(liq_rows)

    if liq_rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "carry_risk.parquet"
        pd.DataFrame([row.__dict__ for row in liq_rows]).to_parquet(path, index=False)
        print(f"\n落盘：\n  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
