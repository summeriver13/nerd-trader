"""carry 排序、跨源校验与多重检验校正：30 只永续的净 carry 值不值得做。

四张表回答四个问题：
1. 跨源校验——「费率取 Binance、执行按 OKX」这个代理假设误差多大；
2. 排名——扣掉成本与资本占用后，各标的近 365 天还剩多少；
3. 风险调整与 DSR——从 29 只里挑出来的冠军，能不能与「试 N 次碰运气」区分；
4. 等权组合与分段——分散能不能把「单标的费率转负」摊掉，各时期是否都成立；
5. 基差——真实损益 = 资金费 − Δbasis，这一项到底占多少。

分析逻辑在 src/research/carry_ranking.py 与 src/research/carry_basis.py，
本脚本只负责打印与落盘。
输出：data/results/carry/{carry_daily,carry_ranking,carry_cross_check,carry_basis}.parquet

用法：
    uv run python scripts/carry_ranking.py
    uv run python scripts/carry_ranking.py --recent-days 730 --top 15

只读本地 Parquet 缓存，不联网。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    ANNUAL_DAYS,
    annualize,
    annualized_sharpe,
    max_drawdown,
    total_return,
)
from src.research.carry_basis import BasisRow, basis_rows
from src.research.carry_ranking import (
    COMPARISON_EXCHANGE,
    LONG_HISTORY_EXCHANGE,
    OUTPUT_DIR,
    CrossRow,
    RankRow,
    RiskRow,
    SegmentRow,
    cross_row,
    list_stems,
    load_daily,
    net_recent_daily,
    portfolio_daily,
    rank_row,
    risk_rows,
    save_outputs,
    segment_rows,
)

NAME_WIDTH = 14
BP = 1e4
# 基差用全历史：telescope 求和（basis_first − basis_last）需要完整区间才有意义
BASIS_DEFAULT_DAYS = 3300


def _rule(width: int) -> str:
    return "-" * width


def print_cross_table(rows: list[CrossRow]) -> None:
    """跨源同期对比表。"""
    print(
        f"\n=== 跨源校验：同期 {LONG_HISTORY_EXCHANGE} 费率 vs {COMPARISON_EXCHANGE} 费率 ==="
    )
    print("同一时段（两所都有数据的重叠期）按日求和的费率对比；diff = Binance - OKX")
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'binance_bp':>12}{'okx_bp':>10}"
        f"{'diff_bp':>10}{'annual_diff':>13}{'corr':>8}{'sign_agree':>12}"
    )
    print(header)
    print(_rule(len(header)))
    for row in rows:
        print(
            f"{row.stem:<{NAME_WIDTH}}{row.days:>6}{row.binance_bp:>12.4f}"
            f"{row.okx_bp:>10.4f}{row.diff_bp:>10.4f}{row.annual_diff:>12.2%}"
            f"{row.corr:>8.3f}{row.sign_agree:>12.1%}"
        )
    if not rows:
        print("（无足够重叠期）")
        return
    diffs = np.array([row.diff_bp for row in rows])
    corrs = np.array([row.corr for row in rows])
    print(_rule(len(header)))
    print(
        f"{'median':<{NAME_WIDTH}}{'':>6}{'':>12}{'':>10}"
        f"{float(np.median(diffs)):>10.4f}{'':>13}{float(np.median(corrs)):>8.3f}"
    )
    print(
        f"\n费率单位 bp/日（1bp = 0.01%）。日均差中位 {float(np.median(diffs)):.4f} bp、"
        f"相关中位 {float(np.median(corrs)):.3f}；annual_diff = 日均差 × 365。"
    )


def print_rank_table(rows: list[RankRow], recent_days: int) -> None:
    """净 carry 排名表。"""
    print(f"\n=== carry 净收益排名（按最近 {recent_days} 天口径降序）===")
    print(
        "recent_net 已扣双腿进出费与再平衡并按资本占用 1.5x 折算；hist_net 为全历史口径"
    )
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>7}{'hist_days':>10}{'recent_net':>12}"
        f"{'hist_net':>10}{'roll_p5':>10}{'neg_share':>11}{'pos_share':>11}{'start':>12}"
    )
    print(header)
    print(_rule(len(header)))
    for row in rows:
        report = row.report
        start = f"{report.start:%Y-%m-%d}"
        print(
            f"{report.symbol:<{NAME_WIDTH}}{row.recent_days:>7}{report.days:>10}"
            f"{row.recent_net:>11.2%}{report.net_annual:>10.2%}"
            f"{report.rolling_p5:>10.2%}{report.negative_share:>11.1%}"
            f"{report.positive_share:>11.1%}{start:>12}"
        )


def print_risk_table(rows: list[RiskRow], expected_max: float) -> None:
    """风险调整指标与去膨胀夏普表。"""
    print("\n=== 风险调整与多重检验校正（近期窗口，已扣成本）===")
    print(
        f"零假设「{len(rows)} 次试验的最大夏普」期望值：{expected_max:.4f} 逐日 / "
        f"{expected_max * float(np.sqrt(ANNUAL_DAYS)):.2f} 年化"
        f"（sr/bar 低于它就无法与运气区分）"
    )
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'ann_vol':>10}{'sharpe':>9}"
        f"{'max_dd':>10}{'sr/bar':>9}{'dsr':>8}{'obs':>6}{'pass':>7}"
    )
    print(header)
    print(_rule(len(header)))
    for row in sorted(rows, key=lambda item: item.dsr, reverse=True):
        print(
            f"{row.stem:<{NAME_WIDTH}}{row.days:>6}{row.annual_vol:>10.2%}"
            f"{row.annual_sharpe:>9.2f}{row.max_dd:>10.2%}{row.daily_sharpe:>9.4f}"
            f"{row.dsr:>8.3f}{row.observations:>6}{'PASS' if row.passed else 'fail':>7}"
        )
    passed = sum(1 for row in rows if row.passed)
    print(_rule(len(header)))
    print(
        f"\n通过 DSR 门槛（0.95）的标的：{passed}/{len(rows)}。"
        f"未通过 = 该标的的夏普无法与「在同样多次试验里靠运气拿到」区分。"
    )


def print_portfolio(
    wide: pd.DataFrame, segments: list[SegmentRow], recent_days: int
) -> None:
    """等权组合的区间表现与分段稳定性。"""
    portfolio = portfolio_daily(wide)
    if portfolio.empty:
        print("\n（组合成员数不足，未生成组合统计）")
        return
    recent = net_recent_daily(portfolio, recent_days)
    print(f"\n=== 等权组合（{wide.shape[1]} 只标的，{len(portfolio)} 天）===")
    print("组合日收益 = 当天有结算的标的的等权平均；年化与回撤按资本占用 1.5x 折算")
    print(
        f"全区间：年化 {annualize(total_return(portfolio), len(portfolio)) / 1.5:>8.2%}"
        f" | 夏普 {annualized_sharpe(portfolio, ANNUAL_DAYS):>5.2f}"
        f" | 最大回撤 {max_drawdown(portfolio):>8.2%}"
        f"（名义口径，未折算）"
    )
    print(
        f"近 {len(recent)} 天：年化 {annualize(total_return(recent), len(recent)) / 1.5:>8.2%}"
        f" | 夏普 {annualized_sharpe(recent, ANNUAL_DAYS):>5.2f}"
        f" | 最大回撤 {max_drawdown(recent):>8.2%}"
    )

    print("\n分段（日历等分；positive 是段内日均费率为正的标的数）")
    header = (
        f"{'segment':<18}{'days':>7}{'members':>9}{'positive':>10}"
        f"{'annual':>10}{'sharpe':>9}"
    )
    print(header)
    print(_rule(len(header)))
    for row in segments:
        print(
            f"{row.label:<18}{row.days:>7}{row.members:>9}{row.positive:>10}"
            f"{row.annual:>9.2%}{row.sharpe:>9.2f}"
        )


def print_basis_table(rows: list[BasisRow]) -> None:
    """基差维度：真实损益 = 资金费 − Δbasis。"""
    print("\n=== 基差维度：真实损益 = 资金费 − Δbasis ===")
    print(
        "basis = (永续 − 现货) / 现货；累计贡献 = basis_first − basis_last（telescope 求和）；"
        "ratio = |基差贡献| / |资金费贡献|，drop 为剔除的异常日数"
    )
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'basis_mean':>12}{'basis_std':>11}"
        f"{'funding_sum':>13}{'basis_sum':>11}{'adjusted':>11}{'ratio':>8}{'drop':>6}"
    )
    print(header)
    print(_rule(len(header)))
    for row in rows:
        ratio = (
            abs(row.basis_sum) / abs(row.funding_sum)
            if row.funding_sum
            else float("nan")
        )
        print(
            f"{row.stem:<{NAME_WIDTH}}{row.days:>6}{row.basis_mean:>12.4%}"
            f"{row.basis_std:>11.4%}{row.funding_sum:>13.2%}{row.basis_sum:>11.2%}"
            f"{row.adjusted_sum:>11.2%}{ratio:>8.3f}{row.dropped:>6}"
        )
    if not rows:
        print("（无可用标的：缺现货或数据不足）")
        return
    ratios = [
        abs(row.basis_sum) / abs(row.funding_sum) for row in rows if row.funding_sum
    ]
    abs_basis = [abs(row.basis_sum) for row in rows]
    if ratios:
        print(_rule(len(header)))
        print(
            f"中位：|基差贡献| {float(np.median(abs_basis)):.2%}、"
            f"ratio {float(np.median(ratios)):.3f}、最大 ratio {max(ratios):.3f}"
        )
    print(
        "\n基差对长期收益的贡献很小（相对资金费），可以忽略；"
        "但 basis_std 是永续空头的日间保证金风险来源。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="carry 净收益排名、跨源校验与多重检验校正"
    )
    parser.add_argument(
        "--recent-days", type=int, default=ANNUAL_DAYS, help="近期口径天数，默认 365"
    )
    parser.add_argument(
        "--top", type=int, default=0, help="排名表只显示前 N 行，0 为全部"
    )
    parser.add_argument(
        "--basis-days",
        type=int,
        default=BASIS_DEFAULT_DAYS,
        help=f"基差口径的资金费天数，默认 {BASIS_DEFAULT_DAYS}（全历史）",
    )
    args = parser.parse_args()

    stems = list_stems(LONG_HISTORY_EXCHANGE)
    if not stems:
        print(
            f"未找到 {LONG_HISTORY_EXCHANGE} 资金费率缓存，先跑 scripts/fetch_funding_history.py"
        )
        return 1
    print(
        f"\n长历史源 {LONG_HISTORY_EXCHANGE}：{len(stems)} 个标的"
        f" | 对照源 {COMPARISON_EXCHANGE}：{len(list_stems(COMPARISON_EXCHANGE))} 个标的"
    )

    wide = load_daily(LONG_HISTORY_EXCHANGE, stems)

    cross_rows = [row for stem in stems if (row := cross_row(stem)) is not None]
    print_cross_table(cross_rows)

    rank_rows = sorted(
        (rank_row(stem, args.recent_days) for stem in stems),
        key=lambda row: row.recent_net,
        reverse=True,
    )
    print_rank_table(
        rank_rows[: args.top] if args.top > 0 else rank_rows, args.recent_days
    )

    risk_list, expected_max = risk_rows(wide, args.recent_days, len(stems))
    print_risk_table(risk_list, expected_max)

    segments = segment_rows(wide)
    print_portfolio(wide, segments, args.recent_days)

    basis_list = basis_rows(stems, args.basis_days)
    print_basis_table(basis_list)

    risk_by_stem = {row.stem: row for row in risk_list}
    written = save_outputs(wide, rank_rows, cross_rows, risk_by_stem)
    if basis_list:
        basis_path = OUTPUT_DIR / "carry_basis.parquet"
        pd.DataFrame([row.__dict__ for row in basis_list]).to_parquet(
            basis_path, index=False
        )
        written.append(basis_path)
    print("\n落盘：")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
