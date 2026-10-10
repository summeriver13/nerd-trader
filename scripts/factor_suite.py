"""
scripts/factor_suite.py
截面因子统一测试台：读本地面板，跑 IC / 分位组合 / 相关 / 容量 / 幸存者偏差。

用法：
    uv run python scripts/factor_suite.py
    uv run python scripts/factor_suite.py --horizons 1,7,30 --min-listed-days 365
    uv run python scripts/factor_suite.py --no-save

只读 data/storage/crypto/okx/<market>/<timeframe>/ 下的缓存，不触发采集。
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import pandas as pd

from src.research.factors.panel import Panel, UniverseRule, load_panel
from src.research.factors.portfolio import DEFAULT_ONE_WAY_COST, PortfolioOptions
from src.research.factors.report import (
    DEFAULT_HORIZONS,
    DEFAULT_PORTFOLIO_HORIZON,
    DEFAULT_QUANTILES,
    FactorReport,
    build_report,
    save_outputs,
)

FACTOR_WIDTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="截面因子统一测试台")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument("--market", default="spot", help="市场层，默认 spot")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="前向窗口（逗号分隔），默认 1,7,30",
    )
    parser.add_argument(
        "--portfolio-horizon",
        type=int,
        default=DEFAULT_PORTFOLIO_HORIZON,
        help=f"分位组合调仓周期，默认 {DEFAULT_PORTFOLIO_HORIZON}",
    )
    parser.add_argument(
        "--quantiles", type=int, default=DEFAULT_QUANTILES, help="分位档数，默认 5"
    )
    parser.add_argument(
        "--min-notional",
        type=float,
        default=1_000_000.0,
        help="日名义额门槛（美元，30 日中位），默认 100 万",
    )
    parser.add_argument(
        "--min-listed-days", type=int, default=180, help="上市天数门槛，默认 180"
    )
    parser.add_argument(
        "--cost-bp",
        type=float,
        default=None,
        help=f"单边成本（bp），默认 {DEFAULT_ONE_WAY_COST * 1e4:.1f}",
    )
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def print_universe(panel: Panel, report: FactorReport) -> None:
    """池子口径与面板宽度：任何截面结论都要先看这段。"""
    counts = report.universe["tradable"]
    print(f"\n截面因子统一测试 | {panel.timeframe} | {len(panel.index)} 根")
    print(
        f"时间 {cast(pd.Timestamp, panel.index[0]).date()}"
        f" -> {cast(pd.Timestamp, panel.index[-1]).date()}"
    )
    print(
        f"缓存 {len(panel.symbols)} 标的 | 静态剔除 {len(panel.excluded)} 个非加密仪器"
    )
    print(f"池子口径：{report.rule_label}")
    print(
        f"面板宽度：首 {int(counts.iloc[0])} -> 末 {int(counts.iloc[-1])}"
        f" | 峰值 {int(counts.max())} | 均值 {counts.mean():.1f}"
    )
    for floor in (10, 30):
        print(f"  宽度 >= {floor} 只的期次占比 {(counts >= floor).mean():.1%}")


def print_ic(report: FactorReport) -> None:
    """RankIC 表：t 值来自非重叠子样本，判定用 Bonferroni 校正后的门槛。"""
    print(
        f"\n=== RankIC（t 值取非重叠子样本）| 试验 {report.trials} 次"
        f"，Bonferroni 双侧 5% 门槛 |t| > {report.critical_t:.2f} ==="
    )
    print(
        f"{'因子':<{FACTOR_WIDTH}}{'h':>3}{'IC均值':>9}{'t(非重叠)':>11}"
        f"{'独立期数':>9}{'截面N':>8}{'判定':>7}"
    )
    for _, row in report.ic.iterrows():
        verdict = "通过" if bool(row["passes"]) else "—"
        print(
            f"{row['factor']:<{FACTOR_WIDTH}}{int(row['horizon']):>3}"
            f"{row['mean_ic']:>9.4f}{row['t_stat']:>11.2f}"
            f"{int(row['independent']):>9}{row['names']:>8.1f}{verdict:>7}"
        )


def print_portfolio(report: FactorReport) -> None:
    """分位组合表：多空价差与只多腿并列，单调性说明只多腿是否可信。

    这里的「只多年化」是算术折算，不是可复利收益；判定要看 print_benchmark。
    """
    first = report.portfolio.iloc[0]
    print(
        f"\n=== 分位组合（非重叠期次，名义基准，{first['quantiles']} 分位"
        f"，成本 {first['cost_one_way'] * 1e4:.1f}bp/边）==="
    )
    print(
        f"{'因子':<{FACTOR_WIDTH}}{'期数':>5}{'Q高':>9}{'Q低':>9}{'多空净':>9}"
        f"{'多空年化':>10}{'只多净':>9}{'只多年化':>10}{'单调性':>8}{'换手':>7}"
    )
    for _, row in report.portfolio.iterrows():
        print(
            f"{row['factor']:<{FACTOR_WIDTH}}{int(row['periods']):>5}"
            f"{row['top_leg']:>9.4f}{row['bottom_leg']:>9.4f}"
            f"{row['net_spread']:>9.4f}{row['annual_net']:>10.3f}"
            f"{row['long_only_net']:>9.4f}{row['long_only_annual']:>10.3f}"
            f"{row['monotonicity']:>8.2f}{row['long_turnover']:>7.3f}"
        )


def print_benchmark(report: FactorReport) -> None:
    """只多腿 vs 等权全体与 BTC：不比基准就分不清收益来自因子还是来自市场 beta。

    「只多算术」是期均值×每年期数，年化波动 95% 的腿上它比实际到手高十几个
    百分点；「只多复利」才是拿到手的。判定看「超额」与「BTC复利」两列：
    跑输等权说明因子无用，跑赢等权但跑输 BTC 说明不如直接买 BTC。
    """
    print("\n=== 只多腿 vs 市场基准（等权全体与 BTC 买入持有，同期同起点）===")
    print(
        f"{'因子':<{FACTOR_WIDTH}}{'只多算术':>10}{'只多复利':>10}{'只多夏普':>10}"
        f"{'只多回撤':>10}{'等权复利':>10}{'超额':>10}{'BTC复利':>10}{'BTC夏普':>10}"
    )
    for _, row in report.portfolio.iterrows():
        print(
            f"{row['factor']:<{FACTOR_WIDTH}}{row['long_only_annual']:>10.3f}"
            f"{row['long_compounded']:>10.3f}{row['long_sharpe']:>10.2f}"
            f"{row['long_drawdown']:>10.3f}{row['market_compounded']:>10.3f}"
            f"{row['excess_compounded']:>10.3f}{row['btc_compounded']:>10.3f}"
            f"{row['btc_sharpe']:>10.2f}"
        )


def print_correlation(report: FactorReport) -> None:
    """因子相关矩阵：两个 IC 都高的因子若是同一条信号，组合不会更好只是多付成本。"""
    print("\n=== 因子截面 rank 相关（逐期算再对期次取平均）===")
    frame = report.correlation
    print(f"{'':<{FACTOR_WIDTH}}" + "".join(f"{c:>9}" for c in frame.columns))
    for key, row in frame.iterrows():
        print(f"{key:<{FACTOR_WIDTH}}" + "".join(f"{v:>9.2f}" for v in row))


def print_capacity(report: FactorReport) -> None:
    """容量：单标的参与率上限下的组合可容纳资金，最挤的那一期决定容量。"""
    print("\n=== 容量（1% 参与率，最挤的一期）===")
    for key, row in report.capacity.iterrows():
        print(f"{key:<{FACTOR_WIDTH}}{row['capacity_usd']:>14,.0f} USD")


def print_survivorship(report: FactorReport) -> None:
    """幸存者偏差代理：缓存里 0 个退市标的，上市初期的收益不可当证据。"""
    print("\n=== 幸存者偏差代理（上市初期 vs 成熟期）===")
    print(f"{'h':>4}{'初期均值':>11}{'成熟期均值':>12}{'缺口':>10}{'初期观测':>10}")
    for _, row in report.survivorship.iterrows():
        print(
            f"{int(row['horizon']):>4}{row['young_mean']:>11.4f}"
            f"{row['mature_mean']:>12.4f}{row['gap']:>10.4f}"
            f"{int(row['young_count']):>10}"
        )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    horizons = tuple(int(item) for item in args.horizons.split(","))
    rule = UniverseRule(
        min_notional_usd=args.min_notional, min_listed_days=args.min_listed_days
    )
    panel = load_panel(args.timeframe, rule, market=args.market)
    options = PortfolioOptions(
        horizon=args.portfolio_horizon,
        quantiles=args.quantiles,
        cost_one_way=(
            DEFAULT_ONE_WAY_COST if args.cost_bp is None else args.cost_bp / 1e4
        ),
    )
    report = build_report(
        panel, rule, horizons=horizons, portfolio_horizon=options.horizon
    )

    print_universe(panel, report)
    print_ic(report)
    print_portfolio(report)
    print_benchmark(report)
    print_correlation(report)
    print_capacity(report)
    print_survivorship(report)

    if not args.no_save:
        print("\n落盘：")
        for path in save_outputs(report):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
