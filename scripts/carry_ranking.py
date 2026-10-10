"""carry 排序与跨源校验：30 只永续的净 carry 排名，以及 Binance 费率代理 OKX 的误差。

两个问题：
1. 30 只标的里，扣掉手续费与再平衡成本后，carry 还剩多少、谁最高（净年化排序）；
2. 「费率取 Binance、执行按 OKX」这一跨源代理假设误差多大——用两所同期 103 天
   重叠期做逐日对比，给出日均费率差、相关性与符号一致率。

口径：收益按日求和（结算间隔 4h/8h 混用时仍可比），成本与资本占用沿用
src/research/carry.py 的假设；排序同时给全历史与最近 365 天两列，后者才是可比口径。

用法：
    uv run python scripts/carry_ranking.py
    uv run python scripts/carry_ranking.py --recent-days 730 --top 15

只读本地 Parquet 缓存，不联网。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.analytics.metrics import ANNUAL_DAYS, annualize, total_return
from src.config.paths import CRYPTO_ROOT, MARKET_FUNDING, crypto_dataset_dir
from src.research.carry import CarryReport, build_carry_report
from src.utils.data_sources.funding import daily_rates

FUNDING_TIMEFRAME = "8h"
FUNDING_EXCHANGES = ("binance", "okx")
LONG_HISTORY_EXCHANGE = "binance"  # 排序主口径：唯一有长历史的源
# 跨源对比需要足够的重叠期，太短的相关性没有意义
MIN_OVERLAP_DAYS = 30
BP = 1e4
NAME_WIDTH = 14


@dataclass(frozen=True)
class CrossRow:
    """一只标的的跨源同期对比。"""

    stem: str
    days: int
    binance_bp: float
    okx_bp: float
    diff_bp: float
    annual_diff: float
    corr: float
    sign_agree: float


@dataclass(frozen=True)
class RankRow:
    """一只标的的 carry 排名数据。"""

    report: CarryReport
    recent_net: float
    recent_days: int


def cache_dir(exchange_id: str) -> Path:
    """资金费率缓存目录。"""
    return crypto_dataset_dir(
        exchange_id, MARKET_FUNDING, FUNDING_TIMEFRAME, root=CRYPTO_ROOT
    )


def list_stems(exchange_id: str) -> list[str]:
    """扫描该交易所缓存目录下的全部标的。"""
    directory = cache_dir(exchange_id)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def load_rates(exchange_id: str, stem: str) -> pd.Series:
    """读一只标的的费率序列（index 为 UTC 结算时刻）。"""
    frame = pd.read_parquet(cache_dir(exchange_id) / f"{stem}.parquet")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return cast(pd.Series, frame.sort_index()["rate"])


def cross_check(stem: str) -> CrossRow | None:
    """同期对比两所的费率：重叠期不足或任一缺失则返回 None。"""
    binance_daily = daily_rates(load_rates("binance", stem))
    okx_daily = daily_rates(load_rates("okx", stem))
    common = binance_daily.index.intersection(okx_daily.index)
    if len(common) < MIN_OVERLAP_DAYS:
        return None
    left = cast(pd.Series, binance_daily.loc[common]).dropna()
    right = cast(pd.Series, okx_daily.loc[common]).dropna()
    shared = left.index.intersection(right.index)
    left = cast(pd.Series, left.loc[shared])
    right = cast(pd.Series, right.loc[shared])
    if len(shared) < MIN_OVERLAP_DAYS:
        return None
    deviation = left - right
    return CrossRow(
        stem=stem,
        days=len(shared),
        binance_bp=float(left.mean()) * BP,
        okx_bp=float(right.mean()) * BP,
        diff_bp=float(deviation.mean()) * BP,
        annual_diff=annualize(total_return(left), len(left))
        - annualize(total_return(right), len(right)),
        corr=float(cast(float, left.corr(right))),
        sign_agree=float((np.sign(left) == np.sign(right)).mean()),
    )


def rank_row(stem: str, recent_days: int) -> RankRow:
    """用长历史交易所的费率算一只标的的 carry 报告与近期口径。"""
    rates = load_rates(LONG_HISTORY_EXCHANGE, stem)
    report = build_carry_report(stem, rates)
    daily = daily_rates(rates)
    recent = cast(pd.Series, daily.iloc[-recent_days:])
    recent_net = (
        annualize(total_return(recent), len(recent)) / 1.5 if len(recent) else 0.0
    )
    return RankRow(report=report, recent_net=recent_net, recent_days=len(recent))


def print_cross_table(rows: list[CrossRow]) -> None:
    """打印跨源同期对比表。"""
    print("\n=== 跨源校验：同期 Binance 费率 vs OKX 费率 ===")
    print("同一时段（两所都有数据的重叠期）按日求和的费率对比；diff = Binance - OKX")
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>6}{'binance_bp':>12}{'okx_bp':>10}"
        f"{'diff_bp':>10}{'annual_diff':>13}{'corr':>8}{'sign_agree':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.stem:<{NAME_WIDTH}}{row.days:>6}{row.binance_bp:>12.4f}"
            f"{row.okx_bp:>10.4f}{row.diff_bp:>10.4f}{row.annual_diff:>12.2%}"
            f"{row.corr:>8.3f}{row.sign_agree:>12.1%}"
        )
    if not rows:
        return
    diffs = np.array([row.diff_bp for row in rows])
    corrs = np.array([row.corr for row in rows])
    print("-" * len(header))
    print(
        f"{'median':<{NAME_WIDTH}}{'':>6}{'':>12}{'':>10}"
        f"{float(np.median(diffs)):>10.4f}{'':>13}{float(np.median(corrs)):>8.3f}"
    )
    print(
        f"\n费率单位 bp/日（1bp = 0.01%）。日均差中位 {float(np.median(diffs)):.4f} bp、"
        f"相关中位 {float(np.median(corrs)):.3f}；"
        f"差值的年化影响 = 日均差 × 365，可见量级。"
    )


def print_rank_table(rows: list[RankRow], recent_days: int) -> None:
    """打印 carry 净收益排名表（按近期口径排序）。"""
    print(f"\n=== carry 净收益排名（按最近 {recent_days} 天口径降序）===")
    print(
        "net_annual 已扣双腿进出费与再平衡，并按资本占用 1.5x 折算；"
        "all_net = 该标的全历史口径（天数不同不可直接横比）"
    )
    header = (
        f"{'symbol':<{NAME_WIDTH}}{'days':>7}{'hist_days':>10}{'recent_net':>12}"
        f"{'all_net':>10}{'roll_p5':>10}{'neg_share':>11}{'pos_share':>11}{'start':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        report = row.report
        # datetime 的 __format__ 走 strftime，不能在同一格式串里再写对齐宽度
        start = f"{report.start:%Y-%m-%d}"
        print(
            f"{report.symbol:<{NAME_WIDTH}}{row.recent_days:>7}{report.days:>10}"
            f"{row.recent_net:>11.2%}{report.net_annual:>10.2%}"
            f"{report.rolling_p5:>10.2%}{report.negative_share:>11.1%}"
            f"{report.positive_share:>11.1%}{start:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="carry 净收益排名与跨源费率校验")
    parser.add_argument(
        "--recent-days", type=int, default=ANNUAL_DAYS, help="近期口径天数，默认 365"
    )
    parser.add_argument("--top", type=int, default=0, help="只显示前 N 行，0 为全部")
    args = parser.parse_args()

    stems = list_stems(LONG_HISTORY_EXCHANGE)
    if not stems:
        print(
            f"未找到 {LONG_HISTORY_EXCHANGE} 资金费率缓存，先跑 scripts/fetch_funding_history.py"
        )
        return 1

    print(
        f"\n长历史源 {LONG_HISTORY_EXCHANGE}：{len(stems)} 个标的"
        f" | 对照源 okx：{len(list_stems('okx'))} 个标的"
    )

    cross_rows = [row for stem in stems if (row := cross_check(stem)) is not None]
    print_cross_table(cross_rows)

    rank_rows = sorted(
        (rank_row(stem, args.recent_days) for stem in stems),
        key=lambda row: row.recent_net,
        reverse=True,
    )
    if args.top > 0:
        rank_rows = rank_rows[: args.top]
    print_rank_table(rank_rows, args.recent_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
