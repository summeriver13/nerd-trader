"""
scripts/maker_spread_study.py
两腿被动挂单的组合层：反转价差用 maker 进场能不能真赚钱。

用法：
    uv run python scripts/maker_spread_study.py
    uv run python scripts/maker_spread_study.py --offsets 0.5,2,5 --horizons 1440
    uv run python scripts/maker_spread_study.py --entry-fee-bp 10   # 对照：进场也吃价
    uv run python scripts/maker_spread_study.py --no-save

只读本地 1m 缓存与日线面板，不触发采集。
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import pandas as pd

from src.research.maker_conditional import MAKER_FEE_BP, TAKER_FEE_BP
from src.research.maker_spread import (
    SpreadOptions,
    daily_frame,
    load_legs,
    save_outputs,
    summary_frame,
)
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

MINUTES_PER_DAY = 1440.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="两腿被动挂单（反转价差）研究")
    parser.add_argument(
        "--offsets", default="0.5,1,2,5,10", help="挂单偏移（bp，逗号分隔）"
    )
    parser.add_argument(
        "--horizons", default="1440,4320", help="持有根数（1m，逗号分隔）"
    )
    parser.add_argument(
        "--min-cross-section", type=int, default=10, help="截面宽度门槛，默认 10"
    )
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser, default="1m", help_text="日内周期，默认 1m")
    parser.add_argument(
        "--entry-fee-bp",
        type=float,
        default=MAKER_FEE_BP,
        help=f"进场费率，默认 {MAKER_FEE_BP:.0f}bp（maker）；给 10 就是进场也吃价的对照",
    )
    parser.add_argument(
        "--exit-fee-bp",
        type=float,
        default=TAKER_FEE_BP,
        help=f"出场费率，默认 {TAKER_FEE_BP:.0f}bp（taker，不建模出场成交）",
    )
    add_no_save(parser)
    return parser.parse_args()


def options_from(args: argparse.Namespace) -> SpreadOptions:
    return SpreadOptions(
        offsets_bp=tuple(float(x) for x in args.offsets.split(",")),
        horizons_bars=tuple(int(x) for x in args.horizons.split(",")),
        min_cross_section=args.min_cross_section,
        market=args.market,
        timeframe=args.timeframe,
        entry_fee_bp=args.entry_fee_bp,
        exit_fee_bp=args.exit_fee_bp,
    )


def print_summary(summary: pd.DataFrame, options: SpreadOptions) -> None:
    """主表：成交率、两腿联动、实际价差与「假设一定成交」的对照。

    均值与中位数并排：重尾分布下两者背离越大，说明结果越被单个名字主导。
    """
    fee_bp = options.entry_fee_bp + options.exit_fee_bp
    print(f"\n=== 两腿被动挂单（往返费率 {fee_bp:.0f}bp）===")
    print(
        f"{'持有':>6}{'偏移':>6}{'天数':>6}{'买腿成交':>9}{'卖腿成交':>9}"
        f"{'两腿都成':>9}{'不平衡':>8}{'价差均值':>10}{'价差中位':>10}{'胜率':>7}"
        f"{'全成交价差':>11}{'年化':>9}{'最差':>9}"
    )
    for _, row in summary.iterrows():
        hold = f"{int(row['horizon_bars']) / MINUTES_PER_DAY:.0f}天"
        print(
            f"{hold:>6}{row['offset_bp']:>6.1f}{int(row['days']):>6}"
            f"{row['long_fill']:>9.1%}{row['short_fill']:>9.1%}"
            f"{row['both_fill']:>9.1%}{row['imbalance']:>8.1%}"
            f"{row['spread_bp']:>10.2f}{row['median_bp']:>10.2f}"
            f"{row['win_rate']:>7.1%}{row['spread_all_bp']:>11.2f}"
            f"{row['annual_pct']:>8.1f}%{row['worst_bp']:>9.0f}"
        )
    print()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    options = options_from(args)
    legs = load_legs(options)
    if legs.empty:
        print("没有腿明细：先确认日内缓存与日线面板有交集标的")
        return 1
    daily = daily_frame(legs, options)
    summary = summary_frame(daily, options)
    days = pd.DatetimeIndex(cast(pd.Series, daily["day"]))
    symbols = cast(pd.Series, legs["day"]).size
    print(f"两腿被动挂单 | {options.market} · {options.timeframe} | 腿明细 {symbols:,}")
    print(
        f"时间 {cast(pd.Timestamp, days.min()).date()} → "
        f"{cast(pd.Timestamp, days.max()).date()} | "
        f"截面 ≥{options.min_cross_section} 只 | 交易日 {len(days)}"
    )
    print_summary(summary, options)
    if not args.no_save:
        print("落盘：")
        for path in save_outputs(summary, daily):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
