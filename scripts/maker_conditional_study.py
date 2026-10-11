"""
scripts/maker_conditional_study.py
条件化挂单研究：每天在反转选中的币上挂买单，量成交率与逆向选择。

用法：
    uv run python scripts/maker_conditional_study.py
    uv run python scripts/maker_conditional_study.py --offsets 0.5,2,5 --horizons 1440
    uv run python scripts/maker_conditional_study.py --no-save

只读本地 1m 缓存与日线面板，不触发采集。全量跑一次约 4 分钟（30 标的 × 6100 万根）。
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import pandas as pd

from src.research.maker_conditional import (
    GROUP_LOSER,
    MAKER_FEE_BP,
    TAKER_FEE_BP,
    MakerOptions,
    aggregate,
    annual_frame,
    load_events,
    save_outputs,
)

COLUMN_WIDTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="条件化挂单（反转信号）研究")
    parser.add_argument(
        "--offsets", default="0.5,1,2,5,10", help="挂单偏移（bp，逗号分隔）"
    )
    parser.add_argument(
        "--horizons", default="1440,4320", help="持有根数（1m，逗号分隔）"
    )
    parser.add_argument(
        "--min-cross-section", type=int, default=10, help="截面宽度门槛，默认 10"
    )
    parser.add_argument("--market", default="swap", help="市场层，默认 swap")
    parser.add_argument("--timeframe", default="1m", help="日内周期，默认 1m")
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def options_from(args: argparse.Namespace) -> MakerOptions:
    return MakerOptions(
        offsets_bp=tuple(float(x) for x in args.offsets.split(",")),
        horizons_bars=tuple(int(x) for x in args.horizons.split(",")),
        min_cross_section=args.min_cross_section,
        market=args.market,
        timeframe=args.timeframe,
    )


def print_cells(frame: pd.DataFrame, options: MakerOptions) -> None:
    """主表：成交率、三种收益、逆向选择与每笔优势。"""
    print(
        f"\n=== 条件化挂单（费率：maker {MAKER_FEE_BP:.0f}bp / "
        f"taker {TAKER_FEE_BP:.0f}bp）==="
    )
    print(
        f"{'持有':>6}{'组':<6}{'偏移':>6}{'事件':>7}{'成交率':>8}{'挂单收益':>10}"
        f"{'市价入场':>10}{'逆向选择':>10}{'每笔优势':>10}{'期望/次':>9}"
    )
    for horizon in options.horizons_bars:
        block = frame[frame["horizon_bars"] == horizon]
        label = f"{horizon / 1440:.0f}天"
        for _, row in block.iterrows():
            print(
                f"{label:>6}{row['group']:<6}{row['offset_bp']:>6.1f}"
                f"{int(row['events']):>7}{row['fill_rate']:>8.1%}"
                f"{row['maker_bp']:>10.2f}{row['base_bp']:>10.2f}"
                f"{row['adverse_bp']:>10.2f}{row['per_trade_bp']:>10.2f}"
                f"{row['expected_bp']:>9.2f}"
            )
        print()


def print_annual(annual: pd.DataFrame, options: MakerOptions) -> None:
    """年度分解：结论随市场状态翻转，只看全样本会被单一 regime 主导。"""
    block = cast(pd.DataFrame, annual[annual["group"] == GROUP_LOSER])
    block = cast(pd.DataFrame, block[block["fill_rate"] > 0])
    if block.empty:
        return
    print(f"=== 年度分解（{GROUP_LOSER}组，跨全部偏移与持有）===")
    print(
        f"{'年':>5}{'事件':>7}{'成交率':>8}{'挂单收益':>10}{'市价入场':>10}"
        f"{'逆向选择':>10}{'每笔优势':>10}"
    )
    for _, row in block.iterrows():
        print(
            f"{int(row['year']):>5}{int(row['events']):>7}{row['fill_rate']:>8.1%}"
            f"{row['maker_bp']:>10.2f}{row['base_bp']:>10.2f}"
            f"{row['adverse_bp']:>10.2f}{row['per_trade_bp']:>10.2f}"
        )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    options = options_from(args)
    events = load_events(options)
    if events.empty:
        print("没有事件：先确认日内缓存与日线面板有交集标的")
        return 1
    cells = aggregate(events, options)
    days = pd.DatetimeIndex(cast(pd.Series, events["day"]))
    symbols = cast(pd.Series, events["symbol"]).nunique()
    print(f"条件化挂单研究 | {options.market} · {options.timeframe} | 标的 {symbols}")
    print(
        f"时间 {cast(pd.Timestamp, days.min()).date()} → "
        f"{cast(pd.Timestamp, days.max()).date()} | "
        f"截面 ≥{options.min_cross_section} 只 | 事件 {events.shape[0]:,}"
    )
    print_cells(cells, options)
    print_annual(annual_frame(events), options)
    if not args.no_save:
        print("\n落盘：")
        for path in save_outputs(cells):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
