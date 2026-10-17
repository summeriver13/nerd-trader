"""上新币效应统一测试台：上市事件对齐、生命周期、入场时点与做空尾部风险。

用法：
    uv run python scripts/listing_study.py                # 全量并落盘
    uv run python scripts/listing_study.py --no-save      # 只打印
    uv run python scripts/listing_study.py --horizon 180  # 换入场扫描的持有期

数据来源：OKX 现货上市时刻（`public/instruments`，每次实时拉取，约 2 秒）+ 本地日线缓存。
可信样本从 2022-01-01 起（更早的 listTime 被 OKX 批量重置），详见 src/research/listings/listing.py。
"""

import argparse

from src.config.settings import create_data_exchange
from src.research.listings.listing import fetch_listing_times, load_events
from src.research.listings.listing_report import (
    DEFAULT_HORIZON,
    build_report,
    print_report,
    save_outputs,
)
from src.utils.study_cli import add_no_save


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="上新币效应统一测试台")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"入场扫描与年份分组的持有期（根），默认 {DEFAULT_HORIZON}",
    )
    add_no_save(parser)
    return parser.parse_args()


def main() -> int:
    """跑完整流程：拉上市时刻 → 配对缓存 → 六张表 → 打印（可选落盘）。"""
    args = parse_args()
    exchange = create_data_exchange("okx")
    listing_times = fetch_listing_times(exchange)
    events = load_events(listing_times)

    print("=== 面板口径 ===")
    print(f"OKX 现货 live 标的 {len(listing_times)} 只 · 可信上市事件 {len(events)} 个")
    if events:
        first = min(event.list_time for event in events)
        last = max(event.list_time for event in events)
        print(f"上市时刻区间 {first.date()} → {last.date()}")

    tables = build_report(events, horizon=args.horizon)
    print_report(tables, horizon=args.horizon)

    if not args.no_save:
        written = save_outputs(tables)
        print(f"\n已落盘 {len(written)} 份 parquet → {written[0].rsplit('/', 1)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
