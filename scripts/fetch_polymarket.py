"""
Polymarket 前向采集：活跃 crypto 市场的快照 + 价格历史 + 成交明细。

历史回填不可行（容器 ADR-0007），样本只能靠每天抓一次攒出来，且**不能中断**——
价格历史只有约 30 天窗口，错过就永久缺失。

用法：
    uv run python scripts/fetch_polymarket.py --list-only            # 只看会抓哪些市场
    uv run python scripts/fetch_polymarket.py --max-markets 10       # 小样本端到端
    uv run python scripts/fetch_polymarket.py --max-markets 60 --workers 4   # 日常
    uv run python scripts/fetch_polymarket.py --from-snapshot        # 枚举抖动时重取明细

选池口径：`tag_slug=crypto` 的活跃市场里**成交额前 N**；价格历史默认 7 天窗口、逐小时粒度，
窗口重叠靠去重兜底；成交明细从最新一页往前翻，翻到「整页都已落盘」即停（增量水位）。
落盘见 `src/utils/polymarket_store.py`，验收见 `scripts/audit_polymarket.py`。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from src.config.paths import (
    POLYMARKET_EVENTS,
    POLYMARKET_MARKETS,
    POLYMARKET_PRICES,
    POLYMARKET_TRADES,
    polymarket_jsonl_path,
)
from src.utils.data_sources import polymarket as pm
from src.utils.polymarket_collect import CollectSpec, collect_market
from src.utils.polymarket_store import (
    KEY_FIELDS,
    append_jsonl,
    day_stamp,
    latest_market_snapshot,
    update_manifest,
)

DEFAULT_MAX_MARKETS = 40
DEFAULT_EVENT_PAGES = 6
DEFAULT_EVENT_PAGE_SIZE = 20
DEFAULT_WORKERS = 4
DEFAULT_TRADE_PAGES = 8
DEFAULT_TRADE_LIMIT = 500
SLUG_WIDTH = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket 前向采集")
    parser.add_argument("--max-markets", type=int, default=DEFAULT_MAX_MARKETS)
    parser.add_argument("--event-pages", type=int, default=DEFAULT_EVENT_PAGES)
    parser.add_argument("--event-page-size", type=int, default=DEFAULT_EVENT_PAGE_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--trade-pages", type=int, default=DEFAULT_TRADE_PAGES)
    parser.add_argument("--trade-limit", type=int, default=DEFAULT_TRADE_LIMIT)
    parser.add_argument("--price-window-days", type=int, default=pm.PRICE_WINDOW_DAYS)
    parser.add_argument("--fidelity", type=int, default=pm.DEFAULT_FIDELITY)
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-trades", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="只列市场，不取数")
    parser.add_argument(
        "--from-snapshot",
        action="store_true",
        help="用当天的市场快照当选池，不重新枚举（代理抖动时重取明细用）",
    )
    return parser.parse_args()


def event_snapshot(
    events: list[dict[str, Any]], *, fetched_at: str
) -> list[dict[str, Any]]:
    """事件快照：白名单裁剪 + 采集时刻（市场记录在选池时已带 fetched_at）。"""
    stamp = {"fetched_at": fetched_at}
    return [
        {**pm.pick(event, pm.EVENT_FIELDS), "tags": pm.tag_slugs(event), **stamp}
        for event in events
    ]


def print_market_list(markets: list[dict[str, Any]]) -> None:
    print(f"{'slug':<{SLUG_WIDTH}} {'成交额':>14} {'结算日':<12} {'neg':<4} 标签")
    for market in markets:
        print(
            f"{str(market.get('slug'))[:SLUG_WIDTH]:<{SLUG_WIDTH}} "
            f"{pm.volume_of(market):>14,.0f} "
            f"{str(market.get('endDate'))[:10]:<12} "
            f"{str(bool(market.get('negRisk'))):<4} "
            f"{','.join(market.get('tags') or [])[:60]}"
        )


def load_market_snapshot(day: str, *, fetched_at: str) -> list[dict[str, Any]]:
    """当天快照 → 采集用市场列表（按成交额降序，采集时刻刷成本轮）。"""
    markets = [
        {**record, "fetched_at": fetched_at} for record in latest_market_snapshot(day)
    ]
    return sorted(markets, key=pm.volume_of, reverse=True)


def main() -> int:
    args = parse_args()
    fetched_at = datetime.now(UTC).isoformat()
    day = day_stamp(datetime.now(UTC))
    events: list[dict[str, Any]] = []
    if args.from_snapshot:
        markets = load_market_snapshot(day, fetched_at=fetched_at)[: args.max_markets]
        print(f"从当天快照取市场 {len(markets)} 只（不枚举）")
    else:
        try:
            events = pm.fetch_crypto_events(
                pm.create_session(),
                pages=args.event_pages,
                page_size=args.event_page_size,
            )
        except RuntimeError as error:
            print(
                f"枚举失败（代理抖动？）：{error}\n可稍后重试，或改用 --from-snapshot 重取明细"
            )
            return 1
        markets = [
            {**market, "fetched_at": fetched_at}
            for market in pm.active_crypto_markets(events, limit=args.max_markets)
        ]
        print(
            f"枚举事件 {len(events)} 个 → 活跃 crypto 市场 {len(markets)} 只"
            f"（按成交额前 {args.max_markets}）"
        )
    if args.list_only:
        print_market_list(markets)
        return 0
    if not markets:
        print("没有可采集的市场：检查代理、枚举参数或 --from-snapshot 的日期")
        return 1

    spec = CollectSpec(
        price_window_days=args.price_window_days,
        fidelity=args.fidelity,
        trade_pages=args.trade_pages,
        trade_limit=args.trade_limit,
        skip_prices=args.skip_prices,
        skip_trades=args.skip_trades,
    )
    started = time.time()
    touched: dict[str, set[str]] = {name: set() for name in KEY_FIELDS}
    event_rows = event_snapshot(events, fetched_at=fetched_at)
    for dataset, rows in (
        (POLYMARKET_EVENTS, event_rows),
        (POLYMARKET_MARKETS, markets),
    ):
        append_jsonl(polymarket_jsonl_path(dataset, day), rows, KEY_FIELDS[dataset])
        touched[dataset].add(f"{day}.jsonl")

    print_market_list(markets)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(collect_market, market, spec): market for market in markets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            market = futures[future]
            try:
                result = future.result()
            except Exception as error:  # 单个市场失败不拖垮整轮
                print(
                    f"  [{index}/{len(markets)}] {market.get('slug')} 失败: {error!r}"
                )
                continue
            touched[POLYMARKET_PRICES].add(f"{result.slug}.jsonl")
            touched[POLYMARKET_TRADES].add(f"{result.slug}.jsonl")
            note = (" " + ";".join(result.notes)) if result.notes else ""
            print(
                f"  [{index}/{len(markets)}] {result.slug[:SLUG_WIDTH]:<{SLUG_WIDTH}} "
                f"价 +{result.prices_added}/-{result.prices_skipped} "
                f"成交 +{result.trades_added}/-{result.trades_skipped}{note}"
            )

    for dataset in KEY_FIELDS:
        path = update_manifest(dataset, touched=touched[dataset])
        print(f"manifest: {path}")
    print(
        f"完成，用时 {time.time() - started:.1f}s（raw 层只增不改，重复运行不产生重复行）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
