"""采集 OKX 强平事件流（薄壳：取数在 src/utils/data_sources/okx_liquidation.py，落盘在 src/utils/liquidation_store.py）。

为什么必须常驻轮询：OKX 对每个族只滚动保留**最近 100 条**强平记录（BTC 约 11 条/小时，
即约 9 小时），`after=now-24h` 返回 0 条——**没有历史回补入口**，不在存续期内采集就
永久丢失。所以本脚本按固定间隔轮询固定宇宙，按业务键去重，并检测「上一轮最新事件」
与「本轮最老事件」之间的空档（环形缓冲绕过即事件丢失）。

宇宙是**写死的 20 个 USDT 永续族**，不按成交额每日重选：池子抖动会让「同日横截面」
失去可比性，这是仓库里已经付过代价的教训（现货池固定 295 只、不随榜单变化）。

用法：

    uv run python scripts/fetch_liquidation.py                 # 跑 10 分钟（默认 60 秒一轮）
    uv run python scripts/fetch_liquidation.py --minutes 60    # 长驻一小时代替计划任务
    uv run python scripts/fetch_liquidation.py --once          # 只跑一轮，看读数
    uv run python scripts/fetch_liquidation.py --list-only     # 不落盘，只看各族最新事件

落盘：data/storage/crypto/okx/liquidation/<YYYY-MM-DD>.jsonl（按事件 UTC 日）
缺口：data/storage/crypto/okx/liquidation/gaps/<YYYY-MM-DD>.jsonl
manifest：data/manifests/crypto_liquidation.json
验收：uv run python scripts/audit_liquidation.py
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from src.utils.data_sources.base import create_session
from src.utils.data_sources.okx_liquidation import (
    ContractSpec,
    LiquidationEvent,
    fetch_contracts,
    fetch_events,
)
from src.utils.liquidation_store import (
    append_events,
    append_gaps,
    detect_gaps,
    load_last_seen,
    update_manifest,
)

# 固定宇宙：20 个成交额最高的加密 USDT 永续（2026-10-17 快照），刻意不含代币化股票
# 与贵金属族（SOXL/SNDK/XAU 等），它们与加密市场的强平机制无关。
# 2026-10-17 修正：TON 在 OKX 没有 USDT 永续（实测 instFamily 里根本没有 TON，
# 采集时该族恒报 51000 Parameter instFamily error），换成同为大盘主流的 UNI。
UNIVERSE: tuple[str, ...] = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "BNB-USDT",
    "ADA-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "SUI-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "DOT-USDT",
    "TRX-USDT",
    "UNI-USDT",
    "NEAR-USDT",
    "APT-USDT",
    "ARB-USDT",
    "OP-USDT",
    "PEPE-USDT",
)
DEFAULT_MINUTES = 10
DEFAULT_INTERVAL = 60
REQUEST_PAUSE = 0.12
TOP_EVENT_PREVIEW = 3


def now_iso() -> str:
    """本轮抓取时刻（UTC，毫秒精度）。"""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def poll_cycle(
    session: requests.Session, contracts: dict[str, ContractSpec]
) -> tuple[list[LiquidationEvent], int]:
    """跑一轮全宇宙；单个族失败只记失败数，不中断整批（网络抖动是常态）。"""
    events: list[LiquidationEvent] = []
    failures = 0
    for family in UNIVERSE:
        try:
            events.extend(fetch_events(session, family, contracts))
        except Exception as error:  # 单族异常不该丢掉其他 19 个族的事件
            failures += 1
            print(f"    ! {family} 抓取失败：{type(error).__name__}: {error}")
        time.sleep(REQUEST_PAUSE)
    return events, failures


def summarize(
    events: list[LiquidationEvent], added: int, skipped: int, gaps: int
) -> str:
    """一轮的一行摘要：事件数、新增/去重、缺口、最大单笔。"""
    if not events:
        return f"事件 0 | 新增 {added} 跳过 {skipped} | 缺口 {gaps}"
    heaviest = max(events, key=lambda event: event.notional)
    return (
        f"事件 {len(events):,} | 新增 {added:,} 跳过 {skipped:,} | 缺口 {gaps} | "
        f"最大 {heaviest.inst_id} {heaviest.pos_side} ${heaviest.notional:,.0f}"
    )


def run(minutes: int, interval: int, *, once: bool, list_only: bool) -> int:
    """轮询主循环：每 interval 秒采集一轮，跑满 minutes 分钟（--once 只跑一轮）。"""
    session = create_session()
    contracts = fetch_contracts(session)
    families = {spec.inst_family for spec in contracts.values()}
    missing = [family for family in UNIVERSE if family not in families]
    if missing:
        print(f"    ! 宇宙里有 {len(missing)} 个族在当前合约表里找不到：{missing}")
    print(f"    合约面值 {len(contracts):,} 个线性永续，宇宙 {len(UNIVERSE)} 个族")

    today = datetime.now(tz=UTC)
    days = [(today - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in (1, 0)]
    last_seen = load_last_seen(days)
    print(f"    已有标的 {len(last_seen):,} 个有历史，缺口阈值 {interval * 2} 秒")

    deadline = time.monotonic() + minutes * 60
    cycles = 0
    total_added = 0
    total_gaps = 0
    touched: set[str] = set()
    while True:
        started = time.monotonic()
        fetched_at = now_iso()
        events, failures = poll_cycle(session, contracts)
        gaps = detect_gaps(events, last_seen, slack_ms=interval * 2 * 1000)
        added = skipped = 0
        if not list_only:
            added, skipped, day_files = append_events(events, fetched_at)
            touched |= day_files
            if gaps:
                total_gaps += append_gaps(gaps)
        print(
            f"[{fetched_at}] {summarize(events, added, skipped, len(gaps))}"
            + (f" | 失败 {failures}" if failures else "")
        )
        for event in sorted(events, key=lambda item: item.notional, reverse=True)[
            :TOP_EVENT_PREVIEW
        ]:
            print(
                f"      {event.inst_id} {event.pos_side} {event.size_ccy:,.4f} @ {event.price:,.2f}"
                f" = ${event.notional:,.0f}"
            )
        for event in events:
            last_seen[event.inst_id] = max(last_seen.get(event.inst_id, 0), event.ts_ms)
        total_added += added
        cycles += 1
        if once or time.monotonic() >= deadline:
            break
        time.sleep(max(interval - (time.monotonic() - started), 0.0))

    if not list_only:
        update_manifest(touched)
        print(
            f"    完成 {cycles} 轮，新增 {total_added:,} 行，缺口 {total_gaps} 条，"
            f"触及 {len(touched)} 个日文件，manifest 已更新"
        )
    else:
        print(f"    完成 {cycles} 轮（--list-only 未落盘）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="采集 OKX 强平事件流")
    parser.add_argument(
        "--minutes", type=int, default=DEFAULT_MINUTES, help="总运行分钟数"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL, help="轮询间隔（秒）"
    )
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    parser.add_argument("--list-only", action="store_true", help="只看读数，不落盘")
    args: Any = parser.parse_args()
    return run(args.minutes, args.interval, once=args.once, list_only=args.list_only)


if __name__ == "__main__":
    raise SystemExit(main())
