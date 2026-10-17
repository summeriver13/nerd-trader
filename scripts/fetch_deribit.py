"""抓 Deribit 期权链快照（薄壳：取数在 src/utils/data_sources/deribit.py，落盘在 src/utils/deribit_store.py）。

数据湖里这条线的用途有两个：一是给 Polymarket 阈值阶梯当**风险中性定价锚**，
二是将来买期权对冲时的**报价与深度来源**（对冲腿不在币安永续上，在 Deribit 微笑里）。

用法：

    uv run python scripts/fetch_deribit.py                 # BTC + ETH 各一轮快照
    uv run python scripts/fetch_deribit.py --currency BTC  # 只抓一个
    uv run python scripts/fetch_deribit.py --list-only     # 只看链的情况，不落盘

落盘：data/storage/deribit/raw/{options,futures}/<YYYY-MM-DD>.jsonl
manifest：data/manifests/deribit_raw_{options,futures}.json
验收：uv run python scripts/audit_deribit.py
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

from src.utils.data_sources.base import create_session
from src.utils.data_sources.deribit import DeribitChain, fetch_chain
from src.utils.deribit_store import (
    DERIBIT_FUTURES,
    DERIBIT_OPTIONS,
    SOURCE,
    append_chain,
    update_manifest,
)
from src.utils.jsonl_store import day_stamp

DEFAULT_CURRENCIES = ("BTC", "ETH")
DEFAULT_SLEEP_SECONDS = 2.0


def parse_currencies(raw: str) -> tuple[str, ...]:
    """币种列表，去重保序（重复抓同一币种既费请求又写重复行）。"""
    seen: list[str] = []
    for item in raw.replace(" ", "").split(","):
        code = item.upper()
        if code and code not in seen:
            seen.append(code)
    return tuple(seen)


def describe(chain: DeribitChain) -> str:
    """一行摘要：指数价、期权档数、到期月份数、最近到期——够判断链是否抓全。"""
    expiries = sorted({quote.expiry for quote in chain.options})
    nearest = expiries[0] if expiries else "-"
    return (
        f"{chain.currency} 指数 {chain.index_price:,.2f} | 期权 {len(chain.options):,} 档"
        f" | 到期 {len(expiries)} 个月（最近 {nearest}）| 期货 {len(chain.futures)} 张"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="抓 Deribit 期权链快照")
    parser.add_argument("--currency", default=",".join(DEFAULT_CURRENCIES))
    parser.add_argument("--list-only", action="store_true", help="只打印摘要，不落盘")
    parser.add_argument(
        "--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="币种间间隔"
    )
    args = parser.parse_args()

    currencies = parse_currencies(args.currency)
    session = create_session()
    started = datetime.now(tz=UTC)
    touched: dict[str, list[str]] = {DERIBIT_OPTIONS: [], DERIBIT_FUTURES: []}
    failures = 0

    for currency in currencies:
        try:
            chain = fetch_chain(session, currency)
        except Exception as error:  # 单个币种失败不该丢掉另一个币种的结果
            failures += 1
            print(f"[{currency}] 取链失败：{type(error).__name__}: {error}")
            continue
        print(f"[{currency}] {describe(chain)}")
        if args.list_only:
            continue
        # 分区日跟快照自己的时刻（UTC）走，不跟运行时刻：跑在 UTC 午夜前后时
        # 两者会分到不同的文件里（采集任务在 00:20 跑，正好踩这个窗口）。
        day = day_stamp(chain.fetched_at.to_pydatetime())
        counts = append_chain(chain, day=day)
        for dataset, (added, skipped) in counts.items():
            print(f"    {dataset}: 新增 {added:,} 行，跳过重复 {skipped:,} 行")
            touched[dataset].append(f"{day}.jsonl")
        time.sleep(args.sleep_seconds)

    if not args.list_only:
        for dataset, names in touched.items():
            if not names:
                continue
            path = update_manifest(dataset, touched=set(names))
            print(f"manifest 已更新：{path}")

    print(
        f"\n来源 {SOURCE}；运行时刻 {started.isoformat()}；验收：scripts/audit_deribit.py"
    )
    return 1 if failures == len(currencies) else 0


if __name__ == "__main__":
    raise SystemExit(main())
