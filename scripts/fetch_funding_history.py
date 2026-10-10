"""资金费率历史抓取：Binance 长历史 + OKX 近 103 天，落本地 Parquet 缓存。

Binance 公开接口可回溯到 2019-09（长历史唯一可用源）；OKX 只保留约 103 天，
但 carry 研究的执行腿在 OKX，所以「费率取 Binance」这一跨源代理假设必须用
同期 OKX 费率校验，两份都要抓。

重复运行是安全的增量补数：已有缓存只补尾部，起点已探测过更早则不再重下。

用法：
    uv run python scripts/fetch_funding_history.py                  # 两所 × Top30
    uv run python scripts/fetch_funding_history.py --exchanges okx
    uv run python scripts/fetch_funding_history.py --symbols BTC/USDT:USDT,ETH/USDT:USDT

只读公开接口、不需要凭据。Binance 上不存在的永续（小币）会记为 SKIP 而不是失败。
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.utils.data_sources.funding import FundingSource

DEFAULT_EXCHANGES = ("binance", "okx")
# 回溯天数按交易所给：Binance 长历史，OKX 接口只保留约 103 天
LOOKBACK_BY_EXCHANGE: dict[str, int] = {"binance": 3300, "okx": 120}
# 连续多少个标的失败就判定数据源不可用并中止（避免整批空跑）
MAX_CONSECUTIVE_FAILURES = 3
# Top30 永续：2026-10-10 按 OKX 24h 成交额选出，与 1m 回填同清单
TOP_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "ZEC/USDT:USDT",
    "SOL/USDT:USDT",
    "MAGIC/USDT:USDT",
    "STRK/USDT:USDT",
    "XRP/USDT:USDT",
    "NEAR/USDT:USDT",
    "DOGE/USDT:USDT",
    "PUMP/USDT:USDT",
    "WLD/USDT:USDT",
    "HYPE/USDT:USDT",
    "SUI/USDT:USDT",
    "FIL/USDT:USDT",
    "UNI/USDT:USDT",
    "PEPE/USDT:USDT",
    "ADA/USDT:USDT",
    "SAND/USDT:USDT",
    "CAP/USDT:USDT",
    "AAVE/USDT:USDT",
    "ONDO/USDT:USDT",
    "OP/USDT:USDT",
    "ENA/USDT:USDT",
    "DOT/USDT:USDT",
    "BAT/USDT:USDT",
    "TRUMP/USDT:USDT",
    "MET/USDT:USDT",
    "LIT/USDT:USDT",
    "XDP/USDT:USDT",
    "BNB/USDT:USDT",
)


@dataclass(frozen=True)
class FetchResult:
    """一只标的的抓取结果。"""

    symbol: str
    settlements: int
    start: pd.Timestamp
    end: pd.Timestamp
    elapsed: float


def fetch_one(source: FundingSource, symbol: str, days: int) -> FetchResult | None:
    """抓一只标的的资金费率，空结果返回 None。"""
    started = time.monotonic()
    frame = source.load(symbol, days)
    elapsed = time.monotonic() - started
    if frame.empty:
        return None
    index = frame.index
    return FetchResult(
        symbol=symbol,
        settlements=len(frame),
        start=cast(pd.Timestamp, index[0]),
        end=cast(pd.Timestamp, index[-1]),
        elapsed=elapsed,
    )


def run_exchange(exchange_id: str, symbols: list[str], days: int) -> bool:
    """抓一个交易所的全部标的，返回是否全部成功。"""
    source = FundingSource(exchange_id=exchange_id)
    print(
        f"\n[{exchange_id}] 计划：{len(symbols)} 个标的，回溯 {days} 天",
        flush=True,
    )
    total = 0
    failed: list[str] = []
    consecutive = 0
    for position, symbol in enumerate(symbols, start=1):
        tag = f"[{position:>2}/{len(symbols)}]"
        try:
            result = fetch_one(source, symbol, days)
        except Exception as error:  # noqa: BLE001 - 单只失败不该杀掉整批
            message = str(error)[:70]
            print(
                f"{tag} SKIP {symbol:<20} {type(error).__name__}: {message}", flush=True
            )
            failed.append(symbol)
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[{exchange_id}] 连续 {consecutive} 只失败，判定数据源不可用，中止",
                    flush=True,
                )
                traceback.print_exc()
                return False
            continue
        consecutive = 0
        if result is None:
            print(f"{tag} SKIP {symbol:<20} 无数据", flush=True)
            failed.append(symbol)
            continue
        total += result.settlements
        print(
            f"{tag} OK   {result.symbol:<20}{result.settlements:>7} 条  "
            f"{result.start:%Y-%m-%d} -> {result.end:%Y-%m-%d}  {result.elapsed:>5.1f}s",
            flush=True,
        )
    print(
        f"[{exchange_id}] 完成：成功 {len(symbols) - len(failed)}/{len(symbols)} 标的，"
        f"共 {total:,} 条结算",
        flush=True,
    )
    if failed:
        print(f"[{exchange_id}] 未取到：{', '.join(failed)}", flush=True)
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="资金费率历史抓取（Binance 长历史 + OKX 近期）"
    )
    parser.add_argument(
        "--exchanges",
        default=",".join(DEFAULT_EXCHANGES),
        help=f"逗号分隔，默认 {','.join(DEFAULT_EXCHANGES)}",
    )
    parser.add_argument(
        "--symbols", default=None, help=f"逗号分隔，默认 Top{len(TOP_SYMBOLS)} 永续"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="回溯天数，默认按交易所给（Binance 3300、OKX 120）",
    )
    args = parser.parse_args()

    symbols = (
        [item.strip() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(TOP_SYMBOLS)
    )
    ok = True
    for exchange_id in [
        item.strip() for item in args.exchanges.split(",") if item.strip()
    ]:
        days = (
            args.days
            if args.days is not None
            else LOOKBACK_BY_EXCHANGE.get(exchange_id, 120)
        )
        if not run_exchange(exchange_id, symbols, days):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
