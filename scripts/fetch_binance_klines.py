"""
scripts/fetch_binance_klines.py
回填 Binance 现货 K 线，**保留成交笔数与 taker 买卖拆分**。

de Prado 第 2 章的 tick / imbalance / run bars 需要两个别处没有的字段：
`trades`（成交笔数，建 tick bars）与 `taker_buy_*`（主动买占比，建订单流失衡棒）。
Binance 的 K 线带这两类列，OKX 的 candle 接口只有 OHLCV；而仓库既有的批量抓取路径
（`CCXTSource`）只保留五列，所以这里直连 Binance REST，把 9 列原样落盘——列名与本地
已有的 binance 现货 1h/1m 缓存同构，面板与审计脚本可直接读。

用法：
    uv run python scripts/fetch_binance_klines.py --list-only
    uv run python scripts/fetch_binance_klines.py --timeframe 1h --days 3300 --top 30
    uv run python scripts/fetch_binance_klines.py --timeframe 1h --symbols BTCUSDT,ETHUSDT

`--top N` 的选池口径：OKX 永续日线**近一年中位日名义额**前列的标的，试着按 Binance
现货符号抓；Binance 没有该交易对的（HTTP 400）记为 SKIP，所以最终池子 = 候选里
Binance 实际提供的那些。增量：已有缓存从末根之后续抓，`--full` 强制全量重抓。

只读公开行情、不需要凭据；代理沿用 OKX 的配置（见 src/config/settings.py）。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests

from src.config.paths import crypto_dataset_dir
from src.utils.data_sources.binance import (
    binance_proxies,
    fetch_klines,
)

TARGET_EXCHANGE = "binance"
TARGET_MARKET = "spot"
SOURCE_EXCHANGE = "okx"
SOURCE_MARKET = "swap"
SOURCE_TIMEFRAME = "1d"
# 选池的流动性窗口（日历天）：只看近一年，避免「当年红过、现在没量」的标的混进来
LIQUIDITY_WINDOW_DAYS = 365
DEFAULT_TIMEFRAME = "1h"
DEFAULT_DAYS = 3300
DEFAULT_TOP = 30
SYMBOL_WIDTH = 14
MS_PER_DAY = 86_400_000
NS_PER_MS = 1_000_000


def select_candidates(limit: int) -> list[str]:
    """按 OKX 永续近一年中位日名义额取前 N，转成 Binance 现货符号（BTC-USDT → BTCUSDT）。"""
    source_dir = crypto_dataset_dir(SOURCE_EXCHANGE, SOURCE_MARKET, SOURCE_TIMEFRAME)
    cutoff = pd.Timestamp.now(tz="UTC") - np.timedelta64(LIQUIDITY_WINDOW_DAYS, "D")
    scores: list[tuple[float, str]] = []
    for path in sorted(source_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["close", "volume"])
        recent = frame[frame.index >= cutoff]
        if recent.empty:
            continue
        notional = float(cast(pd.Series, recent["close"] * recent["volume"]).median())
        scores.append((float(notional), f"{path.stem.replace('-', '')}"))
    scores.sort(reverse=True)
    return [symbol for _, symbol in scores[:limit]]


def _now_ms() -> int:
    """当前时刻的毫秒时间戳（走 stdlib，免得踩 pandas 时间戳的类型桩）。"""
    return int(datetime.now(UTC).timestamp() * 1000)


def _last_open_ms(path: Path) -> int | None:
    """缓存末根的毫秒时间戳；文件不存在或为空时返回 None。"""
    if not path.exists():
        return None
    index = pd.read_parquet(path).index
    if len(index) == 0:
        return None
    stamps = np.asarray(index, dtype="datetime64[ns]").astype("int64")
    return int(stamps[-1]) // NS_PER_MS


def fetch_symbol(
    session: requests.Session, symbol: str, target: Path, args: Any
) -> int:
    """抓一个标的并按需与既有缓存合并落盘，返回最终行数。"""
    last_ms = None if args.full else _last_open_ms(target)
    start_ms = _now_ms() - args.days * MS_PER_DAY if last_ms is None else last_ms + 1
    fresh = fetch_klines(session, symbol, interval=args.timeframe, start_ms=start_ms)
    if last_ms is None:
        merged = fresh
    else:
        merged = pd.concat([pd.read_parquet(target), fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(target)
    return len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="回填 Binance 现货 K 线（带笔数与 taker 分列）"
    )
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="K 线周期")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="首次回溯天数")
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="候选数（按 OKX 永续流动性）"
    )
    parser.add_argument("--symbols", default="", help="显式 Binance 符号，逗号分隔")
    parser.add_argument("--full", action="store_true", help="忽略缓存全量重抓")
    parser.add_argument("--list-only", action="store_true", help="只打印选池，不抓数")
    args = parser.parse_args()

    explicit = [
        part.strip().upper() for part in args.symbols.split(",") if part.strip()
    ]
    candidates = explicit or select_candidates(args.top)
    if not candidates:
        print("没有候选标的")
        return 1
    print(
        f"Binance {TARGET_MARKET} {args.timeframe} | 候选 {len(candidates)} 只 | "
        f"回溯 {args.days} 天 | {'全量' if args.full else '增量'}"
    )
    print("、".join(candidates))
    if args.list_only:
        return 0

    target_dir = crypto_dataset_dir(TARGET_EXCHANGE, TARGET_MARKET, args.timeframe)
    target_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    proxies = binance_proxies()
    if proxies is not None:
        session.proxies.update(proxies)

    started = time.time()
    written = skipped = failed = 0
    for index, symbol in enumerate(candidates, 1):
        target = target_dir / f"{symbol}.parquet"
        try:
            rows = fetch_symbol(session, symbol, target, args)
        except ValueError as error:
            skipped += 1
            print(f"[{index}/{len(candidates)}] {symbol} SKIP {error}", flush=True)
            continue
        except Exception as error:  # noqa: BLE001 - 单只失败不该杀掉整批
            failed += 1
            print(
                f"[{index}/{len(candidates)}] {symbol} 失败 {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        written += 1
        print(
            f"[{index}/{len(candidates)}] {symbol:<{SYMBOL_WIDTH}} {rows:>7,} 行 "
            f"{time.time() - started:>6.0f}s",
            flush=True,
        )

    print(f"\nALLDONE 写入 {written} / 跳过 {skipped} / 失败 {failed} → {target_dir}")
    print(f"耗时 {time.time() - started:.0f} 秒")
    if written:
        print(
            "下一步：uv run python scripts/build_crypto_manifests.py && "
            "uv run python scripts/audit_crypto_bars.py --timeframe "
            f"{args.timeframe} --exchange binance"
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
