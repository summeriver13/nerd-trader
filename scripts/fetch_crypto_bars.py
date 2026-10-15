"""批量回填行情缓存：把整段历史拉到本地，供研究直接离线读。

研究路径（research.py）按需取数，单标的、只覆盖评估窗口；分钟级高换手研究需要
先把「多标的 × 全历史」整段拉到本地，否则每次研究都要重等网络。本脚本做这件事。

重复运行是安全的增量补数：已有缓存只补尾部，源侧给不出更早历史时不再重下
（`data_loader._probed_earlier` 的备忘），所以中断后直接重跑即可。

用法：
    uv run python scripts/fetch_crypto_bars.py --top 30                    # 永续 Top30 成交额，1m 全历史
    uv run python scripts/fetch_crypto_bars.py --symbols BTC/USDT:USDT
    uv run python scripts/fetch_crypto_bars.py --timeframe 1d --days 3200

后台跑（预计超过 10 分钟必须这样跑，前台超时被杀会留下半套缓存）：
    nohup uv run python scripts/fetch_crypto_bars.py --top 30 > data/results/logs/fetch_swap_1m.log 2>&1 &

只读公开行情、不需要凭据；非加密永续（代币化股票/商品）默认剔除。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.config.settings import create_data_exchange
from src.utils.data_loader import MarketDataHandler, TimeFrameStr
from src.utils.data_sources.crypto import CCXTSource

DEFAULT_TOP = 30
DEFAULT_TIMEFRAME = "1m"
# 覆盖 OKX 永续上线以来的全部历史（最早 2020-03-25），源侧给不出来自然收敛
DEFAULT_DAYS = 3300
# 分段回填的粒度（天）：每段成功即落盘，一次限速/抖动最多损失最后一段
DEFAULT_CHUNK_DAYS = 120
# 入口步骤（选标的）的网络重试：代理切节点会让 SSL 直接断开，一次抖动不该杀掉整个任务
SELECT_RETRY_ATTEMPTS = 8
SELECT_RETRY_BACKOFF = 5.0
# 连续多少个标的失败就判定数据源不可用并中止（避免整批空跑）
MAX_CONSECUTIVE_SYMBOL_FAILURES = 3
# OKX 合约分类：1=加密，3=美股，4=商品；只拉加密，避免混入代币化股票/商品
CRYPTO_CATEGORY = "1"
QUOTE = "USDT"
MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class Candidate:
    """候选标的：ccxt 统一符号 + 24h 成交额（USDT）。"""

    symbol: str
    notional_24h: float


def _to_float(value: Any) -> float:
    """OKX 对无成交的仪器会返回空串而不是数字，解析失败按 0 处理。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_top_swaps(exchange: Any, top: int) -> list[Candidate]:
    """按 24h 成交额取前 N 个加密永续（剔除非加密仪器）。"""
    exchange.load_markets()
    tickers = exchange.fetch_tickers(params={"instType": "SWAP"})
    rows: list[Candidate] = []
    for symbol, ticker in tickers.items():
        market = exchange.markets.get(symbol) or {}
        instrument = market.get("info") or {}
        quote = ticker.get("info") or {}
        if not (market.get("swap") and market.get("quote") == QUOTE):
            continue
        # instCategory 在合约定义里（1=加密/3=美股/4=商品）；成交额只在行情里，
        # 两边不是同一份 info，取错就会全员 0（排序退化成原始顺序）
        if instrument.get("instCategory") != CRYPTO_CATEGORY:
            continue
        last = _to_float(ticker.get("last"))
        size = _to_float(quote.get("volCcy24h"))
        rows.append(Candidate(symbol=symbol, notional_24h=size * last))
    rows.sort(key=lambda row: row.notional_24h, reverse=True)
    return rows[:top]


def format_span(frame: pd.DataFrame) -> str:
    """覆盖区间与行数，供逐标的进度行使用。"""
    if frame.empty:
        return "空"
    return (
        f"{len(frame):>9,} 根  "
        f"{frame.index[0]:%Y-%m-%d %H:%M} → {frame.index[-1]:%Y-%m-%d %H:%M}"
    )


def format_duration(seconds: float) -> str:
    """秒 → 人读的时长（进度行要一眼看出还要多久）。"""
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分"
    return f"{seconds / 3600:.1f} 时"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量回填 crypto 行情缓存")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="按成交额取前 N 个加密永续"
    )
    source.add_argument(
        "--symbols", help="显式标的，逗号分隔（ccxt 符号，永续带 :USDT 后缀）"
    )
    source.add_argument(
        "--refresh",
        help="按数据集目录里已有 parquet 的标的刷新，格式 exchange/market/timeframe"
        "（只补尾部，专给计划任务用；新增标的仍走 --top/--symbols）",
    )
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="K 线周期")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="回溯天数")
    parser.add_argument(
        "--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="分段回填粒度（天）"
    )
    return parser.parse_args()


def with_retry(func: Any, label: str) -> Any:
    """入口步骤的网络重试：代理切节点时 SSL 会直接断，退避重试到恢复为止。"""
    for attempt in range(1, SELECT_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except Exception as error:
            if attempt == SELECT_RETRY_ATTEMPTS:
                raise
            wait = SELECT_RETRY_BACKOFF * attempt
            print(
                f"{label} 第 {attempt} 次失败（{type(error).__name__}），{wait:.0f}s 后重试",
                flush=True,
            )
            time.sleep(wait)


def symbols_from_dataset(spec: str) -> list[str]:
    """把数据集目录里已有的 parquet 文件名反解成 ccxt 符号（BTC-USDT → BTC/USDT）。

    用途是计划任务的增量刷新：盘的标的集合由已有缓存决定，不重新按成交额选池，
    否则每天选池会让池子抖动、也会悄悄丢掉落榜的老标的。
    """
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(f"--refresh 需要 exchange/market/timeframe 三段，收到：{spec}")
    exchange, market, timeframe = parts
    directory = crypto_dataset_dir(exchange, market, timeframe)
    if not directory.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{directory}")
    symbols = []
    for path in sorted(directory.glob("*.parquet")):
        base, _, quote = path.stem.rpartition("-")
        if base:
            symbols.append(f"{base}/{quote}")
    print(f"从 {directory} 读到 {len(symbols)} 个已有标的，只补尾部")
    return symbols


def resolve_symbols(exchange: Any, args: argparse.Namespace) -> list[str]:
    if args.refresh:
        return symbols_from_dataset(args.refresh)
    if args.symbols:
        return [item.strip() for item in args.symbols.split(",") if item.strip()]
    candidates = with_retry(lambda: select_top_swaps(exchange, args.top), "选取标的")
    print(f"按 24h 成交额选出 {len(candidates)} 个加密永续：")
    for rank, candidate in enumerate(candidates, 1):
        print(
            f"  {rank:2d}. {candidate.symbol:20s} {candidate.notional_24h / 1e9:>7.2f}B"
        )
    return [candidate.symbol for candidate in candidates]


def backfill(
    handler: MarketDataHandler,
    symbol: str,
    timeframe: TimeFrameStr,
    *,
    days: int,
    chunk_days: int,
) -> pd.DataFrame:
    """从近到远分段回填：每段成功即落盘，抖动最多损失最后一段。

    一次性拉全历史时，一次限速/抖动就丢掉整段成果（实测 12.8 分钟白干）。分段后
    重跑从已落盘进度继续：data_loader 只补缺失的头部，不重拉已有区间。
    """
    stage = chunk_days
    while stage < days:
        handler.prepare_data(symbol, timeframe, days_back=stage)
        stage += chunk_days
    return handler.prepare_data(symbol, timeframe, days_back=days)


def fetch_all(
    handler: MarketDataHandler,
    symbols: list[str],
    timeframe: TimeFrameStr,
    *,
    days: int,
    chunk_days: int,
) -> list[tuple[str, str]]:
    """逐个回填，返回失败清单；单个标的失败不打断整批。"""
    failed: list[tuple[str, str]] = []
    durations: list[float] = []
    total = len(symbols)
    consecutive_failures = 0
    for index, symbol in enumerate(symbols, 1):
        started = time.time()
        try:
            frame = backfill(
                handler, symbol, timeframe, days=days, chunk_days=chunk_days
            )
        except Exception as error:  # 单标的失败要能继续，否则一个坏标的会拖住整批
            failed.append((symbol, str(error)))
            consecutive_failures += 1
            print(f"[{index}/{total}] FAIL {symbol}: {error}", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_SYMBOL_FAILURES:
                print(
                    f"连续 {consecutive_failures} 个标的失败，判定为数据源不可用，中止；"
                    f"已落盘的进度保留，恢复网络后重跑即可续上",
                    flush=True,
                )
                break
            continue
        consecutive_failures = 0
        elapsed = time.time() - started
        durations.append(elapsed)
        eta = statistics.median(durations) * (total - index)
        print(
            f"[{index}/{total}] OK   {symbol:20s} {format_span(frame)}  "
            f"{format_duration(elapsed)}  剩余约 {format_duration(eta)}",
            flush=True,
        )
    return failed


def main() -> int:
    args = parse_args()
    timeframe = cast(TimeFrameStr, args.timeframe)
    exchange = create_data_exchange("okx")
    handler = MarketDataHandler(CCXTSource(exchange))

    symbols = resolve_symbols(exchange, args)
    if not symbols:
        print("没有可拉取的标的")
        return 1

    print(
        f"计划：{len(symbols)} 个标的 × {timeframe} × {args.days} 天历史"
        f"（约 {args.days * MINUTES_PER_DAY:,} 根/标的）",
        flush=True,
    )
    started = time.time()
    failed = fetch_all(
        handler,
        symbols,
        timeframe,
        days=args.days,
        chunk_days=args.chunk_days,
    )

    print(
        f"\n完成：{len(symbols) - len(failed)}/{len(symbols)} 成功，"
        f"总耗时 {format_duration(time.time() - started)}",
        flush=True,
    )
    for symbol, message in failed:
        print(f"  FAIL {symbol}: {message}")
    print("下一步：uv run python scripts/build_crypto_manifests.py", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
