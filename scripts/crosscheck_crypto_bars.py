"""数据交叉校验：缓存里的 K 线与数据源、第二来源逐根比对。

信任链第一环（数据正确）现在只有 manifest + hash，缺的是「与实际数据源对账」。
本脚本做两件事，任何研究结论发布前都该先跑一遍：

1. 同源复抓：从缓存取最近 N 根，按同一区间重新抓一次，要求 OHLCV 逐根完全相等
   —— 验证缓存/合并/去重/清洗这条链路没有改数。
2. 跨源对照：同一区间从第二家交易所取价，比较收盘价差异。
   两家交易所存在基差，不可能逐根相等，故只报差异分布并给一个经验阈值。

用法：
    uv run python scripts/crosscheck_crypto_bars.py [--symbol BTC/USDT] [--timeframe 1d] [--bars 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import ccxt
import numpy as np
import pandas as pd

from src.config.paths import PROJECT_ROOT, STORAGE_ROOT
from src.config.settings import GLOBAL_CONFIG
from src.utils.data_sources.crypto import CCXTSource
from src.utils.data_sources.funding import create_funding_exchange

OKX_LABEL = "okx"
BINANCE_LABEL = "binance"
COLUMNS = ["open", "high", "low", "close", "volume"]
# 跨源阈值：现货同标的两家交易所日线收盘差异中位应远小于此值，超出说明有一侧数据不对
CROSS_SOURCE_MEDIAN_BP = 20.0
CROSS_SOURCE_P95_BP = 100.0


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-").upper()
    return STORAGE_ROOT / "crypto" / OKX_LABEL / timeframe / f"{safe}.parquet"


def _load_cached(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        raise SystemExit(f"❌ 无缓存：{path}（先跑一次 research.py 生成）")
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    # 未收盘的最后一根不入比对：它的成交量还在变，复抓必然与缓存不同
    bar_seconds = int(ccxt.Exchange.parse_timeframe(timeframe))
    now = pd.Timestamp.now(tz="UTC")
    closed = frame[frame.index + pd.Timedelta(bar_seconds, unit="s") <= now]
    return closed.iloc[-bars:]


def check_same_source(cached: pd.DataFrame, symbol: str, timeframe: str) -> bool | None:
    """同源复抓：逐根比对 OHLCV，任何一格不同都要报出来。"""
    exchange: Any = ccxt.okx(
        GLOBAL_CONFIG.to_ccxt_config()  # pyright: ignore[reportArgumentType]
    )
    try:
        source = CCXTSource(exchange)
        start = cast(pd.Timestamp, cached.index[0]).to_pydatetime()
        end = cast(pd.Timestamp, cached.index[-1]).to_pydatetime()
        fetched = source.fetch_bars(
            symbol,
            timeframe,  # type: ignore[arg-type]
            start,
            end,
        )
    except Exception as e:
        # 「校验未完成」与「校验未通过」是两件事，必须分开报
        print(f"[同源] 未完成：数据源不可达（{type(e).__name__}: {str(e)[:80]}）")
        return None
    finally:
        exchange.close()

    fetched = fetched.reindex(cached.index)
    missing = int(fetched["close"].isna().sum())
    diff = (fetched[COLUMNS] - cached[COLUMNS]).abs()
    mismatched = int((diff.max(axis=1) > 0).sum())
    print(
        f"[同源] 复抓 {len(cached)} 根：缺失 {missing} 根、逐根不一致 {mismatched} 根"
    )
    if mismatched:
        worst = diff.max(axis=1).sort_values(ascending=False).head(5)
        for stamp, value in worst.items():
            print(f"   ⚠️ {stamp}: 最大偏差 {value:.6g}")
    return missing == 0 and mismatched == 0


def check_cross_source(cached: pd.DataFrame, symbol: str) -> bool | None:
    """跨源对照：第二家交易所的同区间收盘价差异（基差 → 只比分布，不比相等）。"""
    exchange: Any = None
    try:
        exchange = create_funding_exchange()  # Binance 公共接口，代理沿用同一配置
        since = int(cast(pd.Timestamp, cached.index[0]).timestamp() * 1000)
        page = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=len(cached) + 5)
    except Exception as e:
        print(f"[跨源] 未完成：第二来源不可达（{type(e).__name__}: {str(e)[:80]}）")
        return None
    finally:
        if exchange is not None:
            exchange.close()

    if not page:
        print("[跨源] Binance 无数据，跳过")
        return False
    other = pd.DataFrame(
        {"close": [float(row[4]) for row in page]},
        index=pd.to_datetime([int(row[0]) for row in page], unit="ms", utc=True),
    )
    aligned = pd.DataFrame({"okx": cached["close"], "binance": other["close"]}).dropna()
    if aligned.empty:
        print("[跨源] 两源无重叠区间，跳过")
        return False

    bp = ((aligned["okx"] / aligned["binance"] - 1).abs() * 10_000).sort_values()
    median = float(bp.median())
    p95 = float(np.percentile(bp, 95))
    worst_stamp = bp.index[-1]
    print(
        f"[跨源] {len(aligned)} 根重叠：|收盘差| 中位 {median:.1f}bp、P95 {p95:.1f}bp、"
        f"最大 {bp.iloc[-1]:.1f}bp（{worst_stamp:%Y-%m-%d}）"
    )
    ok = median <= CROSS_SOURCE_MEDIAN_BP and p95 <= CROSS_SOURCE_P95_BP
    if not ok:
        print(
            f"   ⚠️ 超过阈值（中位 ≤{CROSS_SOURCE_MEDIAN_BP:.0f}bp、"
            f"P95 ≤{CROSS_SOURCE_P95_BP:.0f}bp）：需人工确认哪一侧数据有问题"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="K 线缓存的数据源交叉校验")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--bars", type=int, default=100)
    args = parser.parse_args()

    cached = _load_cached(args.symbol, args.timeframe, args.bars)
    print(
        f"校验 {args.symbol} {args.timeframe}：缓存 {len(cached)} 根 "
        f"({cached.index[0]:%Y-%m-%d} → {cached.index[-1]:%Y-%m-%d})"
    )
    same = check_same_source(cached, args.symbol, args.timeframe)
    cross = check_cross_source(cached, args.symbol)
    verdict = {True: "通过", False: "未通过", None: "未完成（数据源不可达）"}
    print(f"\n结论：同源复抓 {verdict[same]}；跨源对照 {verdict[cross]}")
    print(
        f"（仓库根目录 {PROJECT_ROOT.name}，缓存 {_cache_path(args.symbol, args.timeframe).name}）"
    )
    return 0 if same and cross else 1


if __name__ == "__main__":
    sys.exit(main())
