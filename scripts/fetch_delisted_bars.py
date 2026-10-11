"""
scripts/fetch_delisted_bars.py
退市标的行情回填：把幸存者偏差补回来。

为什么需要这个脚本：OKX 对退市标的**完全不保留历史**——`instruments` 接口只列
live/preopen，`history-candles` 对退市标的返回 51001。实测本仓库缓存的 295 个现货
标的与 OKX 官方退市公告名单**零交集**，即「小市值溢价」这类截面结论天然建立在一个
删掉了全部尾部亏损的数据集上。

Binance 的现货 K 线接口对退市标的仍然可用（数据停在退市日，例：ACA 停在
2026-02-13、BAL 停在 2025-04-16），本脚本用它把 OKX 退市标的的日线补回本地。
这 15 个是「退市标的里还够格上 Binance 的」，因此由此估出的偏差是**保守**下界。

用法：
    uv run python scripts/fetch_delisted_bars.py                 # 全量回填
    uv run python scripts/fetch_delisted_bars.py --symbols KDA,ACA

只读公开行情、不需要凭据；代理沿用 OKX 的配置（见 src/config/settings.py）。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, cast

import pandas as pd
import requests

from src.config.paths import crypto_bar_path
from src.config.settings import load_config

# OKX 官方退市公告（help/section/announcements-delistings）中，Binance 仍保留历史的标的。
# 名单是 2026-10-11 的快照；再退市的标的需要重新抓公告页补进来。
DELISTED_SYMBOLS: tuple[str, ...] = (
    "RDNT",
    "KDA",
    "ALPHA",
    "BADGER",
    "MLN",
    "FORTH",
    "SC",
    "USTC",
    "LUNC",
    "ACA",
    "CLV",
    "BAL",
    "PERP",
    "FLM",
    "GHST",
)

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
# 单次最多 1000 根；全历史分页拉取
PAGE_LIMIT = 1000
COLUMNS = ("open", "high", "low", "close", "volume")
REQUEST_TIMEOUT = 30.0
# 请求间隔：Binance 现货权重限制宽松，留一点余量即可
REQUEST_PAUSE = 0.25
# 网络抖动重试（代理切节点会让 SSL 直接断开）
RETRY_ATTEMPTS = 4
RETRY_BACKOFF = 3.0


def _proxies() -> dict[str, str] | None:
    """沿用 OKX 的代理开关：Binance 同样走本机代理。"""
    config = load_config()
    if not (config.use_proxy and config.proxy_url):
        return None
    return {"http": config.proxy_url, "https": config.proxy_url}


def _get_klines(
    session: requests.Session, symbol: str, start_ms: int
) -> list[list[Any]]:
    """拉一页日线；网络错误按次数重试，业务错误直接抛。"""
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "limit": PAGE_LIMIT,
    }
    last_error: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = session.get(
                BINANCE_KLINES, params=params, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            # 400 = 标的不存在（已从 Binance 彻底下架），不该重试
            raise ValueError(f"HTTP {response.status_code}: {response.text[:120]}")
        except (requests.RequestException, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            last_error = error
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"{symbol} 连续 {RETRY_ATTEMPTS} 次请求失败") from last_error


def fetch_daily(session: requests.Session, base: str) -> pd.DataFrame:
    """拉取单个标的的全部日线，返回与缓存同构的 DataFrame。"""
    symbol = f"{base}USDT"
    rows: list[list[Any]] = []
    cursor = 0
    while True:
        page = _get_klines(session, symbol, cursor)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        # 下一页从最后一根之后开始（+1ms 避免重复取到同一根）
        cursor = int(page[-1][0]) + 1
        time.sleep(REQUEST_PAUSE)

    if not rows:
        raise ValueError(f"{symbol} 无任何 K 线")

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [int(row[0]) for row in rows], unit="ms", utc=True
            ),
            "open": [float(row[1]) for row in rows],
            "high": [float(row[2]) for row in rows],
            "low": [float(row[3]) for row in rows],
            "close": [float(row[4]) for row in rows],
            "volume": [float(row[5]) for row in rows],
        }
    )
    # 与 okx 缓存同构：DatetimeIndex 名为 datetime，五列全 float64
    return cast(pd.DataFrame, frame.set_index("datetime")[list(COLUMNS)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="回填 OKX 退市标的的日线（数据源 Binance）"
    )
    parser.add_argument("--symbols", default="", help="逗号分隔的币种主干，默认全量")
    parser.add_argument(
        "--pause", type=float, default=REQUEST_PAUSE, help="标的间隔秒数"
    )
    args = parser.parse_args()

    wanted = (
        tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
        if args.symbols
        else DELISTED_SYMBOLS
    )

    proxies = _proxies()
    session = requests.Session()
    if proxies:
        session.proxies.update(proxies)
    session.headers.update({"User-Agent": "nerd-trader/1.0"})

    ok, skipped = 0, 0
    for index, base in enumerate(wanted, start=1):
        target = crypto_bar_path("binance", f"{base}/USDT", "1d")
        try:
            frame = fetch_daily(session, base)
        except ValueError as error:
            print(f"[{index}/{len(wanted)}] {base:8s} SKIP  {error}", flush=True)
            skipped += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target)
        first, last = frame.index[0], frame.index[-1]
        print(
            f"[{index}/{len(wanted)}] {base:8s} OK    {len(frame):5d} 根  "
            f"{first:%Y-%m-%d} -> {last:%Y-%m-%d}  末价 {frame['close'].iloc[-1]:.6g}",
            flush=True,
        )
        ok += 1
        time.sleep(args.pause)

    print(
        f"\nALLDONE 写入 {ok} 个，跳过 {skipped} 个 -> {crypto_bar_path('binance', 'KDA/USDT', '1d').parent}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
