"""
src/utils/data_sources/binance.py
Binance 现货 K 线的 transport 层：分页、重试、代理与字段解析。

两个脚本原先各抄一份这段逻辑（`fetch_binance_klines.py` 取 9 列含笔数与 taker 分列、
`fetch_delisted_bars.py` 取 OHLCV 5 列），任何一处改动（限速、错误码语义）都会静默分叉，
所以请求与解析只在这里出一份，列子集由调用方自己挑。

`/api/v3/klines` 每页返回定长数组，位置固定（Binance 官方口径，不随周期变）：
`[0]` 开盘毫秒、`[1..4]` OHLC、`[5]` 成交量（base）、`[7]` 成交额（quote）、
`[8]` 成交笔数、`[9]` taker 买入量（base）、`[10]` taker 买入额（quote）。

错误语义：网络类错误（`requests.RequestException`）按 `RETRY_ATTEMPTS` 次重试；
HTTP 非 200 直接抛 `ValueError`——400 通常意味着标的已下架，重试只会浪费时间。
"""

from __future__ import annotations

import time
from typing import Any, cast

import pandas as pd
import requests

from src.config.settings import load_config

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
PAGE_LIMIT = 1000
REQUEST_TIMEOUT = 30.0
REQUEST_PAUSE = 0.25
RETRY_ATTEMPTS = 4
RETRY_BACKOFF = 3.0

# 缓存口径：索引名为 datetime（UTC）、数值列全 float64
COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
)


def binance_proxies() -> dict[str, str] | None:
    """沿用 OKX 的代理开关：Binance 同样走本机代理。"""
    config = load_config()
    if not (config.use_proxy and config.proxy_url):
        return None
    return {"http": config.proxy_url, "https": config.proxy_url}


def fetch_page(
    session: requests.Session, symbol: str, interval: str, start_ms: int
) -> list[list[Any]]:
    """拉一页原始 K 线；网络错误重试，业务错误（标的不存在）当场抛出。"""
    params = {
        "symbol": symbol,
        "interval": interval,
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
                return cast(list[list[Any]], response.json())
            raise ValueError(f"HTTP {response.status_code}: {response.text[:120]}")
        except (requests.RequestException, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            last_error = error
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"{symbol} 连续 {RETRY_ATTEMPTS} 次请求失败") from last_error


def fetch_klines(
    session: requests.Session,
    symbol: str,
    *,
    interval: str = "1d",
    start_ms: int = 0,
    pause: float = REQUEST_PAUSE,
) -> pd.DataFrame:
    """分页拉单标的 K 线，返回与本地缓存同构的 DataFrame（索引为开盘时刻，10 列全量）。

    `start_ms=0` 表示从头拉（退市标的只有短历史时用得到）；调用方按自己的列子集切片。
    """
    rows: list[list[Any]] = []
    cursor = start_ms
    while True:
        page = fetch_page(session, symbol, interval, cursor)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        # 下一页从最后一根之后开始（+1ms 避免重复取到同一根）
        cursor = int(page[-1][0]) + 1
        time.sleep(pause)
    if not rows:
        raise ValueError(f"{symbol} 该区间内无任何 K 线")
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
            "quote_volume": [float(row[7]) for row in rows],
            "trades": [float(row[8]) for row in rows],
            "taker_buy_base": [float(row[9]) for row in rows],
            "taker_buy_quote": [float(row[10]) for row in rows],
        }
    )
    frame = frame.drop_duplicates(subset="datetime").sort_values("datetime")
    return cast(pd.DataFrame, frame.set_index("datetime")[list(COLUMNS)])
