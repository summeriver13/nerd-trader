"""
资金费率数据源：Binance 永续（历史最长），带本地 Parquet 缓存。

OKX 的资金费率接口只保留约 100 天，做 carry 的长历史研究必须换源；
Binance 公开接口可回溯到 2019-09，8 小时结算口径与 OKX 一致。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import ccxt
import pandas as pd

from src.config.paths import STORAGE_ROOT
from src.config.settings import GLOBAL_CONFIG

FUNDING_TIMEFRAME = "8h"
FUNDING_INTERVAL = pd.Timedelta(
    8, unit="h"
)  # 结算间隔由交易所决定，此处只作缓存新鲜度容差
PAGE_LIMIT = 1000
REQUEST_TIMEOUT_MS = 20000
RATE_LIMIT_SLEEP = 0.3
UTC_TZ = "UTC"


def create_funding_exchange() -> ccxt.Exchange:
    """构造 Binance 永续客户端（只用公共接口，代理沿用 OKX 配置）。"""
    config: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": REQUEST_TIMEOUT_MS,
        "options": {"defaultType": "future"},
    }
    if GLOBAL_CONFIG.use_proxy and GLOBAL_CONFIG.proxy_url:
        proxy = GLOBAL_CONFIG.proxy_url
        config["proxies"] = {"http": proxy, "https": proxy}
    # ccxt 的 ConstructorConfig 比文档里的 dict 严格，配置本身就是任意字典
    exchange = ccxt.binance(config)  # pyright: ignore[reportArgumentType]
    exchange.load_markets()
    return exchange


def daily_rates(rates: pd.Series) -> pd.Series:
    """把逐次结算的资金费率按日求和（结算间隔由交易所决定，不能假设恒为 8h）。

    缺日产出 NaN 而不是 0：把没结算的日子当成「费率为 0」会低估拖累。
    """
    return cast(pd.Series, rates.resample("1D").sum(min_count=1))


class FundingSource:
    """资金费率历史抓取 + 缓存（旁挂 meta.json 记录已请求过的起点）。

    exchange 不传则首次需要联网时才构造：缓存够新时整个研究可离线跑。
    """

    def __init__(
        self, exchange: ccxt.Exchange | None = None, data_root: Path | None = None
    ) -> None:
        self._exchange = exchange
        self.data_root = data_root if data_root is not None else STORAGE_ROOT

    @property
    def exchange(self) -> Any:
        """按需构造（load_markets 需要联网，不能让读缓存也依赖网络）。

        ccxt 基类把 fetch_* 声明成抛 NotImplementedError（返回 Never），
        静态分析会把调用之后的代码全判成不可达，故按 Any 处理。
        """
        if self._exchange is None:
            self._exchange = create_funding_exchange()
        return self._exchange

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "-").upper()
        return (
            self.data_root
            / "crypto"
            / "binance"
            / FUNDING_TIMEFRAME
            / f"{safe}.parquet"
        )

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return cast(pd.DataFrame, frame.sort_index())

    def _read_asked_start(self, symbol: str) -> pd.Timestamp | None:
        path = self._cache_path(symbol).with_suffix(".meta.json")
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        value = payload.get("asked_start")
        if not value:
            return None
        try:
            return cast(pd.Timestamp, pd.Timestamp(value))
        except ValueError:
            return None

    def _write_meta(self, symbol: str, asked_start: pd.Timestamp) -> None:
        path = self._cache_path(symbol).with_suffix(".meta.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"asked_start": asked_start.isoformat()}), encoding="utf-8"
        )

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """按 8 小时结算翻页抓取 [start, end] 的资金费率。"""
        unified = symbol if ":" in symbol else f"{symbol}:USDT"
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        while since < end_ms:
            page = self.exchange.fetch_funding_rate_history(unified, since, PAGE_LIMIT)
            if not page:
                break
            rows.extend(page)
            since = int(page[-1]["timestamp"]) + 1
            time.sleep(RATE_LIMIT_SLEEP)
        pairs = [
            (int(row["timestamp"]), float(row["fundingRate"]))
            for row in rows
            if row.get("fundingRate") is not None
        ]
        if not pairs:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {"rate": [rate for _, rate in pairs]},
            index=pd.to_datetime([ts for ts, _ in pairs], unit="ms", utc=True),
        )
        return cast(
            pd.DataFrame, frame[~frame.index.duplicated(keep="last")].sort_index()
        )

    def load(
        self, symbol: str, fetch_lookback_days: int, force_update: bool = False
    ) -> pd.DataFrame:
        """读取缓存并按需增量补齐，返回**整份缓存**（是否截断由调用方按研究窗口决定）。

        fetch_lookback_days 是抓数时往回探测的日历天数，不等于研究窗口；
        已请求过更早起点则不再重复探测。
        """
        path = self._cache_path(symbol)
        local = self._read_cache(path)
        target_start = cast(
            pd.Timestamp,
            pd.Timestamp.now(tz=UTC_TZ) - pd.Timedelta(fetch_lookback_days, unit="D"),
        )
        asked_start = self._read_asked_start(symbol)
        if not force_update and self._is_usable(local, target_start, asked_start):
            return local

        now = cast(pd.Timestamp, pd.Timestamp.now(tz=UTC_TZ))
        fetch_start = (
            target_start if local.empty else cast(pd.Timestamp, local.index[-1])
        )
        print(f"⬇️ [Binance] 获取 {symbol} 资金费率（{fetch_lookback_days} 天）...")
        fetched = self.fetch(symbol, fetch_start.to_pydatetime(), now.to_pydatetime())
        merged = pd.concat([local, fetched])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if merged.empty:
            return cast(pd.DataFrame, merged)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 增量补齐后整份重写：长历史下每次补抓都重写一次，暂可接受
        merged.to_parquet(path, compression="snappy")
        earlier = [value for value in (asked_start, target_start) if value is not None]
        self._write_meta(symbol, min(earlier))
        print(f"💾 资金费率缓存 {path}（{len(merged)} 条结算）")
        return cast(pd.DataFrame, merged)

    @staticmethod
    def _is_usable(
        local: pd.DataFrame,
        target_start: pd.Timestamp,
        asked_start: pd.Timestamp | None,
    ) -> bool:
        """缓存要同时满足「足够新」与「起点已覆盖或已探测过更早起点」。"""
        if local.empty:
            return False
        if pd.Timestamp.now(tz=UTC_TZ) - local.index[-1] >= FUNDING_INTERVAL:
            return False
        covered = local.index[0] <= target_start
        probed = asked_start is not None and asked_start <= target_start
        return bool(covered or probed)
