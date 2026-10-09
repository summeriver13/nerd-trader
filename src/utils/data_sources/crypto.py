# src/utils/data_sources/crypto.py
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import cast

import ccxt
import pandas as pd

from .base import BaseDataSource

logger = logging.getLogger(__name__)


class CCXTSource(BaseDataSource):
    # 连续失败上限：数据源持续不可用时快速失败，不要无限重试把调用方挂死
    MAX_CONSECUTIVE_FAILURES: int = 5
    # 单页 K 线数量（交易所分页上限）
    PAGE_LIMIT: int = 100
    # 起点定位的最少试探次数与总时间预算（越界请求既慢又不稳定）
    PROBE_STEPS: int = 8
    PROBE_BUDGET_SECONDS: float = 60.0

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    @property
    def label(self) -> str:
        return str(getattr(self.exchange, "id", "unknown_cex"))

    def _safe_fetch(
        self, symbol: str, timeframe: str, since_ts: int, limit: int
    ) -> list[list[float]]:
        """
        抓一页 K 线。

        起点越界时 OKX 对 limit>1 返回空页、对 limit=1 直接报参数错误，
        两种情形都是「该起点没有数据」，这里统一收敛成空列表。
        """
        try:
            page = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ts, limit=limit
            )
            return cast(list[list[float]], page)
        except Exception:
            return []

    def _probe_first_bar(
        self, symbol: str, timeframe: str, since_ts: int, end_ts: int
    ) -> int:
        """
        定位该标的真正的最早 K 线时间戳（毫秒）。

        请求起点早于上市时间时拿不到任何数据；先用几何步长往后试探，
        再在「无数据」与「有数据」之间二分细化，收敛到最早可得的一根。
        """
        bar_ms = int(ccxt.Exchange.parse_timeframe(timeframe)) * 1000
        low, high = since_ts, end_ts
        step = max((end_ts - since_ts) // 8, bar_ms)
        deadline = time.time() + self.PROBE_BUDGET_SECONDS
        for _ in range(self.PROBE_STEPS):
            if time.time() > deadline:
                return high
            probe = min(low + step, high)
            page = self._safe_fetch(symbol, timeframe, probe, self.PAGE_LIMIT)
            time.sleep(self.exchange.rateLimit / 1000)
            if page:
                high = max(int(page[0][0]), low)
                break
            low = probe
            if probe >= high:
                break
            step = max(step // 2, bar_ms)
        for _ in range(self.PROBE_STEPS):
            if high - low <= bar_ms or time.time() > deadline:
                break
            mid = (low + high) // 2
            page = self._safe_fetch(symbol, timeframe, mid, self.PAGE_LIMIT)
            time.sleep(self.exchange.rateLimit / 1000)
            if page:
                high = max(int(page[0][0]), low)
            else:
                low = mid
        if high >= end_ts:
            logger.warning(
                f"{symbol} 在请求区间内探不到任何 K 线：标的可能不存在或代码拼错"
            )
        return high

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if not self.exchange:
            return pd.DataFrame()

        since_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        bar_ms = int(ccxt.Exchange.parse_timeframe(timeframe)) * 1000
        all_candles: list[list[float]] = []  # 明确内部类型
        limit = self.PAGE_LIMIT

        print(f"⬇️ [CCXT] 获取 {symbol} ({timeframe}) ...")

        failures = 0
        probe_used = False
        while since_ts < end_ts:
            try:
                # CCXT fetch_ohlcv 返回类型通常为 list[list[float]]
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ts, limit=limit
                )
                failures = 0
            except Exception as e:
                failures += 1
                print(
                    f"   ❌ CCXT Error ({failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{symbol} 连续 {failures} 次抓取失败，请检查网络与代理"
                    ) from e
                time.sleep(3)
                continue

            if not ohlcv:
                if all_candles or probe_used:
                    # 已拿到数据（或起点已收敛过一次）：后面确实没有更多 K 线；
                    # 但没覆盖到请求末尾说明中途断了，必须报出来，否则会永远补不上这段空洞
                    if all_candles and int(all_candles[-1][0]) < end_ts - bar_ms:
                        logger.warning(
                            f"{symbol} 抓取在中途停止：{pd.Timestamp(int(all_candles[-1][0]), unit='ms', tz='UTC')}"
                            f" → {pd.Timestamp(end_ts, unit='ms', tz='UTC')} 出现空洞，"
                            f"已拿到的部分照常入库，下次运行会重试"
                        )
                    break
                probe_used = True
                probed = self._probe_first_bar(symbol, timeframe, since_ts, end_ts)
                if probed >= end_ts:
                    break
                print(
                    f"   ↩️ 起点早于最早 K 线，收敛到 "
                    f"{pd.Timestamp(probed, unit='ms', tz='UTC').date()}"
                )
                since_ts = max(probed, since_ts + 1)
                continue

            last_ts = int(ohlcv[-1][0])
            if last_ts <= since_ts:
                break

            all_candles.extend(ohlcv)  # type: ignore
            since_ts = last_ts + 1
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame()

        # --- 核心修复点：使用 pd.Index ---
        columns = pd.Index(["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles, columns=columns)

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # 转换并返回
        result = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return cast(pd.DataFrame, result)
