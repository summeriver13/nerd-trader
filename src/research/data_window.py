"""
src/research/data_window.py
取数窗口：把「评估根数 + 预热根数」换算成抓数深度，并把缓存裁到恰好这个长度。

单独成模块的原因：根数与日历天是两种单位，混用会让 1h 的「730」只回测 30 天
（tests/research/test_time_units.py 记着这条历史教训），换算与裁剪必须只有一个出处。
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

import ccxt
import pandas as pd

from src.config.settings import create_exchange
from src.utils.data_loader import MarketDataHandler, TimeFrameStr
from src.utils.data_sources.crypto import CCXTSource
from src.utils.data_sources.funding import FundingSource, daily_rates

_SECONDS_PER_DAY = 86_400
FUNDING_LOOKBACK_BUFFER = 7  # 抓数时多往回探几天，免得缓存边界落在窗口内


class DataLoader(Protocol):
    """取数只需这一个方法：真实实现是 MarketDataHandler，测试可注入假加载器"""

    def prepare_data(
        self, symbol: str, timeframe: TimeFrameStr, days_back: int
    ) -> pd.DataFrame: ...


def bars_to_days(bars: int, timeframe: TimeFrameStr) -> int:
    """把 K 线根数换算成日历天，供抓数深度使用（上取整，至少 1 天）"""
    bar_seconds = int(ccxt.Exchange.parse_timeframe(timeframe))
    return max(1, math.ceil(bars * bar_seconds / _SECONDS_PER_DAY))


def load_window(
    loader: DataLoader, symbol: str, timeframe: TimeFrameStr, bars: int, warmup: int
) -> pd.DataFrame | None:
    """取「评估 bars 根 + 预热 warmup 根」的数据。

    多余的历史裁掉：指标在窗口起点的取值只取决于预热根数，
    不裁的话结果会随本地缓存变深而漂移，同一个命令不再可复现。
    数据为空返回 None。
    """
    wanted = bars + warmup
    df = loader.prepare_data(
        symbol, timeframe, days_back=bars_to_days(wanted, timeframe)
    )
    if df.empty:
        return None
    if len(df) < wanted:
        print(
            f"⚠️ {symbol} 仅 {len(df)} 根历史"
            f"（不足 {wanted} 根 = 评估 {bars} + 预热 {warmup}），窗口前段指标未热透。"
        )
    return cast(pd.DataFrame, df.iloc[-wanted:])


def to_funding_symbol(symbol: str) -> str | None:
    """行情代码换 Binance 永续代码：BTC/USDT 与 BTC-USDT 都 → BTC/USDT；非 USDT 返回 None。"""
    base, _, quote = symbol.replace("-", "/").partition("/")
    return f"{base}/USDT" if quote.upper() == "USDT" else None


def load_funding_rates(
    symbol: str, index: pd.DatetimeIndex, source: FundingSource | None = None
) -> pd.Series | None:
    """取该标的的按日资金费率（正 = 多头付钱）；无数据返回 None。

    永续多头每 8 小时结算一次，正费率是长期持多的系统性成本（BTC 平均 12%/年），
    向量化回测不计它就会把多头策略整体高估。缺数据宁可不计，但必须打印出来，
    免得读者以为表格里的数字已经含了 funding。
    """
    funding_symbol = to_funding_symbol(symbol)
    if funding_symbol is None or index.empty:
        return None
    days = int((index[-1] - index[0]).days) + FUNDING_LOOKBACK_BUFFER
    try:
        raw = (source or FundingSource()).load(funding_symbol, days)
    except Exception:
        traceback.print_exc()
        print(f"⚠️ {symbol} 资金费率获取失败，本次回测未计入 funding")
        return None
    if raw.empty:
        print(f"⚠️ {symbol} 无资金费率数据，本次回测未计入 funding")
        return None

    daily = daily_rates(cast(pd.Series, raw["rate"])).dropna()
    # pandas-stubs 未声明 DatetimeIndex.normalize，走 .dt 访问器（按自然日归组）
    wanted = pd.DatetimeIndex(pd.DatetimeIndex(index).to_series().dt.normalize())
    covered = wanted[(wanted >= daily.index[0]) & (wanted <= daily.index[-1])]
    if len(covered) < len(wanted.unique()):
        print(
            f"⚠️ 资金费率只覆盖 {daily.index[0]:%Y-%m-%d} → {daily.index[-1]:%Y-%m-%d}，"
            f"窗口内 {len(wanted.unique()) - len(covered.unique())} 天按 0 计"
        )
    print(
        f"💸 已计入资金费率: {funding_symbol} {daily.index[0]:%Y-%m-%d} → "
        f"{daily.index[-1]:%Y-%m-%d}（{len(daily)} 天按日汇总）"
    )
    return daily


@contextmanager
def open_loader() -> Iterator[MarketDataHandler]:
    """每次运行只开一个交易所会话，结束后关闭（不关会随多标的批量累积）。"""
    exchange = create_exchange("okx")
    try:
        yield MarketDataHandler(CCXTSource(exchange))
    finally:
        exchange.close()


def fetch_window(
    symbol: str, timeframe: TimeFrameStr, bars: int, warmup: int
) -> pd.DataFrame | None:
    """按评估根数与预热根数取数；失败打 traceback，不把代码错误伪装成网络问题。"""
    try:
        with open_loader() as loader:
            return load_window(loader, symbol, timeframe, bars, warmup)
    except Exception:
        traceback.print_exc()
        print("❌ 数据获取失败（见上方 traceback）")
        return None
