"""做空新币的组合回测：把事件研究的每笔损益变成可累积的组合曲线。

## 为什么必须单独回测

事件研究给出的是「每笔做空的中位损益」，回答不了「能不能累积」：

- 上市事件在时间上重叠，资金要同时分配给多个头寸；
- 每日等权再平衡会引入波动拖累（年化波动 100%+ 时拖累可达几十个百分点）；
- 换手成本按每日进出的仓位收，不是按天收。

## 三个必须做对的地方

1. **成本按换手**：每日成本 = 单边费率 × 当日进出仓位占组合的比例。误按每日固定
   0.2% 扣，会把 +85% 的累计收益算成 −93%（实测踩过这个坑）。
2. **入场价用收盘价**：首根 bar 的 open 是参考价不是成交价（见 listing.py）。
3. **不建模强平**：日频再平衡的空头在标的翻倍时亏损会超过本金，真实账户早已被强平，
   故这里的亏损是上界——回撤数字不能直接当爆仓线读。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import CRYPTO_ROOT
from src.research.listings.listing import ListingEvent

# 默认单边费率：与项目其余研究一致取永续 taker
DEFAULT_FEE_ONE_WAY = 0.001
ANNUAL_DAYS = 365.25


@dataclass(frozen=True)
class ShortBacktest:
    """做空组合回测结果。"""

    entry: int
    hold: int
    fee_one_way: float
    days: int
    avg_positions: float
    total_return: float
    annual_compounded: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    positive_days: float


def build_price_panel(
    events: list[ListingEvent],
    *,
    root: Path = CRYPTO_ROOT,
    market: str = "spot",
    timeframe: str = "1d",
    exchange: str = "okx",
) -> pd.DataFrame:
    """把事件列表的收盘价拼成一张对齐面板（行=日期，列=标的）。"""
    series: dict[str, pd.Series] = {}
    for event in events:
        path = root / exchange / market / timeframe / f"{event.symbol}.parquet"
        if not path.exists():
            continue
        close = pd.read_parquet(path, columns=["close"])["close"]
        series[event.symbol] = cast(pd.Series, close)
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def holding_windows(
    events: list[ListingEvent], index: pd.DatetimeIndex, *, entry: int, hold: int
) -> dict[str, set[int]]:
    """每个标的的持仓位置集合：上市后第 entry+1 根建仓，持有 hold 根。

    定位用 `searchsorted(..., side="right") - 1`：上市时刻（如 06:00）落在当日 00:00
    那一根里，默认的 side="left" 会返回下一根，整条序列错开一天（事件研究从 `closes[0]`
    算、回测从下一根算，两者口径就不一致了）。

    比较用整数纳秒：tz-aware 索引的 `to_numpy()` 返回 Timestamp 对象数组（pandas 2.3），
    searchsorted 会退化成 Timestamp 比较并在 tz 上抛 TypeError；`asi8` 与 `Timestamp.value`
    都是 UTC 纳秒整数，比较无歧义。也不能用 get_indexer：上市时刻不会精确等于 bar
    时间戳，get_indexer 会返回 −1。
    """
    stamps = cast(pd.DatetimeIndex, index).asi8
    windows: dict[str, set[int]] = {}
    for event in events:
        position = int(stamps.searchsorted(event.list_time.value, side="right")) - 1
        if position < 0:
            continue  # 上市时刻早于面板起点，定位不到上市那一根
        start = position + entry
        windows[event.symbol] = set(range(start + 1, start + 1 + hold))
    return windows


def _daily_turnover(held: list[list[str]], positions: np.ndarray) -> np.ndarray:
    """每日换手比例：新进 + 退出 的仓位数量占当时持仓数的比例。

    分母取当日在场仓位，当日清空时回退到前一日——否则全部平仓那天的成本会被静默丢掉。
    """
    turnover = np.zeros(len(held), dtype=float)
    for day in range(1, len(held)):
        scale = positions[day] if positions[day] > 0 else positions[day - 1]
        if scale <= 0:
            continue
        previous = set(held[day - 1])
        current = set(held[day])
        turnover[day] = (len(current - previous) + len(previous - current)) / scale
    return turnover


def short_backtest(
    prices: pd.DataFrame,
    windows: dict[str, set[int]],
    *,
    entry: int = 1,
    hold: int = 90,
    fee_one_way: float = DEFAULT_FEE_ONE_WAY,
) -> ShortBacktest:
    """每日等权做空所有处于持仓窗口内的标的，成本按当日换手收。

    只统计当日有有效收益的标的：面板是各标的索引的并集，末尾未收盘的那一根与上市前的
    空档都是 NaN，当成 0% 收益会静默污染组合日收益（实测会让 entry=30 整列变 NaN）。
    """
    returns = prices.pct_change(fill_method=None)
    valid = returns.notna().to_numpy()
    column_of = {symbol: position for position, symbol in enumerate(prices.columns)}
    symbols = [column for column in prices.columns if windows.get(column)]
    held: list[list[str]] = [
        [
            symbol
            for symbol in symbols
            if day in windows[symbol] and bool(valid[day, column_of[symbol]])
        ]
        for day in range(len(prices))
    ]
    positions = np.array([len(names) for names in held], dtype=float)

    gross = np.array(
        [
            -float(returns.iloc[day][names].mean()) if names else 0.0
            for day, names in enumerate(held)
        ]
    )
    active = positions > 0
    cost = _daily_turnover(held, positions) * fee_one_way
    net = gross - cost
    # 清仓那一天 positions 为 0 但成本不为 0，掩码要把它算进来
    live_mask = active | (cost > 0)

    live = net[live_mask]
    curve = np.cumprod(1.0 + live)
    years = live.size / ANNUAL_DAYS
    drawdown = (
        float((curve / np.maximum.accumulate(curve) - 1.0).min()) if live.size else 0.0
    )
    held_days = positions[active]
    return ShortBacktest(
        entry=entry,
        hold=hold,
        fee_one_way=fee_one_way,
        days=int(live.size),
        avg_positions=float(held_days.mean()) if held_days.size else 0.0,
        total_return=float(curve[-1] - 1.0) if live.size else 0.0,
        annual_compounded=float(curve[-1] ** (1.0 / years) - 1.0)
        if live.size and years > 0
        else 0.0,
        annual_vol=float(live.std(ddof=1) * np.sqrt(ANNUAL_DAYS))
        if live.size > 1
        else 0.0,
        sharpe=(
            float(live.mean() / live.std(ddof=1) * np.sqrt(ANNUAL_DAYS))
            if live.size > 1 and live.std(ddof=1) > 0
            else 0.0
        ),
        max_drawdown=drawdown,
        positive_days=float((live > 0).mean()) if live.size else 0.0,
    )


def backtest_grid(
    prices: pd.DataFrame,
    events: list[ListingEvent],
    *,
    entries: tuple[int, ...] = (1, 7, 30),
    holds: tuple[int, ...] = (30, 90, 180),
    fee_one_way: float = DEFAULT_FEE_ONE_WAY,
) -> pd.DataFrame:
    """入场时点 × 持有期的二维扫描。"""
    rows: list[dict[str, float]] = []
    for entry in entries:
        for hold in holds:
            windows = holding_windows(
                events, cast(pd.DatetimeIndex, prices.index), entry=entry, hold=hold
            )
            result = short_backtest(
                prices, windows, entry=entry, hold=hold, fee_one_way=fee_one_way
            )
            rows.append(
                {
                    "entry": float(entry),
                    "hold": float(hold),
                    "avg_positions": result.avg_positions,
                    "annual_compounded": result.annual_compounded,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe,
                    "annual_vol": result.annual_vol,
                    "max_drawdown": result.max_drawdown,
                }
            )
    return pd.DataFrame(rows).set_index(["entry", "hold"])


PERP_SUFFIX = "-SWAP"
# 做空工具可得性的观察点（现货上市后第几天）
AVAILABILITY_DAYS: tuple[int, ...] = (0, 1, 7, 30, 90)


def fetch_perp_listing_times(
    exchange: object, *, settle: str = "USDT"
) -> dict[str, pd.Timestamp]:
    """拉 OKX 永续的上市时刻，键换成现货符号（BTC-USDT-SWAP → BTC-USDT）。"""
    payload = exchange.public_get_public_instruments({"instType": "SWAP"})["data"]  # type: ignore[attr-defined]
    times: dict[str, pd.Timestamp] = {}
    for row in payload:
        if row.get("settleCcy") != settle or row.get("state") != "live":
            continue
        inst_id = str(row["instId"])
        if not inst_id.endswith(PERP_SUFFIX):
            continue
        times[inst_id[: -len(PERP_SUFFIX)]] = cast(
            pd.Timestamp, pd.Timestamp(int(row["listTime"]), unit="ms", tz="UTC")
        )
    return times


def perp_availability(
    events: list[ListingEvent],
    perp_times: Mapping[str, pd.Timestamp],
    *,
    days: tuple[int, ...] = AVAILABILITY_DAYS,
) -> pd.DataFrame:
    """现货上市后第 d 天，有多少比例的新币已有永续可做空。

    这是「能不能做」的前置问题，也是本研究最容易误判的一处：只看本地永续日线缓存
    会得出「只有一成新币有永续」，而交易所实际是六成——缓存的覆盖范围是抓取限制，
    不是市场事实。任何工具可得性的结论都必须打交易所原始接口复核。
    """
    total = len(events)
    rows: list[dict[str, float]] = []
    for day in days:
        ready = 0
        for event in events:
            listed = perp_times.get(event.symbol)
            if listed is None:
                continue
            gap_days = (listed - event.list_time).total_seconds() / 86400.0
            if gap_days <= day:
                ready += 1
        rows.append(
            {
                "day": float(day),
                "ready": float(ready),
                "share": ready / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("day")
