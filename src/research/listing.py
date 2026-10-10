"""上新币效应：上市事件对齐与生命周期口径。

## 为什么可信样本从 2022 起

OKX 现货的 `listTime` 在 2021-01-29 08:08:06 被批量重置（57 只老标的同一秒），
而 2022 年及以后上市的标的三者完全吻合——`listTime` 与首根日线偏差 < 1 天，
即首根 bar 就是上市当日。更早的样本无法区分「上市日」与「数据起点截断」
（2020-01-01 聚集 22 只），故口径起点取 2022-01-01。

## 首日收益是机械假象

首根 bar 的 `open` **不是成交价**：极端案例里它是 0.03 / 0.05 / 0.1 / 1.0 这类整数
参考价，且 53.4% 的标的 `open == low`（当天没有成交落在 open 附近）。
按 open→close 算首日收益会得到中位 +11%、均值 +225% 的假象。
本模块只用 `close[0]` 作入场价（实测 95.1% 落在当日 low–high 之间）。

## 幸存者偏差是结构性的

样本是「当前仍在 OKX 交易」的标的：退市标的不可见（交易所不提供退市清单），
故所有收益被**系统性高估**。由此得到一条读表规则：
**负面结论对偏差免疫（真值只会更负），正面结论必须先扣掉偏差才能信。**
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import CRYPTO_ROOT

# listTime 可信起点：更早的标的时间戳被 OKX 批量重置到 2021-01-29
# stub 把 Timestamp() 的返回标成 Timestamp | NaTType，字面量运行时必为 Timestamp
MIN_LISTING_DATE = cast(pd.Timestamp, pd.Timestamp("2022-01-01", tz="UTC"))
# 生命周期分段的边界（根），第 i 段是 [SEGMENT_EDGES[i], SEGMENT_EDGES[i+1])
# 标签写的是事件时间（上市后第几天，1 起算）：从第 s 根收盘到第 e 根收盘
# 持有的是第 s+1..e 天，写成 "s+1-e" 才不会差一
SEGMENT_EDGES: tuple[int, ...] = (0, 1, 8, 31, 91, 181, 366)
SEGMENT_LABELS: tuple[str, ...] = ("1", "2-8", "9-31", "32-91", "92-181", "182-366")
# 前向收益的观察点（根）
HORIZONS: tuple[int, ...] = (1, 3, 7, 14, 30, 60, 90, 180, 365)
# 首日 open 与 low 的相对容差：差在这个比例内即视为「没有成交落在 open」
OPEN_AT_LOW_TOL = 1e-4
# 分位点
QUANTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True)
class ListingEvent:
    """一次上市事件：标的、上市时刻、以及上市起的完整收盘价序列。"""

    symbol: str
    list_time: pd.Timestamp
    closes: np.ndarray
    open_at_low: bool

    @property
    def bars(self) -> int:
        """上市至今的日线根数。"""
        return int(self.closes.size)

    @property
    def entry_price(self) -> float:
        """入场价取上市首日收盘——首日 open 是参考价不是成交价。"""
        return float(self.closes[0])


def fetch_listing_times(
    exchange: object, *, quote: str = "USDT"
) -> dict[str, pd.Timestamp]:
    """拉 OKX 现货的上市时刻；只返回 quote 计价且 state=live 的标的。

    退市标的不在返回里，这是幸存者偏差的来源，不是本函数的缺陷。
    """
    payload = exchange.public_get_public_instruments({"instType": "SPOT"})["data"]  # type: ignore[attr-defined]
    times: dict[str, pd.Timestamp] = {}
    for row in payload:
        if row.get("quoteCcy") != quote or row.get("state") != "live":
            continue
        times[str(row["instId"])] = cast(
            pd.Timestamp, pd.Timestamp(int(row["listTime"]), unit="ms", tz="UTC")
        )
    return times


def _open_at_low(frame: pd.DataFrame, tol: float = OPEN_AT_LOW_TOL) -> bool:
    """首根 bar 的 open 是否等于当日最低价（= 没有成交落在 open）。"""
    first = frame.iloc[0]
    return bool(float(first["open"]) <= float(first["low"]) * (1.0 + tol))


def load_events(
    listing_times: Mapping[str, pd.Timestamp],
    *,
    root: Path = CRYPTO_ROOT,
    market: str = "spot",
    timeframe: str = "1d",
    exchange: str = "okx",
    since: pd.Timestamp = MIN_LISTING_DATE,
    min_bars: int = 2,
) -> list[ListingEvent]:
    """把上市时刻与本地日线缓存配对成事件列表。

    缓存里没有的标的、上市早于 since 的标的、根数不足 min_bars 的标的都会被跳过。
    """
    directory = root / exchange / market / timeframe
    events: list[ListingEvent] = []
    for path in sorted(directory.glob("*.parquet")):
        list_time = listing_times.get(path.stem)
        if list_time is None or list_time < since:
            continue
        frame = pd.read_parquet(path)
        if len(frame) < min_bars:
            continue
        events.append(
            ListingEvent(
                symbol=path.stem,
                list_time=list_time,
                closes=frame["close"].to_numpy(dtype=float),
                open_at_low=_open_at_low(frame),
            )
        )
    return events


def holding_return(event: ListingEvent, entry: int, horizon: int) -> float:
    """从第 entry 根收盘买入、持有 horizon 根的收益；数据不够则返回 nan。

    入场与出场都用收盘价，engine 的「信号在 t 根、持仓在 t+1 根」在这里退化为
    「第 entry 根收盘成交」，因为事件研究不存在信号提前量。
    """
    if entry < 0 or horizon <= 0 or event.bars <= entry + horizon:
        return float("nan")
    return float(event.closes[entry + horizon] / event.closes[entry] - 1.0)


def _finite(values: Iterable[float]) -> np.ndarray:
    """去掉 nan 后转成数组。"""
    array = np.asarray(list(values), dtype=float)
    return array[~np.isnan(array)]


def horizon_table(
    events: list[ListingEvent], horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """从上市首日收盘买入、持有各 horizon 的收益分布。"""
    rows: list[dict[str, float]] = []
    for horizon in horizons:
        values = _finite(holding_return(event, 0, horizon) for event in events)
        row: dict[str, float] = {
            "horizon": float(horizon),
            "count": float(values.size),
            "positive": float((values > 0).mean()) if values.size else 0.0,
        }
        if values.size:
            row["mean"] = float(values.mean())
            row["median"] = float(np.median(values))
            for quantile in QUANTILES:
                row[f"p{int(quantile)}"] = float(np.percentile(values, quantile))
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")


def lifecycle_table(events: list[ListingEvent]) -> pd.DataFrame:
    """生命周期分段收益：每段独立计算（不是累计），看衰减发生在哪一段。"""
    rows: list[dict[str, object]] = []
    for label, start, stop in zip(
        SEGMENT_LABELS, SEGMENT_EDGES[:-1], SEGMENT_EDGES[1:], strict=True
    ):
        values = _finite(holding_return(event, start, stop - start) for event in events)
        rows.append(
            {
                "segment": label,
                "count": float(values.size),
                "median": float(np.median(values)) if values.size else 0.0,
                "mean": float(values.mean()) if values.size else 0.0,
                "positive": float((values > 0).mean()) if values.size else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def entry_table(
    events: list[ListingEvent],
    *,
    horizon: int = 90,
    entries: tuple[int, ...] = (0, 1, 3, 7, 14, 30, 60, 90, 120),
) -> pd.DataFrame:
    """入场时点扫描：等第 entry 根再买入，持有 horizon 根的收益。

    若各入场点的中位收益都为负，说明衰减是持续的而不是首日高估的一次性回吐。
    """
    rows: list[dict[str, float]] = []
    for entry in entries:
        values = _finite(holding_return(event, entry, horizon) for event in events)
        rows.append(
            {
                "entry": entry,
                "count": float(values.size),
                "median": float(np.median(values)) if values.size else 0.0,
                "mean": float(values.mean()) if values.size else 0.0,
                "positive": float((values > 0).mean()) if values.size else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("entry")


def cohort_table(events: list[ListingEvent], horizon: int = 90) -> pd.DataFrame:
    """按上市年份分组：年份越早，存活到今天的门槛越高（幸存者偏差的代理）。"""
    frame = pd.DataFrame(
        {
            "year": [event.list_time.year for event in events],
            "ret": [holding_return(event, 0, horizon) for event in events],
        }
    )
    grouped = frame.groupby("year")["ret"]
    return pd.DataFrame(
        {
            "count": grouped.count(),
            "median": grouped.median(),
            "positive": grouped.apply(lambda s: float((s > 0).mean())),
        }
    )


def short_risk_table(
    events: list[ListingEvent],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    wipeout_return: float = 1.0,
) -> pd.DataFrame:
    """做空新币的尾部风险：1x 空头在标的翻倍（+100%）时归零。

    中位收益深负不等于可做空：空头亏损随涨幅线性放大且不封顶，只要有一小部分标的
    翻倍，1x 空头就会被强平。这里直接给出「涨幅超过 wipeout_return 的比例」。
    """
    rows: list[dict[str, float]] = []
    for horizon in horizons:
        values = _finite(holding_return(event, 0, horizon) for event in events)
        rows.append(
            {
                "horizon": float(horizon),
                "count": float(values.size),
                "median": float(np.median(values)) if values.size else 0.0,
                "wipeout": float((values >= wipeout_return).mean())
                if values.size
                else 0.0,
                "worst": float(values.max()) if values.size else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def artifact_summary(events: list[ListingEvent]) -> dict[str, float]:
    """首日假象的量化：open 等于当日最低价的标的占比。"""
    if not events:
        return {"count": 0.0, "open_at_low_share": 0.0}
    return {
        "count": float(len(events)),
        "open_at_low_share": float(np.mean([event.open_at_low for event in events])),
    }
