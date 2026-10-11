"""
src/research/maker_conditional.py
条件化挂单：把「被动挂单能不能救反转」变成可测的问题。

背景（docs/execution/README.md）：1m 尺度的**无条件**被动挂单是负期望——挂得越深
省得越多，但「只有价格真的走了才成交」把这份便宜全额抵消，成交后继续走的距离是
常数（约 −1.1 ~ −1.4bp）。它列出的三条出路里，第二条是「在成交后价格确实会回归的
时机挂单」。反转信号正是那个时机：`docs/factors/README.md` 第四节测出反转的毛边际
真实存在（永续 3 根调仓毛 +38%/年），只是被 taker 费率吃掉。

本模块测的就是这条出路：**每天在反转选中的币上挂买单，成交后持有 N 根，量出**

1. **成交率**：挂单价被触及的比例（只判下一根，与 execution 文档同口径）；
2. **逆向选择** = 成交样本上「按挂单时点市价开仓」的收益 − 全样本市价开仓的收益。
   负值即成交样本本身更差——这是被动挂单的真实成本，与费率无关；
3. **每笔优势** = (挂单收益 − maker 费率) − (市价收益 − taker 费率)。
   它把「省下的费率」与「逆向选择」放在同一把尺子上：> 0 才说明被动挂单值。

三条口径约束，违反任一条结论都会变好看：

- **截面宽度**：窄于 `min_cross_section` 的交易日整日丢弃。1m 缓存里各标的起点不同，
  2020 年可交易截面中位只有 3 只——那种日子里的「分位」是 1 只币，不是截面。
- **池子规则**：复用 `factors/panel.py` 的 `tradable_mask`（流动性 + 上市天数），
  否则会把各币上市初期的暴涨算进来。
- **信号只用已知信息**：00:00 挂单用的是**前一日收盘**算出的 3 日反转（`shift(1)`）。

绝对收益**不可复利**：分位内是等权算术均值，crypto 的右尾会让它远高于实际到手
（样本期全市场日均 +17bp）。有意义的只有**同组内的差**（挂单 vs 市价）与**跨组比较**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_ROOT, crypto_bar_path, crypto_dataset_dir
from src.research.factors.panel import (
    Panel,
    UniverseRule,
    load_panel,
    tradable_mask,
)

BP: Final[float] = 1e4
# OKX Regular 档永续费率（docs/execution 的口径）：maker 2bp、taker 10bp
MAKER_FEE_BP: Final[float] = 2.0
TAKER_FEE_BP: Final[float] = 10.0
# 截面窄于这个数不做排名：与 factors 的 MIN_NAMES_PER_PERIOD 同一口径
MIN_CROSS_SECTION: Final[int] = 10
# 反转信号的回看根数（日线）与分位切点
REVERSAL_LOOKBACK: Final[int] = 3
LOSER_QUANTILE: Final[float] = 0.2
WINNER_QUANTILE: Final[float] = 0.8
GROUP_LOSER: Final[str] = "输家"
GROUP_MIDDLE: Final[str] = "中间"
GROUP_WINNER: Final[str] = "赢家"
BAR_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "execution"


@dataclass(frozen=True)
class MakerOptions:
    """测量口径。"""

    offsets_bp: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
    # 持有根数（1m）：1440 = 1 天，4320 = 3 天
    horizons_bars: tuple[int, ...] = (1440, 4320)
    min_cross_section: int = MIN_CROSS_SECTION
    lookback: int = REVERSAL_LOOKBACK
    timeframe: str = "1m"
    market: str = "swap"


@dataclass(frozen=True)
class SignalContext:
    """跨标的的选币上下文：分组、可交易掩码、截面宽度是否够。"""

    groups: pd.DataFrame
    tradable: pd.DataFrame
    wide: pd.Series


@dataclass(frozen=True)
class MakerCell:
    """一个 (组, 偏移, 持有) 单元的统计。"""

    group: str
    offset_bp: float
    horizon_bars: int
    events: int
    fill_rate: float
    # 成交样本上「按挂单价成交」的收益（含挂得便宜的那部分）
    maker_bp: float
    # 成交样本上「按挂单时点市价开仓」的收益：与全样本基准的差就是逆向选择
    cond_bp: float
    base_bp: float

    @property
    def adverse_bp(self) -> float:
        """逆向选择（bp）：负值 = 成交样本本身更差。"""
        return self.cond_bp - self.base_bp

    @property
    def per_trade_bp(self) -> float:
        """每笔优势：被动挂单相对吃价入场多赚多少（正 = 值得挂）。"""
        return (self.maker_bp - MAKER_FEE_BP) - (self.base_bp - TAKER_FEE_BP)

    @property
    def expected_bp(self) -> float:
        """每次尝试挂单的期望收益：没成交的那部分算 0（资金空转）。"""
        return self.fill_rate * (self.maker_bp - MAKER_FEE_BP)


def _ccxt_symbol(stem: str) -> str:
    """缓存文件名 → ccxt 永续符号（BTC-USDT → BTC/USDT:USDT）。"""
    return f"{stem.replace('-', '/')}:USDT"


def cached_symbols(options: MakerOptions) -> list[str]:
    """有日内缓存的永续标的（文件名口径）。"""
    venue = crypto_dataset_dir("okx", options.market, options.timeframe)
    return sorted(path.stem for path in venue.glob("*.parquet"))


def signal_context(
    options: MakerOptions, rule: UniverseRule | None = None
) -> tuple[Panel, SignalContext]:
    """读日线面板，算出每日的选币分组与截面宽度门槛。"""
    rule = rule or UniverseRule()
    panel = load_panel("1d", rule, market=options.market)
    symbols = [s for s in cached_symbols(options) if s in panel.close.columns]
    close = panel.close[symbols]
    tradable = cast(pd.DataFrame, tradable_mask(panel, rule)[symbols])
    # 方向调整后的反转：数值越大 = 前 lookback 日跌得越狠（= 越该买）
    signal = (-(close / close.shift(options.lookback) - 1)).shift(1)
    ranks = signal.rank(axis=1, pct=True)
    labels = np.select(
        [ranks.to_numpy() <= LOSER_QUANTILE, ranks.to_numpy() >= WINNER_QUANTILE],
        [GROUP_LOSER, GROUP_WINNER],
        default=GROUP_MIDDLE,
    )
    # 信号为 NaN 的格子（该标的当日还没有足够历史）必须留空被剔除，不能默认成「中间」：
    # 那会把「没有信号」当成「信号中性」，把噪音算进中间组的统计里
    groups = pd.DataFrame(labels, index=ranks.index, columns=ranks.columns).where(
        ranks.notna()
    )
    wide = (tradable.sum(axis=1) >= options.min_cross_section) & signal.notna().any(
        axis=1
    )
    return panel, SignalContext(groups=groups, tradable=tradable, wide=wide)


def _day_stamps(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把分钟级时戳归一到自然日（stub 里 DatetimeIndex.floor 不可用，走 Series 的 dt）。

    直接对 tz-aware 的索引取 numpy datetime64 会丢时区并报警告，所以先钉 UTC 再转。
    """
    stamps = pd.Series(index).dt.tz_convert("UTC").dt.normalize()
    return pd.DatetimeIndex(stamps)


def _first_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """每日首根的布尔掩码（相邻自然日不同即为首根）。"""
    keys = pd.Series(index).dt.tz_convert("UTC").dt.normalize().to_numpy()
    return np.concatenate([[True], keys[1:] != keys[:-1]])


def symbol_events(
    symbol: str, context: SignalContext, options: MakerOptions
) -> pd.DataFrame:
    """单标的的逐日挂单明细：成交与否 + 三种收益口径。"""
    frame = pd.read_parquet(
        crypto_bar_path("okx", _ccxt_symbol(symbol), options.timeframe),
        columns=list(BAR_COLUMNS),
    ).reset_index()
    index = pd.DatetimeIndex(frame["datetime"])
    close = frame["close"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    # 每日首根（00:00 UTC）挂单：日线信号在此时点已知
    position = np.flatnonzero(_first_of_day(index))
    days = _day_stamps(index)[position]
    keep = (
        context.tradable[symbol].reindex(days).fillna(False).to_numpy()
        & context.wide.reindex(days).fillna(False).to_numpy()
    )
    records: list[pd.DataFrame] = []
    for horizon in options.horizons_bars:
        for offset in options.offsets_bp:
            usable = keep & (position + horizon + 1 < close.size)
            bars = position[usable]
            limit = close * (1.0 - offset / BP)
            hit = low[bars + 1] <= limit[bars]
            exit_price = close[bars + horizon]
            records.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "day": days[usable],
                        "group": context.groups[symbol]
                        .reindex(days[usable])
                        .to_numpy(),
                        "offset_bp": offset,
                        "horizon_bars": horizon,
                        "hit": hit,
                        "maker_bp": (exit_price / limit[bars] - 1.0) * BP,
                        "cond_bp": (exit_price / close[bars] - 1.0) * BP,
                    }
                )
            )
    events = pd.concat(records, ignore_index=True)
    return events.dropna(subset=["group"])


def load_events(options: MakerOptions) -> pd.DataFrame:
    """跑完全部标的，返回事件级明细。"""
    _, context = signal_context(options)
    symbols = [s for s in cached_symbols(options) if s in context.groups.columns]
    frames = [symbol_events(symbol, context, options) for symbol in symbols]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(events: pd.DataFrame, options: MakerOptions) -> pd.DataFrame:
    """按 (组, 偏移, 持有) 聚合出成交率与三种收益。"""
    del options  # 口径已烘进事件明细，聚合不再需要参数（保留签名供调用方统一）
    cells: list[dict[str, object]] = []
    for key, block in events.groupby(["group", "offset_bp", "horizon_bars"]):
        group, offset, horizon = cast(tuple[str, float, int], key)
        filled = block["hit"].to_numpy(dtype=bool)
        cell = MakerCell(
            group=str(group),
            offset_bp=float(offset),
            horizon_bars=int(horizon),
            events=int(block.shape[0]),
            fill_rate=float(filled.mean()) if filled.size else 0.0,
            maker_bp=float(block.loc[filled, "maker_bp"].mean()),
            cond_bp=float(block.loc[filled, "cond_bp"].mean()),
            base_bp=float(block["cond_bp"].mean()),
        )
        cells.append(
            {
                "group": cell.group,
                "offset_bp": cell.offset_bp,
                "horizon_bars": cell.horizon_bars,
                "events": cell.events,
                "fill_rate": cell.fill_rate,
                "maker_bp": cell.maker_bp,
                "cond_bp": cell.cond_bp,
                "base_bp": cell.base_bp,
                "adverse_bp": cell.adverse_bp,
                "per_trade_bp": cell.per_trade_bp,
                "expected_bp": cell.expected_bp,
            }
        )
    frame = pd.DataFrame(cells)
    return frame.sort_values(["horizon_bars", "group", "offset_bp"]).reset_index(
        drop=True
    )


def save_outputs(frame: pd.DataFrame, out_dir: Path = OUTPUT_DIR) -> list[Path]:
    """落盘：单元格表与年度分解各一份。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in (("maker_cells", frame),):
        path = out_dir / f"{name}.parquet"
        table.to_parquet(path)
        paths.append(path)
    return paths


def annual_frame(events: pd.DataFrame) -> pd.DataFrame:
    """按年分解：结论随市场状态翻转，只看全样本会被单一 regime 主导。"""
    block = events.copy()
    # stub 里 DatetimeIndex.year 不可用，走 Series 的 dt 访问器
    block["year"] = cast(pd.Series, block["day"]).dt.year
    rows: list[dict[str, object]] = []
    for key, cell in block.groupby(["year", "group"]):
        year, group = cast(tuple[int, str], key)
        filled = cell["hit"].to_numpy(dtype=bool)
        maker = float(cell.loc[filled, "maker_bp"].mean())
        base = float(cell["cond_bp"].mean())
        rows.append(
            {
                "year": int(year),
                "group": str(group),
                "events": int(cell.shape[0]),
                "fill_rate": float(filled.mean()),
                "maker_bp": maker,
                "base_bp": base,
                "adverse_bp": float(cell.loc[filled, "cond_bp"].mean()) - base,
                "per_trade_bp": (maker - MAKER_FEE_BP) - (base - TAKER_FEE_BP),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "year"]).reset_index(drop=True)
