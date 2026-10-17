"""
src/research/liquidity/maker_spread.py
两腿被动挂单的组合层：反转价差用 maker 进场能不能真赚钱。

`maker_conditional.py` 测的是**单腿**（在输家上挂买单）的成交质量，结论是逆向选择
几乎为零、每笔优势 +8 ~ +22bp。但反转策略是**两腿**：买输家、卖赢家，两腿都要被动
成交才算真中性。这一步要回答三件单腿测不出来的事：

1. **两腿的成交是否联动**：若市场下跌时买腿成交、卖腿不成交，组合会在下跌中变成
   净多头——这正是「被动挂单 + 双腿策略」最危险的地方；
2. **空仓期的拖累**：未成交的那部分按 0 收益计（资金空转），价差收益被成交率稀释；
3. **净收益**：进场按 maker 2bp、出场按 taker 10bp（保守：不假设出场也能挂上），
   与「两腿全吃价」的对照比较。

口径与单腿模块一致：00:00 挂单、下一根 1m 判触及、信号用前一日收盘、池子规则复用
`tradable_mask`、截面窄于 `min_cross_section` 整日丢弃。**出场一律吃价**——这一层
只隔离「进场方式」的效果，不引入出场成交模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_ROOT, crypto_bar_path
from src.research.liquidity.maker_conditional import (
    BAR_COLUMNS,
    BP,
    GROUP_LOSER,
    GROUP_WINNER,
    MAKER_FEE_BP,
    TAKER_FEE_BP,
    MakerOptions,
    SignalContext,
    _ccxt_symbol,
    _day_stamps,
    _first_of_day,
    cached_symbols,
    signal_context,
)

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "execution"
SIDE_LONG: Final[str] = "long"
SIDE_SHORT: Final[str] = "short"


@dataclass(frozen=True)
class SpreadOptions(MakerOptions):
    """两腿组合的测量口径：在单腿口径上只多两项费率。

    继承 `MakerOptions` 是为了直接复用 `signal_context` / `cached_symbols` 的选币与
    池子逻辑——两腿与单腿必须用**同一套**分组，否则两者不可比。
    """

    # 进场费率：maker 2bp（默认）或 taker 10bp（对照）
    entry_fee_bp: float = MAKER_FEE_BP
    # 出场一律吃价：这一层不建模出场成交
    exit_fee_bp: float = TAKER_FEE_BP


def symbol_legs(
    symbol: str, context: SignalContext, options: SpreadOptions
) -> pd.DataFrame:
    """单标的的腿明细：输家走买腿（挂在收盘下方）、赢家走卖腿（挂在上方）。

    中间组不参与（反转价差只做两端）。腿收益是**含费率**的净收益，未成交不产生行
    （由调用方按 0 计）。
    """
    frame = pd.read_parquet(
        crypto_bar_path("okx", _ccxt_symbol(symbol), options.timeframe),
        columns=list(BAR_COLUMNS),
    ).reset_index()
    index = pd.DatetimeIndex(frame["datetime"])
    close = frame["close"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    position = np.flatnonzero(_first_of_day(index))
    days = _day_stamps(index)[position]
    groups = context.groups[symbol].reindex(days).to_numpy()
    keep = (
        context.tradable[symbol].reindex(days, fill_value=False).to_numpy()
        & context.wide.reindex(days, fill_value=False).to_numpy()
    )
    side = np.where(
        groups == GROUP_LOSER,
        SIDE_LONG,
        np.where(groups == GROUP_WINNER, SIDE_SHORT, ""),
    )
    records: list[pd.DataFrame] = []
    fee_bp = options.entry_fee_bp + options.exit_fee_bp
    for horizon in options.horizons_bars:
        for offset in options.offsets_bp:
            usable = keep & (side != "") & (position + horizon + 1 < close.size)
            bars = position[usable]
            if bars.size == 0:
                continue
            is_long = side[usable] == SIDE_LONG
            entry = close[bars]
            limit = np.where(
                is_long, entry * (1.0 - offset / BP), entry * (1.0 + offset / BP)
            )
            touched = np.where(is_long, low[bars + 1] <= limit, high[bars + 1] >= limit)
            # 买腿收益 = 到期/限价 − 1；卖腿收益 = −(到期/限价 − 1)
            gross = (close[bars + horizon] / limit - 1.0) * BP
            records.append(
                pd.DataFrame(
                    {
                        "day": days[usable],
                        "side": side[usable],
                        "offset_bp": offset,
                        "horizon_bars": horizon,
                        "hit": touched,
                        "ret_bp": np.where(is_long, gross, -gross) - fee_bp,
                    }
                )
            )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def load_legs(options: SpreadOptions) -> pd.DataFrame:
    """跑完全部标的的腿明细。"""
    _, context = signal_context(options)
    symbols = [s for s in cached_symbols(options) if s in context.groups.columns]
    frames = [symbol_legs(s, context, options) for s in symbols]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _leg_bp(side: pd.DataFrame, realized: bool) -> float:
    """一腿的等权收益。

    `realized=True` 时未成交按 0 计（资金已分配给该标的、但没挂上，等于空转）；
    `False` 时是「假设一定成交」的反事实，作对照用。两者之差就是空仓拖累。
    """
    if side.empty:
        return 0.0
    values = side["ret_bp"] * side["hit"] if realized else side["ret_bp"]
    return float(values.mean())


def daily_frame(legs: pd.DataFrame, options: SpreadOptions) -> pd.DataFrame:
    """按日合成价差：两腿各自等权，未成交的标的按 0 计（资金空转）。"""
    rows: list[dict[str, object]] = []
    for key, block in legs.groupby(["day", "offset_bp", "horizon_bars"]):
        day, offset, horizon = cast(tuple[pd.Timestamp, float, int], key)
        long_side = cast(pd.DataFrame, block[block["side"] == SIDE_LONG])
        short_side = cast(pd.DataFrame, block[block["side"] == SIDE_SHORT])
        rows.append(
            {
                "day": day,
                "offset_bp": float(offset),
                "horizon_bars": int(horizon),
                "long_bp": _leg_bp(long_side, realized=True),
                "short_bp": _leg_bp(short_side, realized=True),
                "long_all_bp": _leg_bp(long_side, realized=False),
                "short_all_bp": _leg_bp(short_side, realized=False),
                "long_fill": float(long_side["hit"].mean()) if long_side.size else 0.0,
                "short_fill": (
                    float(short_side["hit"].mean()) if short_side.size else 0.0
                ),
                "long_names": int(long_side.shape[0]),
                "short_names": int(short_side.shape[0]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["spread_bp"] = frame["long_bp"] - frame["short_bp"]
    frame["spread_all_bp"] = frame["long_all_bp"] - frame["short_all_bp"]
    # 两腿都至少有一只成交才算「真中性」；否则是单边敞口
    frame["both_fill"] = (frame["long_fill"] > 0) & (frame["short_fill"] > 0)
    frame["imbalance"] = (frame["long_fill"] - frame["short_fill"]).abs()
    return frame.sort_values(["horizon_bars", "offset_bp", "day"]).reset_index(
        drop=True
    )


def summary_frame(daily: pd.DataFrame, options: SpreadOptions) -> pd.DataFrame:
    """按 (偏移, 持有) 汇总：成交率、敞口不平衡、价差收益与算术年化。

    同时给均值与中位数：每条腿只有 3 ~ 4 个名字，单腿单日能到 ±8500bp，
    价差的分布是重尾的——均值受单个名字主导，中位数才是稳健的中心。
    两者背离本身就是「这个池子太窄」的信号。
    """
    rows: list[dict[str, object]] = []
    for key, block in daily.groupby(["offset_bp", "horizon_bars"]):
        offset, horizon = cast(tuple[float, int], key)
        periods_per_year = 365.0 * 1440.0 / float(horizon)
        spread = cast(pd.Series, block["spread_bp"])
        rows.append(
            {
                "offset_bp": float(offset),
                "horizon_bars": int(horizon),
                "days": int(block.shape[0]),
                "long_fill": float(block["long_fill"].mean()),
                "short_fill": float(block["short_fill"].mean()),
                "both_fill": float(block["both_fill"].mean()),
                "imbalance": float(block["imbalance"].mean()),
                "spread_bp": float(spread.mean()),
                "spread_all_bp": float(block["spread_all_bp"].mean()),
                "median_bp": float(spread.median()),
                "win_rate": float((spread > 0).mean()),
                "annual_pct": float(spread.mean()) * periods_per_year / 100.0,
                "worst_bp": float(spread.min()),
                "std_bp": float(spread.std(ddof=1)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["horizon_bars", "offset_bp"])
        .reset_index(drop=True)
    )


def save_outputs(
    summary: pd.DataFrame, daily: pd.DataFrame, out_dir: Path = OUTPUT_DIR
) -> list[Path]:
    """落盘：汇总表与逐日价差各一份。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in (("maker_spread", summary), ("maker_spread_daily", daily)):
        path = out_dir / f"{name}.parquet"
        table.to_parquet(path)
        paths.append(path)
    return paths
