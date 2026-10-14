"""
棒尺度上的反转因子：同一份前向收益与掩码，只换度量尺度。

本仓库的结论都建立在时间棒上（「3 根反转」= 3 个日历日）。信息驱动采样（美元棒、量棒、
失衡棒）把同一句话变成「3 根信息棒」。要比两者，必须让**除尺度外的一切都相同**：同一批
标的、同一份前向收益、同一个可交易掩码、同一个窗口号。

两个会静默算错的地方，`rescaled_reversal` 里已按正确方式实现：

1. **反转必须逐标的在自己的棒序列上算**。各标的的棒时戳互不相同，先拼成 DataFrame 再
   `shift(w)` 会变成跨标的乱移——实测 IC 处处恰好 0.0000，很容易被读成「信号不存在」。
2. **日线索引是 K 线的起始时刻**，`close[D]` 是 D 日末价格；按日采样要取「D 日收盘之前
   已收盘的最后一根棒」，故目标时刻要加一天，否则因子白落后一天。
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.research.factors.evaluation import non_overlapping_t, rank_ic_series


def load_bar_closes(root: Path) -> dict[str, pd.Series]:
    """读一种棒的收盘序列（索引是每根的收盘时刻）——目录名即棒种。"""
    return {
        path.stem: cast(pd.Series, pd.read_parquet(path, columns=["close"])["close"])
        for path in sorted(root.glob("*.parquet"))
    }


def bar_reversal(close: pd.Series, window: int) -> pd.Series:
    """w 根反转：取负号，值越大越看多（与 library 的方向约定一致）。"""
    return cast(pd.Series, -(close / cast(pd.Series, close.shift(window)) - 1.0))


def calendar_reversal(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """日历尺度：日线收盘上的 w 日反转。"""
    ratio = close / cast(pd.DataFrame, close.shift(window))
    return cast(pd.DataFrame, -(ratio - 1.0))


def rescaled_reversal(
    closes: dict[str, pd.Series], daily_index: pd.DatetimeIndex, window: int
) -> pd.DataFrame:
    """信息尺度：各标的自己棒序列上的 w 根反转，再按日采样到面板索引。"""
    target = daily_index + np.timedelta64(1, "D")
    sampled = {
        symbol: bar_reversal(close, window).reindex(target, method="ffill")
        for symbol, close in closes.items()
    }
    frame = pd.DataFrame(sampled)
    frame.index = daily_index
    if bool(frame.isna().to_numpy().all()):
        raise ValueError("信息尺度因子全为 NaN：棒时戳与面板索引对不上，检查抽样目标日")
    return frame


def keep_symbols(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """把面板裁到给定标的集。"""
    return cast(pd.DataFrame, frame[[sym for sym in symbols if sym in frame.columns]])


def ic_row(
    label: str,
    window: int,
    horizon: int,
    values: pd.DataFrame,
    forward: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[str, object]:
    """一个（尺度 × 窗口 × 期限）的 IC 读数。

    t 值只认非重叠子样本（与 `ic_stats` 同一口径）：重叠样本会把显著性虚高。
    """
    row: dict[str, object] = {
        "scale": label,
        "window": window,
        "horizon": horizon,
    }
    series = rank_ic_series(values, forward, mask)
    if series.empty:
        row.update({"mean_ic": 0.0, "t_stat": 0.0, "periods": 0, "independent": 0})
        return row
    t_stat, independent = non_overlapping_t(series, horizon)
    row.update(
        {
            "mean_ic": float(series.mean()),
            "t_stat": t_stat,
            "periods": int(series.size),
            "independent": independent,
        }
    )
    return row
