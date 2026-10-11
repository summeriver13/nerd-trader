"""
src/research/factors/plateau.py
平台判据：扫描出来的显著性，哪些是「效应」、哪些是「规格」。

扫描器会产出几百个 (族, 窗口) 的显著性。如果按「哪个窗口的 |t| 最大」来挑，
挑到的必然是噪音——每个族有上百个窗口，最大的那个 |t| 就是极值分布的上尾。

本模块的判据是：**真实的因子效应对窗口规格不敏感**。

- **平台**：连续 ≥ K 个窗口同号、且每个窗口的 |t| 都过了门槛。说明效应在一个
  规格区间里都成立，换个窗口也看得见。
- **尖峰**：孤立的显著窗口。相邻窗口不显著意味着「换个窗口就没了」，是过拟合。

只连接**窗口号连续**的段：`w` 与 `w+2` 各自显著不算同一段，中间那个窗口没通过
就说明效应在规格上不稳定。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, cast

import numpy as np
import pandas as pd

# 连续多少个窗口同号且过门槛才算平台
DEFAULT_MIN_PLATEAU: Final[int] = 3

KIND_PLATEAU: Final[str] = "平台"
KIND_SPIKE: Final[str] = "尖峰"


@dataclass(frozen=True)
class Plateau:
    """一段连续窗口：同号，且段内每个窗口的 |t| 都过了门槛。"""

    family: str
    horizon: int
    start: int
    end: int
    windows: int
    mean_ic: float
    min_abs_t: float


def find_runs(
    family: str, horizon: int, ic: pd.Series, t_stat: pd.Series, threshold: float
) -> list[Plateau]:
    """把「同号且 |t| 过门槛」的相邻窗口连成段；返回全部段（含长度为 1 的尖峰）。"""
    windows = np.asarray(ic.index, dtype=int)
    values = np.asarray(ic.to_numpy(dtype=float), dtype=float)
    tvals = np.asarray(t_stat.reindex(ic.index).to_numpy(dtype=float), dtype=float)
    passed = np.isfinite(values) & np.isfinite(tvals) & (np.abs(tvals) > threshold)
    signs = np.where(passed & (values != 0.0), np.sign(values), 0.0)

    runs: list[Plateau] = []
    start = 0
    for position in range(1, windows.size + 1):
        joined = (
            position < windows.size
            and signs[position] == signs[start]
            and signs[position] != 0.0
            and windows[position] == windows[position - 1] + 1
        )
        if joined:
            continue
        if signs[start] != 0.0:
            runs.append(
                _summarize(family, horizon, windows, values, tvals, start, position - 1)
            )
        start = position
    return runs


def _summarize(
    family: str,
    horizon: int,
    windows: np.ndarray,
    values: np.ndarray,
    tvals: np.ndarray,
    start: int,
    end: int,
) -> Plateau:
    """把一段下标区间收成一个 Plateau。"""
    chunk = slice(start, end + 1)
    return Plateau(
        family=family,
        horizon=horizon,
        start=int(windows[start]),
        end=int(windows[end]),
        windows=int(end - start + 1),
        mean_ic=float(values[chunk].mean()),
        min_abs_t=float(np.abs(tvals[chunk]).min()),
    )


def plateau_table(
    table: pd.DataFrame,
    thresholds: pd.DataFrame,
    min_plateau: int = DEFAULT_MIN_PLATEAU,
) -> pd.DataFrame:
    """逐 (族, 期限) 找连续段，按长度标注平台还是尖峰。

    `thresholds` 是逐期限的临界 |t| 表（含 horizon 与 critical_t 两列）。
    """
    if table.empty:
        return pd.DataFrame()
    limits = dict(zip(thresholds["horizon"], thresholds["critical_t"], strict=True))
    rows: list[dict[str, object]] = []
    for key, group in table.groupby(["family", "horizon"]):
        family, horizon = cast(tuple[str, int], key)
        ordered = group.sort_values("window")
        index = ordered["window"].to_numpy(dtype=int)
        ic = pd.Series(ordered["ic"].to_numpy(dtype=float), index=index)
        t_stat = pd.Series(ordered["t_stat"].to_numpy(dtype=float), index=index)
        for run in find_runs(family, horizon, ic, t_stat, float(limits[horizon])):
            record = asdict(run)
            record["kind"] = KIND_PLATEAU if run.windows >= min_plateau else KIND_SPIKE
            rows.append(record)
    return pd.DataFrame(rows)


def verdict_table(plateaus: pd.DataFrame) -> pd.DataFrame:
    """逐 (族, 期限) 汇总平台：段数、最长跨度、平台内平均 IC。尖峰不参与。"""
    if plateaus.empty:
        return pd.DataFrame()
    real = plateaus[plateaus["kind"] == KIND_PLATEAU]
    if real.empty:
        return pd.DataFrame()
    grouped = cast(
        pd.DataFrame,
        real.groupby(["family", "horizon"], as_index=False).agg(
            plateaus=("windows", "size"),
            longest=("windows", "max"),
            mean_ic=("mean_ic", "mean"),
        ),
    )
    ordered = grouped.sort_values("longest", ascending=False)
    return ordered.reset_index(drop=True)
