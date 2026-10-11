"""
src/research/factors/sweep.py
因子扫描器：把「手挑因子」换成「按族扫窗口」，把判据从「点显著」换成「平台显著」。

三条口径：

1. **不预设方向**：扫描用原始因子值（方向恒为 `DIRECTION_LONG`），IC 的符号就是
   发现本身。`mom(w)` 与 `rev(w)` 是同一个计算式的两个符号，所以只扫一个族。
2. **平台而非尖峰**：判据在 `plateau.py`——真实的因子效应对窗口规格不敏感，要求
   连续多个窗口同号且显著；只有孤立窗口显著的一律丢。
3. **有效试验数**：相邻窗口的 IC 序列相关 0.9+，按名义候选数做 Bonferroni 会把真
   信号一起杀掉；不做校正则扫几百个窗口必然撞出几个「显著」的噪音。用 IC 序列
   相关矩阵的特征值谱估有效试验数（Li-Ji），门槛按它算。

**本模块只做信号层（IC），不算组合收益与成本**——这是刻意的：发现要便宜、验证才贵。
平台筛出的窗口再交给 `portfolio.py` 做收益层。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, cast

import numpy as np
import pandas as pd

from src.research.factors.evaluation import (
    forward_returns,
    non_overlapping_t,
    rank_ic_series,
)
from src.research.factors.library import (
    DIRECTION_LONG,
    FactorSpec,
    distance_to_high,
    log_notional,
    momentum,
    volatility,
)
from src.research.factors.panel import Panel
from src.research.factors.plateau import (
    DEFAULT_MIN_PLATEAU,
    plateau_table,
    verdict_table,
)

# === 扫描空间 ===
# 动量族从 1 根起（含 1 根反转）；其余族要满窗口才有值
MOMENTUM_WINDOWS: Final[tuple[int, ...]] = tuple(range(1, 121))
LONG_WINDOWS: Final[tuple[int, ...]] = tuple(range(5, 121))
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 15, 30)
DEFAULT_ALPHA: Final[float] = 0.05


@dataclass(frozen=True)
class SweepFamily:
    """一个可扫描的因子族：同一计算式取不同窗口。"""

    name: str
    label: str
    windows: tuple[int, ...]
    warmup_of: Callable[[int], int]
    compute: Callable[[Panel, int], pd.DataFrame]

    def spec(self, window: int) -> FactorSpec:
        """把 (族, 窗口) 变成一个 FactorSpec；方向恒为 +1，IC 的符号就是发现。"""
        return FactorSpec(
            key=f"{self.name}_{window}",
            label=f"{window} 根{self.label}",
            family=self.name,
            direction=DIRECTION_LONG,
            warmup=self.warmup_of(window),
            compute=lambda panel: self.compute(panel, window),
        )


SWEEP_FAMILIES: dict[str, SweepFamily] = {
    family.name: family
    for family in (
        SweepFamily(
            "mom", "动量（负号即反转）", MOMENTUM_WINDOWS, lambda w: w, momentum
        ),
        SweepFamily("vol", "已实现波动", LONG_WINDOWS, lambda w: w, volatility),
        SweepFamily(
            "liq", "中位名义额（对数）", LONG_WINDOWS, lambda w: w - 1, log_notional
        ),
        SweepFamily("dth", "距最高点", LONG_WINDOWS, lambda w: w - 1, distance_to_high),
    )
}


def resolve_families(names: str | None = None) -> list[SweepFamily]:
    """把逗号分隔的族名解析成列表；省略则取全部。未知族名当场报错。"""
    if not names:
        return list(SWEEP_FAMILIES.values())
    chosen = [item.strip() for item in names.split(",")]
    unknown = [name for name in chosen if name not in SWEEP_FAMILIES]
    if unknown:
        raise KeyError(
            f"未知因子族 {', '.join(unknown)}；可用：{', '.join(SWEEP_FAMILIES)}"
        )
    return [SWEEP_FAMILIES[name] for name in chosen]


# === 显著性门槛：按有效试验数校正 ===


def effective_tests(series: pd.DataFrame) -> float:
    """有效试验数：IC 序列相关矩阵特征值谱的 Li-Ji 估计 `(Σλ)² / Σλ²`。

    相邻窗口的 IC 序列几乎同一条线，名义上几百个检验其实只有几十个独立方向。
    按名义数做 Bonferroni 会把真信号一起杀掉，不校正则必然撞出噪音。
    两两不相交的候选（从无重叠期次）视为不相关，取 0。
    """
    count = series.shape[1]
    if count <= 1:
        return float(count)
    matrix = np.nan_to_num(np.asarray(series.corr().to_numpy(dtype=float), dtype=float))
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        return float(count)
    return float(positive.sum() ** 2 / (positive**2).sum())


def critical_t(tests: float, alpha: float = DEFAULT_ALPHA) -> float:
    """双侧 Bonferroni 门槛（正态近似，与既有套件的 2.99 / 3.48 同口径）。"""
    return float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(tests, 1.0))))


# === 扫描 ===


@dataclass(frozen=True)
class SweepResult:
    """一次扫描的全部产物。"""

    # 长表：key/family/window/horizon/ic/t_stat/periods/independent
    ic: pd.DataFrame
    # 逐期限的名义与有效试验数、两套门槛
    tests: pd.DataFrame
    # 全部连续段，含 kind 列（平台 / 尖峰）
    plateaus: pd.DataFrame
    # 逐 (族, 期限) 的平台汇总（尖峰不参与）
    verdicts: pd.DataFrame


def sweep_ic_table(
    panel: Panel,
    mask: pd.DataFrame,
    horizons: Sequence[int],
    families: Sequence[SweepFamily],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐 (族, 窗口, 期限) 算 IC 与非重叠 t；同时留下 IC 序列供有效试验数用。"""
    forwards = {horizon: forward_returns(panel, horizon) for horizon in horizons}
    rows: list[dict[str, object]] = []
    columns: list[pd.Series] = []
    for family in families:
        for window in family.windows:
            values = family.spec(window).values(panel)
            for horizon, forward in forwards.items():
                ic = rank_ic_series(values, forward, mask)
                if ic.empty:
                    continue
                t_stat, independent = non_overlapping_t(ic, horizon)
                key = f"{family.name}_{window}@{horizon}"
                columns.append(ic.rename(key))
                rows.append(
                    {
                        "key": key,
                        "family": family.name,
                        "window": window,
                        "horizon": horizon,
                        "ic": float(ic.mean()),
                        "t_stat": t_stat,
                        "periods": int(ic.size),
                        "independent": independent,
                    }
                )
    table = pd.DataFrame(rows)
    series = pd.concat(columns, axis=1) if columns else pd.DataFrame()
    return table, series


def test_thresholds(
    table: pd.DataFrame, series: pd.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """逐期限算名义/有效试验数与两套门槛：有效试验数才是该用的那个。"""
    rows: list[dict[str, object]] = []
    for raw_horizon, group in table.groupby("horizon"):
        keys = [str(key) for key in group["key"]]
        effective = effective_tests(cast(pd.DataFrame, series[keys]))
        nominal = float(group.shape[0])
        rows.append(
            {
                "horizon": int(cast(int, raw_horizon)),
                "nominal_tests": int(nominal),
                "effective_tests": effective,
                "critical_t": critical_t(effective, alpha),
                "bonferroni_t": critical_t(nominal, alpha),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def sweep(
    panel: Panel,
    mask: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    families: Sequence[SweepFamily] | None = None,
    min_plateau: int = DEFAULT_MIN_PLATEAU,
    alpha: float = DEFAULT_ALPHA,
) -> SweepResult:
    """跑一次完整扫描：IC 表 → 有效试验数校正 → 平台判定 → 汇总裁决。"""
    chosen = list(families) if families is not None else list(SWEEP_FAMILIES.values())
    table, series = sweep_ic_table(panel, mask, horizons, chosen)
    if table.empty:
        empty = pd.DataFrame()
        return SweepResult(ic=empty, tests=empty, plateaus=empty, verdicts=empty)
    thresholds = test_thresholds(table, series, alpha)
    plateaus = plateau_table(table, thresholds, min_plateau)
    return SweepResult(
        ic=table,
        tests=thresholds,
        plateaus=plateaus,
        verdicts=verdict_table(plateaus),
    )
