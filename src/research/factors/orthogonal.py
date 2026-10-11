"""
src/research/factors/orthogonal.py
正交化检验：`vol_30` 与 `liq_30` 是不是同一条信号。

这是 docs/factors/README.md「下一步」里唯一可能**推翻**既有结论的检验。两个因子的
单调性同为 +0.70、截面相关 −0.25，而低波动腿的成分全是 2017–2021 的老品牌币——
「低波动异象」很可能只是「大市值/成熟币跑赢」的另一个说法（`liq_30` 的正名）。

两种证法互为旁证，一个参数化、一个非参数：

1. **残差 IC**：逐期把目标因子的截面秩对控制因子的截面秩回归，取残差再算 Spearman。
   残差 IC 与原始 IC 同号且量级不塌 → 目标因子带的是控制因子之外的信息；塌到 0 →
   它只是控制变量的代理。回归在**秩空间**做：厚尾下用原始值回归，单个 +400% 的币
   就能主导整期斜率。
2. **双重排序**：先按控制因子分 5 档，再在各档**内部**算目标因子的 IC。它不假设
   线性关系，只问「同一档流动性里，低波动还灵不灵」。

除流动性外，还检验**上市天数**：文档的另一条解释是「低波动只是老币的代理」。
上市天数从首根 K 线起算，早于缓存起点的币（BTC/ETH）按缓存起点计，所以它在样本
早期会把老币读成新币——这个偏差方向是安全的（只会漏掉老币，不会凭空造出老币）。

读表规则：`decay`（控制后保留比例）是主判据；双重排序看**符号是否在每一档都一致**——
只在某一档灵、或符号翻转，说明效应来自档位划分而不是因子本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd

from src.research.factors.evaluation import (
    forward_returns,
    ic_stats,
    non_overlapping_t,
    residual_ic_series,
    spearman,
)
from src.research.factors.library import DIRECTION_LONG, FACTOR_LIBRARY, FactorSpec
from src.research.factors.panel import Panel, listing_age_days
from src.research.factors.portfolio import MIN_NAMES_PER_QUANTILE

# 「低波动只是老币的代理」这个说法要直接检验：把上市天数也当成一个因子
AGE_FACTOR_KEY: Final[str] = "age_days"
# 参与正交化的因子：库里的因子 + 上市天数（方向 +1 表示「越老越强」是待检验的假设）
ORTHOGONAL_FACTORS: Final[dict[str, FactorSpec]] = {
    **FACTOR_LIBRARY,
    AGE_FACTOR_KEY: FactorSpec(
        key=AGE_FACTOR_KEY,
        label="上市天数",
        family="日历",
        direction=DIRECTION_LONG,
        warmup=0,
        compute=listing_age_days,
    ),
}
# 检验的因子对：(目标, 控制)。两条方向都要跑——只跑一条会漏掉「谁是谁的代理」的答案
ORTHOGONAL_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("vol_30", "liq_30"),
    ("liq_30", "vol_30"),
    ("vol_30", AGE_FACTOR_KEY),
    (AGE_FACTOR_KEY, "vol_30"),
)
# 双重排序的档数：与分位组合同为 5 档，读者不必在两套档位之间换算
ORTHOGONAL_BUCKETS: Final[int] = 5


@dataclass(frozen=True)
class OrthogonalIC:
    """控制 ``control`` 之后，``target`` 还剩多少 IC。"""

    target: str
    control: str
    horizon: int
    raw_ic: float
    raw_t: float
    control_ic: float
    residual_ic: float
    residual_t: float
    periods: int
    independent: int

    @property
    def decay(self) -> float:
        """控制后保留的比例：1 = 完全独立，0 = 只是控制变量的代理。"""
        if self.raw_ic == 0.0:
            return float("nan")
        return self.residual_ic / self.raw_ic


def orthogonal_ic(
    target: FactorSpec,
    control: FactorSpec,
    panel: Panel,
    mask: pd.DataFrame,
    horizon: int,
) -> OrthogonalIC:
    """残差 IC：目标秩对控制秩回归后，残差与前瞻收益的 Spearman。"""
    forward = forward_returns(panel, horizon)
    residual = residual_ic_series(
        target.values(panel).where(mask).rank(axis=1, pct=True),
        control.values(panel).where(mask).rank(axis=1, pct=True),
        forward.where(mask),
    )
    raw = ic_stats(target, panel, mask, horizon)
    control_stats = ic_stats(control, panel, mask, horizon)
    residual_t, independent = non_overlapping_t(residual, horizon)
    return OrthogonalIC(
        target=target.key,
        control=control.key,
        horizon=horizon,
        raw_ic=raw.mean,
        raw_t=raw.t_stat,
        control_ic=control_stats.mean,
        residual_ic=float(residual.mean()) if not residual.empty else 0.0,
        residual_t=residual_t,
        periods=int(residual.size),
        independent=independent,
    )


def double_sort_ic(
    target: FactorSpec,
    control: FactorSpec,
    panel: Panel,
    mask: pd.DataFrame,
    horizon: int,
    *,
    buckets: int = ORTHOGONAL_BUCKETS,
) -> pd.DataFrame:
    """双重排序：按控制因子分档后，各档内部的目标因子 IC。

    分档用控制因子的**原始值**（未按方向调整），所以第 1 档永远是「控制变量最小」
    那一档——方向调整过的值会让「Q1 是高波动还是低波动」变成需要查定义的问题。
    只在每一档都装得下 ``MIN_NAMES_PER_QUANTILE`` 只标的的期次参与统计。
    """
    forward = forward_returns(panel, horizon).where(mask)
    target_values = target.values(panel).where(mask).to_numpy(dtype=float)
    control_raw = control.compute(panel).where(mask).to_numpy(dtype=float)
    ahead = forward.to_numpy(dtype=float)
    ic_rows: list[dict[int, float]] = []
    median_rows: list[dict[int, float]] = []
    count_rows: list[dict[int, float]] = []
    for position in range(target_values.shape[0]):
        row = _bucket_rows(
            target_values[position], control_raw[position], ahead[position], buckets
        )
        if row is None:
            continue
        ic_row, median_row, count_row = row
        ic_rows.append(ic_row)
        median_rows.append(median_row)
        count_rows.append(count_row)
    if not ic_rows:
        return pd.DataFrame()
    ic_frame = pd.DataFrame(ic_rows)
    summary = pd.DataFrame(
        {
            "control_median": pd.DataFrame(median_rows).mean(),
            "ic": ic_frame.mean(),
            "t_stat": pd.Series(
                {
                    b: non_overlapping_t(cast(pd.Series, ic_frame[b]), horizon)[0]
                    for b in ic_frame.columns
                }
            ),
            "periods": ic_frame.count(),
            "names": pd.DataFrame(count_rows).mean(),
        }
    )
    summary.index.name = "bucket"
    return summary


def orthogonal_frame(
    panel: Panel, mask: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """正交化表：每个 (目标, 控制, horizon) 一行。"""
    rows: list[dict[str, float | int | str]] = []
    for target_key, control_key in ORTHOGONAL_PAIRS:
        for horizon in horizons:
            result = orthogonal_ic(
                ORTHOGONAL_FACTORS[target_key],
                ORTHOGONAL_FACTORS[control_key],
                panel,
                mask,
                horizon,
            )
            rows.append(
                {
                    "target": result.target,
                    "control": result.control,
                    "horizon": horizon,
                    "raw_ic": result.raw_ic,
                    "raw_t": result.raw_t,
                    "control_ic": result.control_ic,
                    "residual_ic": result.residual_ic,
                    "residual_t": result.residual_t,
                    "decay": result.decay,
                    "periods": result.periods,
                    "independent": result.independent,
                }
            )
    return pd.DataFrame(rows)


def double_sort_frame(
    panel: Panel, mask: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """双重排序表：每个 (目标, 控制, horizon, 档) 一行。"""
    frames: list[pd.DataFrame] = []
    for target_key, control_key in ORTHOGONAL_PAIRS:
        for horizon in horizons:
            table = double_sort_ic(
                ORTHOGONAL_FACTORS[target_key],
                ORTHOGONAL_FACTORS[control_key],
                panel,
                mask,
                horizon,
            )
            if table.empty:
                continue
            frames.append(
                table.assign(
                    target=target_key, control=control_key, horizon=horizon
                ).reset_index()
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bucket_rows(
    target_row: np.ndarray,
    control_row: np.ndarray,
    ahead_row: np.ndarray,
    buckets: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]] | None:
    """一期的分档结果（档 → IC / 控制变量中位 / 标的数）；任何一档太窄就整期丢弃。"""
    valid = np.isfinite(target_row) & np.isfinite(control_row) & np.isfinite(ahead_row)
    if int(valid.sum()) < buckets * MIN_NAMES_PER_QUANTILE:
        return None
    target = target_row[valid]
    control = control_row[valid]
    ahead = ahead_row[valid]
    order = pd.Series(control).rank(method="first").to_numpy(dtype=float)
    labels = np.clip(np.ceil(order / order.size * buckets).astype(int), 1, buckets)
    ic_row: dict[int, float] = {}
    median_row: dict[int, float] = {}
    count_row: dict[int, float] = {}
    for bucket in range(1, buckets + 1):
        selected = labels == bucket
        if int(selected.sum()) < MIN_NAMES_PER_QUANTILE:
            return None
        ic_row[bucket] = spearman(target[selected], ahead[selected])
        median_row[bucket] = float(np.median(control[selected]))
        count_row[bucket] = float(selected.sum())
    return ic_row, median_row, count_row
