"""
src/research/factors/diagnostics.py
因子台的旁证：因子间相关、组合容量、幸存者偏差代理。

这三项都不改变「因子是否有效」的结论，但决定结论能不能用：

- **相关**：两个 IC 都高的因子若是同一条信号，组合并不比单因子好，只是多付一遍成本；
- **容量**：长尾标的上的因子收益，常常在能下单的规模上就消失了（滑点吃掉价差）；
- **幸存者偏差**：缓存里 0 个退市标的（实测），任何「小市值/低流动性溢价」都要先过这一关。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import pandas as pd

from src.research.factors.evaluation import forward_returns
from src.research.factors.library import FactorSpec
from src.research.factors.panel import (
    LIQUIDITY_WINDOW,
    Panel,
    UniverseRule,
    listing_age_days,
    tradable_mask,
)
from src.research.factors.portfolio import MIN_NAMES_PER_QUANTILE

# 单标的能吃掉的最大比例：日名义额的一定比例，超过就会自己推动价格
MAX_PARTICIPATION: Final[float] = 0.01
# 组合两条腿：多头与空头各占一半资金
LEG_COUNT: Final[int] = 2
# 容量按这么多期抽样估算（全样本逐期算没有必要，最挤的那一期决定容量）
REBALANCE_SAMPLES: Final[int] = 40
# 上市不足这么多天的观测用来当「生命周期早期」代理
YOUNG_DAYS: Final[int] = 180


@dataclass(frozen=True)
class SurvivorshipProbe:
    """幸存者偏差代理：上市早期的收益 vs 成熟期（描述性，不做显著性）。"""

    horizon: int
    young_days: int
    young_mean: float
    mature_mean: float
    young_count: int
    mature_count: int

    @property
    def gap(self) -> float:
        """早期收益减成熟期收益；为负说明死亡集中在生命周期早期。"""
        return self.young_mean - self.mature_mean


def factor_correlation(
    specs: list[FactorSpec], panel: Panel, mask: pd.DataFrame
) -> pd.DataFrame:
    """因子两两的截面 rank 相关（逐期算、再对期次取平均）。"""
    ranked = {
        spec.key: spec.values(panel).where(mask).rank(axis=1, pct=True)
        for spec in specs
    }
    keys = [spec.key for spec in specs]
    labels = pd.Index(keys)
    matrix = pd.DataFrame(1.0, index=labels, columns=labels)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            corr = ranked[left].corrwith(ranked[right], axis=1, method="pearson")
            value = float(corr.mean()) if corr.notna().any() else 0.0
            matrix.iloc[i, matrix.columns.get_loc(right)] = value
            matrix.iloc[matrix.index.get_loc(right), i] = value
    return matrix


def capacity_usd(
    spec: FactorSpec,
    panel: Panel,
    mask: pd.DataFrame,
    *,
    quantiles: int = 5,
    max_participation: float = MAX_PARTICIPATION,
) -> float:
    """组合可容纳资金：每条腿每个标的不超过其日名义额的一个比例。

    计算式：AUM ≤ 2 × 每腿标的数 × min(参与率 × 该标的的中位日名义额)，取所有
    调仓期的最小值。持有集合里流动性最差的那只决定容量（等权组合的最保守口径）。

    流动性必须用与池子规则同一把尺子（滚动中位名义额），不能取当日名义额：
    偶发的低成交日会把容量估低一个数量级（实测同一组合 6.4k → 160k）。
    """
    values = spec.values(panel).where(mask)
    median_notional = (
        panel.notional.where(mask)
        .rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW)
        .median()
    )
    step = max(len(panel.index) // REBALANCE_SAMPLES, 1)
    caps: list[float] = []
    for date in panel.index[spec.warmup :: step]:
        row = values.loc[date].dropna()
        if row.size < quantiles * MIN_NAMES_PER_QUANTILE:
            continue
        # 按因子值排序后取两头（等价于最高/最低分位），避开布尔掩码的类型摩擦
        order = row.sort_values().index
        names_per_leg = row.size // quantiles
        held = order[:names_per_leg].union(order[-names_per_leg:])
        liquidity = median_notional.loc[date].reindex(held).dropna()
        if liquidity.empty:
            continue
        caps.append(
            float(liquidity.min()) * max_participation * names_per_leg * LEG_COUNT
        )
    return min(caps) if caps else 0.0


def survivorship_probe(
    panel: Panel,
    rule: UniverseRule,
    horizon: int,
    *,
    young_days: int = YOUNG_DAYS,
) -> SurvivorshipProbe:
    """把「流动性达标但上市不足 young_days」的观测当幸存者偏差的**标志**。

    注意这个指标不能证明方向：缓存里 0 个退市标的（实测），所以它算出来的永远是
    *幸存者* 的上市初期收益。它的用途是定位「选择差异最大的阶段」——退市标的必然
    也集中在生命周期早期，而它们在缓存里不存在，因此该阶段的真实均值不可知，
    任何依赖上市初期收益的因子结论都不能用这个数字支持。
    """
    forward = forward_returns(panel, horizon)
    age = listing_age_days(panel)
    liquidity_ok = tradable_mask(panel, replace(rule, min_listed_days=0))
    young = liquidity_ok & (age < young_days)
    mature = liquidity_ok & (age >= young_days)

    return SurvivorshipProbe(
        horizon=horizon,
        young_days=young_days,
        young_mean=_mean_of(forward, young),
        mature_mean=_mean_of(forward, mature),
        young_count=int((forward.notna() & young).sum().sum()),
        mature_count=int((forward.notna() & mature).sum().sum()),
    )


def _mean_of(forward: pd.DataFrame, mask: pd.DataFrame) -> float:
    """掩码内的前向收益均值（观测级，不是逐期截面均值）。"""
    selected = forward.where(mask).stack()
    return float(selected.mean()) if not selected.empty else 0.0
