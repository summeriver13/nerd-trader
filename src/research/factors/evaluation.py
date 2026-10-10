"""
src/research/factors/evaluation.py
截面因子的评估台：前向收益、RankIC、分位组合、换手与成本。

读结论前必须先看这三条口径：

1. **重叠样本**：日频 IC 是「每天算一次 h 根前向收益」，相邻期高度重叠。
   本模块只输出非重叠子样本的 t 值——重叠样本的 t 会把有效观测数放大 h 倍，
   实测（2026-10-09 主池）同一因子的重叠 t 是独立 t 的 √h 倍，足以把噪声读成显著。
2. **非重叠组合**：分位组合每 h 根调仓一次，收益、换手、成本都在同一套期次上算。
   组合口径的期数远少于 IC 口径，两者不一致时以组合口径为准。
3. **名义基准**：spread 是多空各 1 单位名义的价差（gross exposure = 2），年化按
   期数简单倍数折算（不是可复利的资金曲线）。成本 = 4 × 换手率 × 单边费率：
   每条腿卖掉换掉的 ρ 份再买入 ρ 份，两条腿共 4ρ 次单边成交。

   但 **spread 不是可执行的策略**：做空腿要求能做空（永续 + 资金费 + 强平），
   所以同时给出只多腿口径（long_only_net / long_only_annual，成本 2ρ）。
   两条腿的换手必须分开记（long_turnover / short_turnover）——多空价差用的是两者
   均值，而只多腿只能用它自己那一份。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel

# 单边成交成本（默认口径）：截面调仓是「换标的」，按永续市价单（吃价）计费 + 滑点。
# 注意这是对主池的乐观假设：日均 $1M 量级的标的实际滑点大于 costs.py 的 5bp 默认值。
DEFAULT_ONE_WAY_COST: Final[float] = PERP_FEES.taker_net()

# 截面太窄时 RankIC 没有意义：少于这么多标的的期次直接跳过，不参与统计
MIN_NAMES_PER_PERIOD: Final[int] = 10
# 分位组合里每个分位至少要有的标的数
MIN_NAMES_PER_QUANTILE: Final[int] = 3
# 单条腿换掉 ρ 份要卖 ρ 再买 ρ（2ρ 次单边成交），两条腿合计 4ρ
_LEG_TRADES: Final[int] = 2
_REBALANCE_TRADES: Final[int] = _LEG_TRADES * 2


@dataclass(frozen=True)
class ICStats:
    """一个因子在一个前向窗口上的 RankIC 统计。"""

    factor: str
    horizon: int
    mean: float
    std: float
    t_stat: float
    mean_independent: float
    periods: int
    independent: int
    names: float


@dataclass(frozen=True)
class PortfolioOptions:
    """分位组合的口径。"""

    horizon: int = 30
    quantiles: int = 5
    cost_one_way: float = DEFAULT_ONE_WAY_COST

    @property
    def cost_label(self) -> str:
        return f"{self.cost_one_way * 1e4:.1f}bp/边"


@dataclass(frozen=True)
class PortfolioStats:
    """分位组合的统计（非重叠期次，名义基准）。"""

    factor: str
    horizon: int
    quantiles: int
    periods: int
    quantile_returns: pd.Series
    gross_spread: float
    net_spread: float
    # 两条腿的平均换手（多空价差的成本口径）
    turnover: float
    # 两条腿各自的换手：只多腿是可执行口径，成本只能用它自己那一份
    long_turnover: float
    short_turnover: float
    cost_one_way: float
    annual_gross: float
    annual_net: float
    span_years: float

    @property
    def long_only_net(self) -> float:
        """只多腿净收益：最高分位毛收益 − 该腿自己的换手成本（2ρ）。"""
        if self.quantile_returns.empty:
            return 0.0
        return float(self.quantile_returns.iloc[-1]) - (
            _LEG_TRADES * self.long_turnover * self.cost_one_way
        )

    @property
    def long_only_annual(self) -> float:
        """只多腿年化：与 annual_net 同一套期数折算，便于并列比较。"""
        return self.long_only_net * self.periods / self.span_years


def forward_returns(panel: Panel, horizon: int) -> pd.DataFrame:
    """h 根前向收益：全项目唯一允许「看未来」的一行（它是被解释变量，不是因子）。"""
    return panel.close.shift(-horizon) / panel.close - 1


def rank_ic_series(
    values: pd.DataFrame, forward: pd.DataFrame, mask: pd.DataFrame
) -> pd.Series:
    """逐期截面 Spearman 相关，只在可交易集合内排名。

    两类期次直接跳过：标的太少、以及因子值在截面上完全相同的期（相关系数无定义，
    算出来是 NaN 并会让 pandas 抛 ConstantInputWarning）。筛选要在算相关之前做，
    否则警告照报。
    """
    masked_values = values.where(mask)
    masked_forward = forward.where(mask)
    valid = (masked_values.notna().sum(axis=1) >= MIN_NAMES_PER_PERIOD) & (
        masked_values.nunique(axis=1) >= MIN_NAMES_PER_PERIOD
    )
    if not valid.any():
        return pd.Series(dtype=float)
    ic = masked_values[valid].corrwith(masked_forward[valid], axis=1, method="spearman")
    return ic.dropna()


def ic_stats(
    spec: FactorSpec, panel: Panel, mask: pd.DataFrame, horizon: int
) -> ICStats:
    """因子 IC 统计：点估计用重叠样本（更平滑），显著性只认非重叠子样本。"""
    ic = rank_ic_series(spec.values(panel), forward_returns(panel, horizon), mask)
    if ic.empty:
        return ICStats(spec.key, horizon, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    independent = ic.iloc[::horizon]
    std = float(independent.std(ddof=1)) if independent.size > 1 else 0.0
    t_stat = (
        float(independent.mean() / (std / independent.size**0.5)) if std > 0 else 0.0
    )
    return ICStats(
        factor=spec.key,
        horizon=horizon,
        mean=float(ic.mean()),
        std=float(ic.std(ddof=1)),
        t_stat=t_stat,
        mean_independent=float(independent.mean()),
        periods=int(ic.size),
        independent=int(independent.size),
        names=float(
            spec.values(panel).where(mask).notna().sum(axis=1).reindex(ic.index).mean()
        ),
    )


def portfolio_stats(
    spec: FactorSpec,
    panel: Panel,
    mask: pd.DataFrame,
    options: PortfolioOptions,
) -> PortfolioStats:
    """分位组合统计：非重叠期次的等权分位收益、多空价差、换手与净收益。"""
    values = spec.values(panel).where(mask)
    forward = forward_returns(panel, options.horizon).where(mask)
    dates = panel.index[spec.warmup :: options.horizon]

    per_date: list[pd.Series] = []
    top_holdings: list[set[str]] = []
    bottom_holdings: list[set[str]] = []
    for date in dates:
        row = values.loc[date].dropna()
        if row.size < options.quantiles * MIN_NAMES_PER_QUANTILE:
            continue
        bucket = pd.qcut(row.rank(method="first"), options.quantiles, labels=False)
        names = row.index
        per_date.append(forward.loc[date].reindex(names).groupby(bucket).mean())
        top_holdings.append(set(names[bucket == options.quantiles - 1]))
        bottom_holdings.append(set(names[bucket == 0]))

    if not per_date:
        empty = pd.Series(dtype=float)
        return PortfolioStats(
            spec.key,
            options.horizon,
            options.quantiles,
            0,
            empty,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            options.cost_one_way,
            0.0,
            0.0,
            0.0,
        )

    quantile_returns = pd.DataFrame(per_date).dropna(how="any")
    # stub 把 DataFrame.mean() 标成 Series | float | int，运行时按列归约必为 Series
    average = cast(pd.Series, quantile_returns.mean())
    gross = float(average.iloc[-1] - average.iloc[0])
    long_turnover, short_turnover = _leg_turnovers(top_holdings, bottom_holdings)
    turnover = (long_turnover + short_turnover) / 2
    cost = _REBALANCE_TRADES * turnover * options.cost_one_way
    net = gross - cost
    span_years = max((panel.index[-1] - panel.index[0]).days / 365.0, 1e-9)
    periods_per_year = quantile_returns.shape[0] / span_years

    return PortfolioStats(
        factor=spec.key,
        horizon=options.horizon,
        quantiles=options.quantiles,
        periods=int(quantile_returns.shape[0]),
        quantile_returns=average,
        gross_spread=gross,
        net_spread=net,
        turnover=turnover,
        long_turnover=long_turnover,
        short_turnover=short_turnover,
        cost_one_way=options.cost_one_way,
        annual_gross=gross * periods_per_year,
        annual_net=net * periods_per_year,
        span_years=span_years,
    )


def _leg_turnovers(
    top_holdings: list[set[str]], bottom_holdings: list[set[str]]
) -> tuple[float, float]:
    """两条腿各自的换手率（只多腿与只空腿分开，成本口径不同）。"""
    return _turnover_of(top_holdings), _turnover_of(bottom_holdings)


def _turnover_of(holdings: list[set[str]]) -> float:
    """一条腿的换手率：被替换掉的标的占该腿的比例。"""
    ratios = [
        1.0 - len(current & previous) / len(current)
        for previous, current in zip(holdings, holdings[1:], strict=False)
        if current
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0
