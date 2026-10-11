"""
src/research/factors/portfolio.py
分位组合的构建与统计：分位收益、多空价差、只多腿、换手、成本与复利口径。

读结论前必须先看这五条口径：

1. **名义基准**：spread 是多空各 1 单位名义的价差（gross exposure = 2），年化按
   期数简单倍数折算（不是可复利的资金曲线）。成本 = 4 × 换手率 × 单边费率：
   每条腿卖掉换掉的 ρ 份再买入 ρ 份，两条腿共 4ρ 次单边成交。
2. **spread 不是可执行的策略**：做空腿要求能做空（永续 + 资金费 + 强平），所以
   同时给出只多腿口径（long_only_net / long_only_annual，成本 2ρ）。两条腿的换手
   必须分开记（long_turnover / short_turnover）——多空价差用两者均值，而只多腿
   只能用它自己那一份。
3. **算术年化不是收益，复利年化才是**：*_annual 是「期均值 × 每年期数」的算术
   折算，在年化波动 95% 的腿上它比实际到手高出十几个百分点（波动拖累）。报收益
   必须用 long_leg / market_leg 的 annual_compounded，算术值只用于与行业惯例对齐
   的夏普。
4. **必须跟等权全体比**：只多腿的绝对收益里含市场 beta（实测 0.85），不减基准就
   分不清收益来自因子还是来自市场。market_leg 是每期「等权全体」基准，
   excess_compounded 是只多腿相对它的复利超额。
5. **跑赢等权还不够**：等权基准会变（池子每天不同），BTC 不会。池子跑赢等权但
   跑输 BTC 是常态（实测六个因子全部跑输），不报 btc_leg 就会把「在山寨币里选
   大的」误读成「值得做」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.evaluation import forward_returns
from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel

# 单边成交成本（默认口径）：截面调仓是「换标的」，按永续市价单（吃价）计费 + 滑点。
# 注意这是对主池的乐观假设：日均 $1M 量级的标的实际滑点大于 costs.py 的 5bp 默认值。
DEFAULT_ONE_WAY_COST: Final[float] = PERP_FEES.taker_net()

# 分位组合里每个分位至少要有的标的数
MIN_NAMES_PER_QUANTILE: Final[int] = 3
# 买入持有基准：任何「跑赢市场」的说法都要先过它这一关
BENCHMARK_SYMBOL: Final[str] = "BTC-USDT"
# 单条腿换掉 ρ 份要卖 ρ 再买 ρ（2ρ 次单边成交），两条腿合计 4ρ
_LEG_TRADES: Final[int] = 2
_REBALANCE_TRADES: Final[int] = _LEG_TRADES * 2


@dataclass(frozen=True)
class LegStats:
    """一条腿的复利口径。"""

    annual_compounded: float
    volatility: float
    sharpe: float
    max_drawdown: float


def leg_stats(returns: pd.Series, periods_per_year: float) -> LegStats:
    """把每期收益序列折成复利年化、年化波动、夏普与最大回撤。

    复利年化用几何平均（真正拿到手的），夏普用算术均值（行业惯例）：二者的差
    就是波动拖累，在年化波动 95% 的腿上这个差可以到十几个百分点，只报算术年化
    会把一条腿说得比实际好得多。
    """
    if returns.size < 2 or periods_per_year <= 0:
        return LegStats(0.0, 0.0, 0.0, 0.0)
    years = returns.size / periods_per_year
    growth = float((1.0 + returns).prod())
    annual = growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0
    volatility = float(returns.std(ddof=1)) * periods_per_year**0.5
    sharpe = (
        float(returns.mean()) * periods_per_year / volatility if volatility > 0 else 0.0
    )
    curve = (1.0 + returns).cumprod()
    return LegStats(
        annual_compounded=annual,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=float((curve / curve.cummax() - 1.0).min()),
    )


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
    # 每期收益序列（只多腿与「等权全体」基准）：算术均值既看不出波动拖累，也
    # 看不出相对基准的超额，原始序列必须留着
    long_returns: pd.Series
    market_returns: pd.Series
    # BTC 买入持有（同期同起点）：比等权基准更硬的一道关，跑不赢它就没有价值
    btc_leg: LegStats

    @property
    def periods_per_year(self) -> float:
        return self.periods / self.span_years if self.span_years > 0 else 0.0

    @property
    def long_leg(self) -> LegStats:
        """只多腿（最高分位）的复利口径。"""
        return leg_stats(self.long_returns, self.periods_per_year)

    @property
    def market_leg(self) -> LegStats:
        """市场基准（每期等权全体）的复利口径。"""
        return leg_stats(self.market_returns, self.periods_per_year)

    @property
    def excess_compounded(self) -> float:
        """只多腿相对等权基准的复利年化超额。"""
        return self.long_leg.annual_compounded - self.market_leg.annual_compounded

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

    @property
    def break_even_cost(self) -> float:
        """盈亏平衡成本（单边）：把毛价差吃到零的那个费率。

        与实测成本并排看才知道一个因子是「信号不行」还是「成本不允许」：
        平衡成本远高于 10bp（taker）说明还有余地，低于 maker 费率（2bp）说明
        这个频率下根本做不了。毛价差为负时返回 0（无成本也不赚，谈不上平衡点）。
        """
        if self.turnover <= 0.0 or self.gross_spread <= 0.0:
            return 0.0
        return self.gross_spread / (_REBALANCE_TRADES * self.turnover)


def _buy_and_hold_leg(panel: Panel, first_pos: int, last_pos: int) -> LegStats:
    """BTC 买入持有（与组合同期同起点）的复利口径。

    等权基准会变（池子每天不同），BTC 不会；池子跑赢等权但跑输 BTC 是常态，
    不报这一行就会把「在山寨币里选大的」误读成「值得做」。
    """
    if BENCHMARK_SYMBOL not in panel.close.columns:
        return LegStats(0.0, 0.0, 0.0, 0.0)
    window = panel.close[BENCHMARK_SYMBOL].iloc[first_pos : last_pos + 1].dropna()
    if window.size < 2:
        return LegStats(0.0, 0.0, 0.0, 0.0)
    span_years = max(
        (window.index[-1] - window.index[0]).total_seconds() / (365.25 * 86400.0), 1e-9
    )
    daily = window.pct_change().dropna()
    return leg_stats(daily, daily.size / span_years)


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
    market: list[float] = []
    top_holdings: list[set[str]] = []
    bottom_holdings: list[set[str]] = []
    kept_dates: list[pd.Timestamp] = []
    for date in dates:
        row = values.loc[date].dropna()
        if row.size < options.quantiles * MIN_NAMES_PER_QUANTILE:
            continue
        bucket = pd.qcut(row.rank(method="first"), options.quantiles, labels=False)
        names = row.index
        leg = forward.loc[date].reindex(names)
        per_date.append(leg.groupby(bucket).mean())
        # 等权全体：只多腿要跟它比，才知道超额来自因子还是来自市场
        market.append(float(leg.mean()))
        kept_dates.append(cast(pd.Timestamp, date))
        top_holdings.append(set(names[bucket == options.quantiles - 1]))
        bottom_holdings.append(set(names[bucket == 0]))

    if not per_date:
        empty = pd.Series(dtype=float)
        return PortfolioStats(
            factor=spec.key,
            horizon=options.horizon,
            quantiles=options.quantiles,
            periods=0,
            quantile_returns=empty,
            gross_spread=0.0,
            net_spread=0.0,
            turnover=0.0,
            long_turnover=0.0,
            short_turnover=0.0,
            cost_one_way=options.cost_one_way,
            annual_gross=0.0,
            annual_net=0.0,
            span_years=0.0,
            long_returns=empty,
            market_returns=empty,
            btc_leg=LegStats(0.0, 0.0, 0.0, 0.0),
        )

    # 行索引必须显式给：groupby().mean() 会把 date 当 name 带下去，靠它对齐太脆
    index = pd.DatetimeIndex(kept_dates)
    raw = pd.DataFrame(per_date, index=index)
    market_returns = pd.Series(market, index=index, dtype=float)
    valid = raw.notna().all(axis=1)
    quantile_returns = raw[valid]
    market_returns = cast(pd.Series, market_returns[valid])
    # stub 把 DataFrame.mean() 标成 Series | float | int，运行时按列归约必为 Series
    average = cast(pd.Series, quantile_returns.mean())
    gross = float(average.iloc[-1] - average.iloc[0])
    long_turnover, short_turnover = _leg_turnovers(top_holdings, bottom_holdings)
    turnover = (long_turnover + short_turnover) / 2
    cost = _REBALANCE_TRADES * turnover * options.cost_one_way
    net = gross - cost
    # 组合的实际投资期 = 首期建仓 → 末期持有结束（末期还要再持 h 根），不是面板
    # 跨度：面板从 2018 年起，但池子要到 2020 年才凑够 15 只可交易标的
    positions = panel.index.get_indexer(quantile_returns.index)
    first_pos = int(positions[0])
    last_pos = min(int(positions[-1]) + options.horizon, len(panel.index) - 1)
    span_years = max(
        (panel.index[last_pos] - panel.index[first_pos]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    periods_per_year = quantile_returns.shape[0] / span_years
    btc_leg = _buy_and_hold_leg(panel, first_pos, last_pos)

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
        long_returns=cast(pd.Series, quantile_returns.iloc[:, -1]),
        market_returns=market_returns,
        btc_leg=btc_leg,
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
