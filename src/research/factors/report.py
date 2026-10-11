"""
src/research/factors/report.py
因子台的报告层：把 IC、分位组合、相关、容量、幸存者偏差汇成可落盘的表。

五条口径，缺一条结论就不能用：

1. **显著性只认非重叠 t 值**，且必须过多重检验校正。因子库有 N 个因子 × H 个
   horizon，报「最好的那个」等于没校正；门槛用 Bonferroni 双侧临界值。
   注意 Bonferroni 假设检验独立，而量价因子彼此相关（mom 与 rev 相关 -0.54），
   所以它是保守下界——过了就是真过了，没过不代表一定无效。
2. **只多腿与多空价差并列**。spread 要能做空（永续 + 资金费 + 强平）才拿得到，
   只多腿才是可直接执行的口径；实测二者可以差一个量级，只报 spread 会误导。
3. **容量是前提不是结论**：截面因子的收益常常在能下单的规模上就没了。
4. **幸存者偏差是前提不是结论**：缓存里 0 个退市标的，任何依赖上市初期收益的
   数字都不能当证据。
5. **必须跟等权全体比，且报复利不报算术**：只多腿的绝对收益里含市场 beta，
   不减去基准就分不清收益来自因子还是来自市场；*_annual 是算术折算，在年化波动
   95% 的腿上它比实际到手高出十几个百分点。判定看 excess_compounded。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Final

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.factors.diagnostics import (
    capacity_usd,
    factor_correlation,
    survivorship_probe,
)
from src.research.factors.evaluation import ic_stats
from src.research.factors.library import FACTOR_LIBRARY, FactorSpec
from src.research.factors.orthogonal import double_sort_frame, orthogonal_frame
from src.research.factors.panel import (
    Panel,
    UniverseRule,
    tradable_counts,
    tradable_mask,
)
from src.research.factors.portfolio import (
    DEFAULT_ONE_WAY_COST,
    PortfolioOptions,
    PortfolioStats,
    portfolio_stats,
)

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "factors"
# IC 看三个尺度：1 根（日频反转/波动）、7 根（周）、30 根（月）
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 7, 30)
# 分位组合的主口径：与调仓周期一致，且期数够多（约 76 期）
DEFAULT_PORTFOLIO_HORIZON: Final[int] = 30
DEFAULT_QUANTILES: Final[int] = 5
ALPHA: Final[float] = 0.05


@dataclass(frozen=True)
class FactorReport:
    """一次完整评估的全部表格。"""

    rule_label: str
    # 逐日可交易标的数（面板宽度）：读任何截面结论前先看这条曲线
    universe: pd.DataFrame
    ic: pd.DataFrame
    portfolio: pd.DataFrame
    correlation: pd.DataFrame
    capacity: pd.DataFrame
    survivorship: pd.DataFrame
    # vol_30 与 liq_30 的正交化：唯一可能推翻「低波动异象」的检验
    orthogonal: pd.DataFrame
    double_sort: pd.DataFrame
    # 多重检验：试验次数 = 因子数 × horizon 数
    trials: int
    critical_t: float


def critical_t(trials: int, alpha: float = ALPHA) -> float:
    """Bonferroni 校正后的双侧临界 t 值（正态近似）。

    trials 的语义是「一共试了多少次」：因子数 × horizon 数，而不是只数因子——
    换 horizon 也是一次尝试，只算因子会把门槛放低。
    """
    return NormalDist().inv_cdf(1 - alpha / (2 * trials))


def build_report(
    panel: Panel,
    rule: UniverseRule,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    portfolio_horizon: int = DEFAULT_PORTFOLIO_HORIZON,
    quantiles: int = DEFAULT_QUANTILES,
    cost_one_way: float = DEFAULT_ONE_WAY_COST,
) -> FactorReport:
    """跑完一整套评估。只读面板与掩码，不触发任何采集。"""
    specs = list(FACTOR_LIBRARY.values())
    mask = tradable_mask(panel, rule)
    options = PortfolioOptions(
        horizon=portfolio_horizon, quantiles=quantiles, cost_one_way=cost_one_way
    )
    trials = len(specs) * len(horizons)
    threshold = critical_t(trials)
    ic = ic_frame(specs, panel, mask, horizons)
    ic["passes"] = ic["t_stat"].abs() > threshold

    return FactorReport(
        rule_label=rule.describe(),
        universe=tradable_counts(mask).rename("tradable").to_frame(),
        ic=ic,
        portfolio=portfolio_frame(specs, panel, mask, options),
        correlation=factor_correlation(specs, panel, mask),
        capacity=capacity_frame(specs, panel, mask, quantiles),
        survivorship=survivorship_frame(panel, rule, horizons),
        orthogonal=orthogonal_frame(panel, mask, horizons),
        double_sort=double_sort_frame(panel, mask, horizons),
        trials=trials,
        critical_t=threshold,
    )


def ic_frame(
    specs: list[FactorSpec],
    panel: Panel,
    mask: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """因子 × horizon 的 RankIC 表。t 值来自非重叠子样本，不是重叠样本。"""
    rows: list[dict[str, float | int | str]] = []
    for spec in specs:
        for horizon in horizons:
            stats = ic_stats(spec, panel, mask, horizon)
            rows.append(
                {
                    "factor": spec.key,
                    "label": spec.label,
                    "horizon": horizon,
                    "mean_ic": stats.mean,
                    "t_stat": stats.t_stat,
                    "mean_independent": stats.mean_independent,
                    "periods": stats.periods,
                    "independent": stats.independent,
                    "names": stats.names,
                }
            )
    return pd.DataFrame(rows)


def portfolio_frame(
    specs: list[FactorSpec],
    panel: Panel,
    mask: pd.DataFrame,
    options: PortfolioOptions,
) -> pd.DataFrame:
    """分位组合表：多空价差与只多腿并列，两条腿的换手分开列。"""
    return pd.DataFrame(
        [
            _portfolio_row(spec, portfolio_stats(spec, panel, mask, options))
            for spec in specs
        ]
    )


def _portfolio_row(
    spec: FactorSpec, stats: PortfolioStats
) -> dict[str, float | int | str]:
    """把 PortfolioStats 摊平成一行（含只多腿的可执行口径）。"""
    top = (
        float(stats.quantile_returns.iloc[-1])
        if not stats.quantile_returns.empty
        else 0.0
    )
    bottom = (
        float(stats.quantile_returns.iloc[0])
        if not stats.quantile_returns.empty
        else 0.0
    )
    long_leg = stats.long_leg
    market_leg = stats.market_leg
    return {
        "factor": spec.key,
        "label": spec.label,
        "horizon": stats.horizon,
        "quantiles": stats.quantiles,
        "periods": stats.periods,
        "span_years": stats.span_years,
        "top_leg": top,
        "bottom_leg": bottom,
        "gross_spread": stats.gross_spread,
        "net_spread": stats.net_spread,
        "annual_net": stats.annual_net,
        "long_only_net": stats.long_only_net,
        "long_only_annual": stats.long_only_annual,
        "turnover": stats.turnover,
        "long_turnover": stats.long_turnover,
        "cost_one_way": stats.cost_one_way,
        # 盈亏平衡成本（单边）：与实测成本并排看，才分得清「信号不行」与「成本不允许」
        "break_even_cost": stats.break_even_cost,
        # 分位收益与分位序号的秩相关：低单调性说明「只多腿好看」来自极端分位
        # 而非单调排序，这种收益换个起始点就消失
        "monotonicity": _monotonicity(stats.quantile_returns),
        # 只多腿的复利口径：算术年化会因波动拖累高估实际收益
        "long_compounded": long_leg.annual_compounded,
        "long_sharpe": long_leg.sharpe,
        "long_drawdown": long_leg.max_drawdown,
        # 市场基准（每期等权全体）：不跟它比就分不清因子与 beta
        "market_compounded": market_leg.annual_compounded,
        "market_sharpe": market_leg.sharpe,
        "market_drawdown": market_leg.max_drawdown,
        "excess_compounded": stats.excess_compounded,
        # BTC 买入持有（同期同起点）：比等权基准更硬的一道关
        "btc_compounded": stats.btc_leg.annual_compounded,
        "btc_sharpe": stats.btc_leg.sharpe,
        "btc_drawdown": stats.btc_leg.max_drawdown,
    }


def _monotonicity(quantile_returns: pd.Series) -> float:
    """分位收益对分位序号的 Spearman 相关：+1 完全单调，0 说明排序不携带信息。"""
    if quantile_returns.size < 2:
        return 0.0
    order = pd.Series(range(quantile_returns.size), index=quantile_returns.index)
    return float(order.corr(quantile_returns, method="spearman"))


def capacity_frame(
    specs: list[FactorSpec], panel: Panel, mask: pd.DataFrame, quantiles: int
) -> pd.DataFrame:
    """容量表：每个因子对应的多空组合能容纳多少资金（美元）。"""
    return pd.DataFrame(
        {
            "capacity_usd": [
                capacity_usd(spec, panel, mask, quantiles=quantiles) for spec in specs
            ],
        },
        index=pd.Index([spec.key for spec in specs], name="factor"),
    )


def survivorship_frame(
    panel: Panel, rule: UniverseRule, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """幸存者偏差代理：上市初期收益 vs 成熟期收益（描述性，不做显著性）。"""
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        probe = survivorship_probe(panel, rule, horizon)
        rows.append(
            {
                "horizon": horizon,
                "young_mean": probe.young_mean,
                "mature_mean": probe.mature_mean,
                "gap": probe.gap,
                "young_count": probe.young_count,
                "mature_count": probe.mature_count,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(report: FactorReport, out_dir: Path = OUTPUT_DIR) -> list[Path]:
    """把八张表写到 data/results/factors/，供下游组合分析复用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "factor_universe": report.universe,
        "factor_ic": report.ic,
        "factor_portfolio": report.portfolio,
        "factor_correlation": report.correlation,
        "factor_capacity": report.capacity,
        "factor_survivorship": report.survivorship,
        "factor_orthogonal": report.orthogonal,
        "factor_double_sort": report.double_sort,
    }
    paths: list[Path] = []
    for name, frame in tables.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path)
        paths.append(path)
    return paths
