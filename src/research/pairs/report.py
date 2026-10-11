"""
src/research/pairs/report.py
配对研究的表格层：把逐日贡献折成「能下判断」的五张表。

表的结构对应五个问题，顺序不能颠倒：

1. **组合 vs 基准**：市场中性策略唯一的卖点是「不靠 beta」，所以必须先跟 BTC
   买入持有、等权只多两条腿同起点比；
2. **单笔分布与集中度**：配对收益是右偏厚尾的，只看均值会把「一笔跳空」当成
   「稳定的边际」；
3. **成本敏感性**：两腿交易的成本是只多腿的两倍，这条曲线决定策略能不能活；
4. **分段稳定性**：段与段之间翻转 = 样本内挑参数；
5. **随机对照**：同池随机选对（不查 ADF、不查半衰期）跑同样的规则，用来回答
   「这套收益是协整筛选带来的，还是随便挑两只能涨的币都会有的」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.analytics.metrics import bars_per_year
from src.research.factors.panel import Panel
from src.research.factors.portfolio import LegStats, leg_stats
from src.research.pairs.backtest import PairsOptions, PairsRun, run_pairs

# 与 factors 同口径：任何「跑赢市场」的说法都要先过 BTC 这一关
BENCHMARK_SYMBOL: str = "BTC-USDT"
# 成本敏感性档位（单边，含滑点）：0 是上界，10bp 是本库默认的永续吃价口径
COST_LEVELS: tuple[float, ...] = (0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005)
SEGMENTS: int = 4


@dataclass(frozen=True)
class PairsReport:
    """五张表 + 口径标签。"""

    headline: pd.DataFrame
    trade_stats: pd.Series
    cost: pd.DataFrame
    segments: pd.DataFrame
    control: pd.DataFrame
    options_label: str
    funding_label: str


def build_report(
    panel: Panel,
    run: PairsRun,
    mask: pd.DataFrame,
    options: PairsOptions,
    funding_label: str = "未计资金费",
    controls: int = 0,
) -> PairsReport:
    """把一次运行折成五张表；``controls`` > 0 时额外跑随机对照。"""
    index = pd.DatetimeIndex(run.daily.index)
    if index.empty:
        raise ValueError("组合没有产生任何交易日：先确认选对条件与面板长度")
    returns = panel.close.pct_change(fill_method=None)
    btc = cast(pd.Series, returns[BENCHMARK_SYMBOL].reindex(index))
    market = cast(pd.Series, returns.where(mask).reindex(index).mean(axis=1))
    return PairsReport(
        headline=_headline(run.daily, btc, market),
        trade_stats=_trade_stats(run, options),
        cost=_cost_table(run),
        segments=_segments(run.daily),
        control=(
            _control_table(panel, run, options, funding_label, controls)
            if controls > 0
            else pd.DataFrame()
        ),
        options_label=options.describe(),
        funding_label=funding_label,
    )


def _leg(returns: pd.Series) -> LegStats:
    """一条收益序列的复利口径（每年根数按区间实际跨度算）。"""
    clean = returns.dropna()
    if clean.size < 2:
        return LegStats(0.0, 0.0, 0.0, 0.0)
    return leg_stats(clean, bars_per_year(clean.index))


def _beta_corr(returns: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """与基准的 beta 与相关（只用重叠日；缺失日当 0 收益会稀释 beta）。"""
    frame = pd.concat([returns, bench], axis=1, keys=["s", "b"]).dropna()
    if frame.shape[0] < 3:
        return float("nan"), float("nan")
    series = cast(pd.Series, frame["s"])
    benchmark = cast(pd.Series, frame["b"])
    variance = float(benchmark.var(ddof=1))
    if not np.isfinite(variance) or variance == 0.0:
        return float("nan"), float("nan")
    return float(series.cov(benchmark) / variance), float(series.corr(benchmark))


def _headline(daily: pd.Series, btc: pd.Series, market: pd.Series) -> pd.DataFrame:
    """组合与两条基准腿的对照表。"""
    rows: dict[str, dict[str, float]] = {}
    for label, series in (
        ("配对组合", daily),
        ("BTC 买入持有", btc),
        ("等权只多", market),
    ):
        stats = _leg(series)
        beta, corr = _beta_corr(daily, series)
        rows[label] = {
            "复利年化": stats.annual_compounded,
            "年化波动": stats.volatility,
            "夏普": stats.sharpe,
            "最大回撤": stats.max_drawdown,
            "累计收益": float((1.0 + series.dropna()).prod() - 1.0),
            "与组合beta": beta,
            "与组合相关": corr,
        }
    return pd.DataFrame(rows).T


def _trade_stats(run: PairsRun, options: PairsOptions) -> pd.Series:
    """单笔分布与集中度：配对收益是厚尾的，这一行是主表之外的必读。"""
    trades = run.trades
    net = run.daily
    if trades.empty:
        return pd.Series(dtype=float)
    top5 = float(cast(pd.Series, trades["net_total"]).nlargest(5).sum())
    return pd.Series(
        {
            "交易期数": float(run.windows),
            "平均对数": run.pairs_per_window,
            "配对笔数": float(len(trades)),
            "成交次数": float(run.rebalances),
            "持仓天数占比": float(
                trades["bars_held"].sum() / (options.trading_bars * len(trades))
            ),
            "止损比例": float(trades["stopped"].mean()),
            "中位单笔": float(trades["net_total"].median()),
            "均值单笔": float(trades["net_total"].mean()),
            "前5笔合计": float(top5),
            "最大单日": float(net.max()),
            "最差单日": float(net.min()),
            "平均gross暴露": float(run.exposure.mean()),
            "资金费标的覆盖": (
                float(run.funding_symbols / run.symbols) if run.symbols else 0.0
            ),
            "资金费日覆盖": run.funding_coverage,
        }
    )


def _cost_table(run: PairsRun) -> pd.DataFrame:
    """成本敏感性：换手已留痕，各档位不必重跑回测。"""
    turnover = (
        float(run.contributions["turnover"].sum())
        if not run.contributions.empty
        else 0.0
    )
    rows: dict[str, dict[str, float]] = {}
    for cost in COST_LEVELS:
        daily = run.at_cost(cost)
        stats = _leg(daily)
        rows[f"{cost * 1e4:.0f}bp"] = {
            "复利年化": stats.annual_compounded,
            "夏普": stats.sharpe,
            "最大回撤": stats.max_drawdown,
            "累计成本": turnover * cost,
        }
    return pd.DataFrame(rows).T


def _segments(daily: pd.Series, segments: int = SEGMENTS) -> pd.DataFrame:
    """分段稳定性：段与段之间翻转 = 样本内挑参数。"""
    clean = daily.dropna()
    if clean.size < segments * 2:
        return pd.DataFrame()
    rows: dict[str, dict[str, object]] = {}
    positions = np.arange(clean.size)
    for number, part_positions in enumerate(np.array_split(positions, segments), 1):
        part = clean.iloc[part_positions]
        stats = _leg(part)
        rows[f"段 {number}"] = {
            "起点": str(part.index.strftime("%Y-%m-%d")[0]),
            "复利年化": stats.annual_compounded,
            "夏普": stats.sharpe,
            "最大回撤": stats.max_drawdown,
        }
    return pd.DataFrame(rows).T


def _control_table(
    panel: Panel,
    run: PairsRun,
    options: PairsOptions,
    funding_label: str,
    controls: int,
) -> pd.DataFrame:
    """随机选对对照：同池、同数量、同规则，只把协整筛选换成随机挑。"""
    rows: list[dict[str, float]] = []
    for seed in range(controls):
        random_options = PairsOptions(
            selection=options.selection,
            trading_bars=options.trading_bars,
            step_bars=options.step_bars,
            entry_z=options.entry_z,
            exit_z=options.exit_z,
            stop_z=options.stop_z,
            cost_one_way=options.cost_one_way,
            random_seed=seed,
        )
        random_run = run_pairs(panel, random_options)
        stats = _leg(random_run.daily)
        rows.append(
            {
                "复利年化": stats.annual_compounded,
                "夏普": stats.sharpe,
                "最大回撤": stats.max_drawdown,
            }
        )
    random_frame = pd.DataFrame(rows)
    real = _leg(run.daily)
    beats = float((random_frame["复利年化"] >= real.annual_compounded).mean())
    summary = {
        "真实选对": {
            "复利年化": real.annual_compounded,
            "夏普": real.sharpe,
            "最大回撤": real.max_drawdown,
        },
        "随机中位": cast(pd.Series, random_frame.median()).to_dict(),
        "随机最差": cast(pd.Series, random_frame.min()).to_dict(),
        "随机最好": cast(pd.Series, random_frame.max()).to_dict(),
        "随机跑赢比例": {
            "复利年化": beats,
            "夏普": float("nan"),
            "最大回撤": float("nan"),
        },
    }
    frame = pd.DataFrame(summary).T
    frame.attrs["funding_label"] = funding_label
    return frame
