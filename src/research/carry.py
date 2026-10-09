"""
Delta-neutral carry 研究：资金费率的真实可获收益。

做多现货 + 做空等额永续，价格风险基本对冲，收益只剩资金费率本身。
报告给出毛年化，再依次扣除保证金占用与双腿换手成本，
回答「carry 现在还剩多少、值不值得占用资金」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.utils.data_sources.funding import FundingSource, create_funding_exchange

ANNUAL_DAYS = 365
ROLLING_DAYS = 30
SPOT_FEE = 0.0005
PERP_FEE = 0.0005
ROUND_TRIP_COST = 2 * (SPOT_FEE + PERP_FEE)  # 建仓 + 平仓，双腿各一次
REBALANCE_PER_YEAR = 12
REBALANCE_COST = REBALANCE_PER_YEAR * (SPOT_FEE + PERP_FEE) * 2
CAPITAL_MULTIPLE = 1.5  # 现货名义 1.0 + 永续保证金 0.5，收益率按此折算


@dataclass(frozen=True)
class YearStats:
    """单个自然年的资金费率收益。"""

    year: int
    settlements: int
    annual: float
    positive_share: float
    worst_rolling: float


@dataclass(frozen=True)
class CarryReport:
    """一个标的的 carry 全样本统计。"""

    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    settlements: int
    gross_annual: float
    capital_adjusted: float
    net_annual: float
    positive_share: float
    median_rate: float
    rolling_median: float
    rolling_p5: float
    rolling_worst: float
    rolling_best: float
    years: list[YearStats]


def daily_funding(funding: pd.Series) -> pd.Series:
    """把 8 小时结算费率按日求和（每天 3 次结算）。"""
    daily = funding.resample("1D").sum()
    return cast(pd.Series, daily[daily.index <= funding.index[-1]])


def annualize(returns: pd.Series, bars: int) -> float:
    """按 365 天复利折算年化。"""
    total = float((1 + returns).prod() - 1)
    if total <= -1 or bars <= 0:
        return -1.0
    return (1 + total) ** (ANNUAL_DAYS / bars) - 1


def rolling_annual(daily: pd.Series, window: int = ROLLING_DAYS) -> pd.Series:
    """滚动 window 天的复利年化。"""
    cumulative = cast(pd.Series, daily.rolling(window).sum()).dropna()
    cumulative = cumulative[cumulative > -1]
    return cast(pd.Series, (1 + cumulative) ** (ANNUAL_DAYS / window) - 1)


def _years(index: pd.Index) -> np.ndarray:
    """把时间索引转成年份数组，便于按年切片。"""
    return np.asarray(cast(pd.DatetimeIndex, index).year)


def yearly_stats(funding: pd.Series, daily: pd.Series) -> list[YearStats]:
    """逐年统计年化、正费率占比与最差滚动窗口。"""
    rolling = rolling_annual(daily)
    daily_year = _years(daily.index)
    funding_year = _years(funding.index)
    rolling_year = _years(rolling.index)
    out: list[YearStats] = []
    for year in sorted({int(value) for value in daily_year}):
        group = daily[daily_year == year]
        same_year = funding[funding_year == year]
        window = rolling[rolling_year == year]
        out.append(
            YearStats(
                year=year,
                settlements=int(same_year.size),
                annual=annualize(group, len(group)),
                positive_share=float((same_year > 0).mean()) if same_year.size else 0.0,
                worst_rolling=float(window.min()) if window.size else 0.0,
            )
        )
    return out


def build_carry_report(symbol: str, funding: pd.Series) -> CarryReport:
    """把资金费率序列整理成 carry 报告（含成本与资本占用折算）。"""
    daily = daily_funding(funding)
    rolling = rolling_annual(daily)
    gross = annualize(daily, len(daily))
    adjusted = gross / CAPITAL_MULTIPLE
    return CarryReport(
        symbol=symbol,
        start=cast(pd.Timestamp, funding.index[0]),
        end=cast(pd.Timestamp, funding.index[-1]),
        settlements=int(funding.size),
        gross_annual=gross,
        capital_adjusted=adjusted,
        net_annual=adjusted - ROUND_TRIP_COST - REBALANCE_COST,
        positive_share=float((funding > 0).mean()),
        median_rate=float(funding.median()),
        rolling_median=float(rolling.median()) if rolling.size else 0.0,
        rolling_p5=float(np.percentile(rolling, 5)) if rolling.size else 0.0,
        rolling_worst=float(rolling.min()) if rolling.size else 0.0,
        rolling_best=float(rolling.max()) if rolling.size else 0.0,
        years=yearly_stats(funding, daily),
    )


def print_carry_report(report: CarryReport) -> None:
    """打印单个标的的 carry 报告。"""
    print(f"\n{'=' * 118}")
    print(
        f"{report.symbol} | {report.start.date()} → {report.end.date()}"
        f" | {report.settlements} 次结算 | 正费率占比 {report.positive_share:.1%}"
        f" | 单次中位 {report.median_rate:.4%}"
    )
    print("=" * 118)
    print(
        f"毛年化（按现货名义本金）      : {report.gross_annual:>8.2%}"
        f"   （8 小时结算，复利）"
    )
    print(
        f"按资本占用 {CAPITAL_MULTIPLE:.1f} 倍折算     : {report.capital_adjusted:>8.2%}"
        f"   （现货 1.0 + 永续保证金 0.5）"
    )
    print(
        f"扣一次性进出 {ROUND_TRIP_COST:.2%}        : {report.capital_adjusted - ROUND_TRIP_COST:>8.2%}"
        f"   （建仓 + 平仓，双腿）"
    )
    print(
        f"再扣每月再平衡 {REBALANCE_COST:.2%}/年 : {report.net_annual:>8.2%}"
        f"   （每年 {REBALANCE_PER_YEAR} 次双腿调仓）"
    )
    print(
        f"{ROLLING_DAYS} 天滚动年化: 中位 {report.rolling_median:.2%}"
        f" | P5 {report.rolling_p5:.2%} | 最差 {report.rolling_worst:.2%}"
        f" | 最好 {report.rolling_best:.2%}"
    )

    print(f"\n逐年（{len(report.years)} 年）")
    print(f"{'年份':<8}{'结算次数':>10}{'年化':>10}{'正费率占比':>12}{'最差滚动':>12}")
    print("-" * 118)
    for stat in report.years:
        print(
            f"{stat.year:<8}{stat.settlements:>10}{stat.annual:>9.1%} "
            f"{stat.positive_share:>11.1%} {stat.worst_rolling:>11.1%}"
        )


def run_carry(symbols: list[str], days: int) -> None:
    """资金费率 carry 研究入口：拉取 Binance 资金费率并打印报告。"""
    source = FundingSource(create_funding_exchange())
    print(
        "\n资金费率 carry：做多现货 + 做空等额永续，价格风险对冲，收益只剩资金费率。"
    )
    for symbol in symbols:
        try:
            frame = source.load(symbol, days)
        except Exception as e:
            print(f"❌ {symbol} 资金费率获取失败: {e}")
            continue
        if frame.empty:
            print(f"❌ {symbol} 无资金费率数据")
            continue
        print_carry_report(build_carry_report(symbol, cast(pd.Series, frame["rate"])))
    print(
        "\n注：执行需要在同一账户同时持有现货多头与永续空头"
        "（OKX AU 需通过衍生品资格评估，零售杠杆上限 2x）。"
    )
