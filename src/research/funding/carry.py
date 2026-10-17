"""
Delta-neutral carry 研究：资金费率的真实可获收益。

做多现货 + 做空等额永续，价格风险基本对冲，收益只剩资金费率本身。
报告给出毛年化，再按持有期摊薄双腿成本，回答「carry 现在还剩多少、值不值得占用资金」。

口径约定：
- 资金费按日求和（结算间隔由交易所决定，历史上出现过 2h/4h/8h，按日求和对此不敏感）；
- 毛年化按现货名义本金，成本按资本占用（现货 1.0 + 永续保证金 0.5）折算；
- 一次性进出费摊到持有年数，再平衡费按换手率而非全名义换手。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.analytics.metrics import ANNUAL_DAYS, annualize, total_return
from src.core.costs import PERP_TAKER_FEE, SPOT_TAKER_FEE
from src.utils.console import pad
from src.utils.data_sources.funding import FundingSource, daily_rates

ROLLING_DAYS = 30
# 双腿（现货 + 永续）× 建仓/平仓，各一次 taker；现货 taker 0.10% 高于永续 0.05%
ROUND_TRIP_COST = 2 * (SPOT_TAKER_FEE + PERP_TAKER_FEE)
REBALANCE_PER_YEAR = 12
TURNOVER_SHARE = 0.05  # 每次再平衡调掉的名义占比（现货与永续之间的漂移量）
TURNOVER_SHARE_STRESS = 0.20  # 敏感性：漂移放大到 20%
CAPITAL_MULTIPLE = 1.5  # 现货名义 1.0 + 永续保证金 0.5，收益率按此折算
MISSING_SHARE_WARN = 0.05  # 缺日占比超过此值，年化仅供参考
MIN_YEAR_DAYS = 365  # 少于此天数的年份是不完整年，年化是外推


@dataclass(frozen=True)
class YearStats:
    """单个自然年的资金费率收益。"""

    year: int
    days: int
    settlements: int
    annual: float
    positive_share: float
    worst_rolling: float

    @property
    def complete(self) -> bool:
        """是否为完整自然年（否则年化是外推值）。"""
        return self.days >= MIN_YEAR_DAYS


@dataclass(frozen=True)
class CarryReport:
    """一个标的的 carry 全样本统计。"""

    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    settlements: int
    missing_days: int
    gross_annual: float
    capital_adjusted: float
    amortized_round_trip: float
    rebalance_cost: float
    net_annual: float
    net_annual_stress: float
    positive_share: float
    median_daily: float
    rolling_median: float
    rolling_p5: float
    rolling_worst: float
    rolling_best: float
    rolling_dropped: int
    cumulative_p5: float
    cumulative_worst: float
    negative_share: float
    years: list[YearStats]

    @property
    def rolling_p5_capital(self) -> float:
        """滚动 P5 折到资本口径（滚动统计本身是现货名义口径）。"""
        return self.rolling_p5 / CAPITAL_MULTIPLE

    @property
    def cumulative_p5_capital(self) -> float:
        """30 天累计 P5 折到资本口径。"""
        return self.cumulative_p5 / CAPITAL_MULTIPLE

    @property
    def cumulative_worst_capital(self) -> float:
        """30 天累计最差折到资本口径。"""
        return self.cumulative_worst / CAPITAL_MULTIPLE

    @property
    def holding_years(self) -> float:
        return self.days / ANNUAL_DAYS


def daily_funding(funding: pd.Series) -> tuple[pd.Series, int]:
    """按日汇总资金费率，返回（日序列, 缺日数）。

    缺日不再静默补 0：resample 用 min_count=1 产出 NaN 并计数，缺日多则年化不可信。
    最后一天通常还没结算完，按整日计入会失真，若结算次数少于常态则剔除。
    """
    daily = daily_rates(funding)
    # pandas-stubs 未声明 Index.normalize，走 .dt 访问器（按自然日归组）
    counts = funding.groupby(funding.index.to_series().dt.normalize()).size()
    if len(counts) > 1 and int(counts.iloc[-1]) < float(counts.median()):
        daily = daily.iloc[:-1]
    missing = int(daily.isna().sum())
    return cast(pd.Series, daily.fillna(0.0)), missing


def rolling_cumulative(daily: pd.Series, window: int = ROLLING_DAYS) -> pd.Series:
    """滚动 window 天的累计收益（不年化，现货名义口径）。"""
    return cast(pd.Series, daily.rolling(window).sum()).dropna()


def rolling_annual(
    daily: pd.Series, window: int = ROLLING_DAYS
) -> tuple[pd.Series, int]:
    """滚动 window 天的复利年化（外推值），返回（序列, 亏光被剔除的窗口数）。"""
    cumulative = rolling_cumulative(daily, window)
    dropped = int((cumulative <= -1).sum())
    kept = cumulative[cumulative > -1]
    return cast(pd.Series, (1 + kept) ** (ANNUAL_DAYS / window) - 1), dropped


def cost_drag(
    holding_years: float, turnover: float = TURNOVER_SHARE
) -> tuple[float, float]:
    """资本口径的年化成本：（一次性进出摊薄, 再平衡）。

    一次性进出只在头尾各发生一次，必须摊到持有年数；再平衡调的是漂移量而非全部名义。
    """
    amortized = ROUND_TRIP_COST / CAPITAL_MULTIPLE / max(holding_years, 1 / ANNUAL_DAYS)
    rebalance = REBALANCE_PER_YEAR * ROUND_TRIP_COST * turnover / CAPITAL_MULTIPLE
    return amortized, rebalance


def _years(index: pd.Index) -> np.ndarray:
    """把时间索引转成年份数组，便于按年切片。"""
    return np.asarray(index.to_series().dt.year)


def yearly_stats(
    funding: pd.Series, daily: pd.Series, rolling: pd.Series
) -> list[YearStats]:
    """逐年统计年化、正费率占比与最差滚动窗口（滚动序列由调用方算好传入）。"""
    daily_year = _years(daily.index)
    funding_year = _years(funding.index)
    rolling_year = _years(rolling.index)
    out: list[YearStats] = []
    for year in sorted({int(value) for value in daily_year}):
        # stub 把布尔索引的返回标成 Series | ndarray，运行时必为 Series
        group = cast(pd.Series, daily[daily_year == year])
        same_year = funding[funding_year == year]
        window = rolling[rolling_year == year]
        out.append(
            YearStats(
                year=year,
                days=len(group),
                settlements=int(same_year.size),
                annual=annualize(total_return(group), len(group)),
                positive_share=float((group > 0).mean()) if len(group) else 0.0,
                worst_rolling=float(window.min()) if window.size else 0.0,
            )
        )
    return out


def build_carry_report(symbol: str, funding: pd.Series) -> CarryReport:
    """把资金费率序列整理成 carry 报告（含持有期成本与资本占用折算）。"""
    daily, missing = daily_funding(funding)
    rolling, dropped = rolling_annual(daily)
    cumulative = rolling_cumulative(daily)
    gross = annualize(total_return(daily), len(daily))
    adjusted = gross / CAPITAL_MULTIPLE
    holding = len(daily) / ANNUAL_DAYS
    amortized, rebalance = cost_drag(holding)
    _, stress_rebalance = cost_drag(holding, TURNOVER_SHARE_STRESS)
    return CarryReport(
        symbol=symbol,
        start=cast(pd.Timestamp, funding.index[0]),
        end=cast(pd.Timestamp, funding.index[-1]),
        days=len(daily),
        settlements=int(funding.size),
        missing_days=missing,
        gross_annual=gross,
        capital_adjusted=adjusted,
        amortized_round_trip=amortized,
        rebalance_cost=rebalance,
        net_annual=adjusted - amortized - rebalance,
        net_annual_stress=adjusted - amortized - stress_rebalance,
        positive_share=float((daily > 0).mean()) if len(daily) else 0.0,
        median_daily=float(daily.median()) if len(daily) else 0.0,
        rolling_median=float(rolling.median()) if rolling.size else 0.0,
        rolling_p5=float(np.percentile(rolling, 5)) if rolling.size else 0.0,
        rolling_worst=float(rolling.min()) if rolling.size else 0.0,
        rolling_best=float(rolling.max()) if rolling.size else 0.0,
        rolling_dropped=dropped,
        cumulative_p5=float(np.percentile(cumulative, 5)) if cumulative.size else 0.0,
        cumulative_worst=float(cumulative.min()) if cumulative.size else 0.0,
        negative_share=float((cumulative < 0).mean()) if cumulative.size else 0.0,
        years=yearly_stats(funding, daily, rolling),
    )


def _print_cost_lines(report: CarryReport) -> None:
    """成本与净收益几行：每一项都标出口径，避免读者拿名义口径对资本口径。"""
    print(
        f"毛年化（现货名义本金）        : {report.gross_annual:>8.2%}"
        f"   （按日复利，结算间隔由交易所决定）"
    )
    print(
        f"按资本占用 {CAPITAL_MULTIPLE:.1f} 倍折算       : {report.capital_adjusted:>8.2%}"
        f"   （现货 1.0 + 永续保证金 0.5）"
    )
    print(
        f"扣一次性进出（摊 {report.holding_years:.1f} 年）  : {report.amortized_round_trip:>8.2%}/年"
        f"   （双腿 {ROUND_TRIP_COST:.2%} 摊到持有期）"
    )
    print(
        f"再扣再平衡（漂移 {TURNOVER_SHARE:.0%}）      : {report.rebalance_cost:>8.2%}/年"
        f"   （每年 {REBALANCE_PER_YEAR} 次 × 双腿 × 换手 {TURNOVER_SHARE:.0%}）"
    )
    print(f"净年化                        : {report.net_annual:>8.2%}")
    print(
        f"  敏感性（漂移 {TURNOVER_SHARE_STRESS:.0%}）       : {report.net_annual_stress:>8.2%}"
    )


def print_carry_report(report: CarryReport) -> None:
    """打印单个标的的 carry 报告。"""
    gap = f" | 缺日 {report.missing_days} 天" if report.missing_days else ""
    print(f"\n{'=' * 118}")
    print(
        f"{report.symbol} | {report.start.date()} → {report.end.date()}"
        f" | {report.days} 天 | {report.settlements} 次结算{gap}"
        f" | 正费率占比 {report.positive_share:.1%} | 日均中位 {report.median_daily:.4%}"
    )
    print("=" * 118)
    _print_cost_lines(report)
    print(
        f"{ROLLING_DAYS} 天滚动年化（外推，现货名义）: 中位 {report.rolling_median:.2%}"
        f" | P5 {report.rolling_p5:.2%} | 最差 {report.rolling_worst:.2%}"
        f" | 最好 {report.rolling_best:.2%}"
    )
    print(
        f"{ROLLING_DAYS} 天累计（不年化）  : P5 {report.cumulative_p5:+.2%} 名义 / "
        f"{report.cumulative_p5_capital:+.2%} 资本 | 最差 {report.cumulative_worst:+.2%} 名义 / "
        f"{report.cumulative_worst_capital:+.2%} 资本 | 负窗口 {report.negative_share:.1%}"
    )
    if report.rolling_dropped:
        print(
            f"（另有 {report.rolling_dropped} 个窗口累计跌破 -100%，已从滚动统计剔除）"
        )
    if report.missing_days > report.days * MISSING_SHARE_WARN:
        print(
            f"⚠️ 缺日 {report.missing_days} 天"
            f"（{report.missing_days / report.days:.1%}），按 0% 补齐参与年化，结论仅供参考"
        )
    print(
        f"\n📌 窗口净年化 {report.net_annual:.2%}，但 {report.negative_share:.1%} 的 "
        f"{ROLLING_DAYS} 天窗口为负（累计 P5 {report.cumulative_p5_capital:+.2%} 资本、"
        f"最差 {report.cumulative_worst_capital:+.2%} 资本）：仓位由「能承受多大逆向波动而不被追保」"
        f"决定，不由费率的 P5 决定。"
    )

    print(f"\n逐年（{len(report.years)} 年，* = 不完整年，年化是外推值）")
    print(
        pad("年份", 6)
        + pad("天数", 6, "right")
        + pad("结算次数", 10, "right")
        + pad("年化", 10, "right")
        + pad("正费率占比", 12, "right")
        + pad("最差滚动", 12, "right")
    )
    print("-" * 118)
    for stat in report.years:
        mark = " " if stat.complete else "*"
        print(
            pad(f"{stat.year}{mark}", 6)
            + pad(str(stat.days), 6, "right")
            + pad(str(stat.settlements), 10, "right")
            + pad(f"{stat.annual:.1%}", 10, "right")
            + pad(f"{stat.positive_share:.1%}", 12, "right")
            + pad(f"{stat.worst_rolling:.1%}", 12, "right")
        )


def run_carry(symbols: list[str], days: int) -> bool:
    """资金费率 carry 研究入口：按研究窗口截断数据并打印报告。返回是否全部取到数据。"""
    source = FundingSource()  # 惰性联网：缓存够新时完全离线
    window_start = cast(
        pd.Timestamp, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days, unit="D")
    )
    print("\n资金费率 carry：做多现货 + 做空等额永续，价格风险对冲，收益只剩资金费率。")
    print(f"研究窗口：最近 {days} 天（{window_start.date()} 起）")
    ok = True
    for symbol in symbols:
        try:
            frame = source.load(symbol, days)
        except Exception:
            print(f"❌ {symbol} 资金费率获取失败")
            traceback.print_exc()
            ok = False
            continue
        frame = frame[frame.index >= window_start]
        if frame.empty:
            print(f"❌ {symbol} 研究窗口内无资金费率数据")
            ok = False
            continue
        print_carry_report(build_carry_report(symbol, cast(pd.Series, frame["rate"])))
    print(
        "\n注：费率历史取 Binance 8h 结算（长历史唯一可用源，OKX 只保留约 100 天），"
        "执行按 OKX 永续——「费率水平取 Binance、执行按 OKX」是跨源代理假设。\n"
        "    执行需要在同一账户同时持有现货多头与永续空头"
        "（OKX AU 需通过衍生品资格评估，零售杠杆上限 2x）。"
    )
    return ok
