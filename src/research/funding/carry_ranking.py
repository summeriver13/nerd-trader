"""carry 横截面分析：宽表、跨源校验、排名、风险调整、多重检验与分段。

`scripts/carry_ranking.py` 只负责打印，分析逻辑都在这里。补的是「这些收益
值不值得相信」：carry 的收益曲线很稳但尾部很重，只看年化会误判；而从 30 只
标的里挑出来的冠军等于 30 次试验，必须按试验次数做去膨胀夏普校正。

口径沿用 `carry.py`：收益按日求和（结算间隔 4h/8h 混用仍可比）、资本占用 1.5x、
成本按双腿 taker 摊销。夏普一律用**扣成本后的日收益**，否则会把成本算漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    ANNUAL_DAYS,
    annualized_sharpe,
    max_drawdown,
    sharpe_ratio,
)
from src.config.paths import (
    CRYPTO_ROOT,
    MARKET_FUNDING,
    RESULTS_ROOT,
    crypto_dataset_dir,
)
from src.research.funding.carry import (
    CAPITAL_MULTIPLE,
    CarryReport,
    build_carry_report,
    cost_drag,
)
from src.research.validation.overfitting import (
    deflated_sharpe,
    expected_max_sharpe,
    sharpe_estimator_variance,
)
from src.utils.data_sources.funding import daily_rates

FUNDING_TIMEFRAME = "8h"
# 排序主口径：唯一有长历史的源；跨源校验必须用 okx 对账
LONG_HISTORY_EXCHANGE = "binance"
COMPARISON_EXCHANGE = "okx"
# 跨源对比与组合都需要足够重叠/成员数，太低的相关没有意义
MIN_OVERLAP_DAYS = 30
MIN_PORTFOLIO_MEMBERS = 5
SEGMENTS = 4
OUTPUT_DIR = RESULTS_ROOT / "carry"
BP = 1e4


@dataclass(frozen=True)
class CrossRow:
    """一只标的的跨源同期对比（`diff` 为 Binance − OKX）。"""

    stem: str
    days: int
    binance_bp: float
    okx_bp: float
    diff_bp: float
    annual_diff: float
    corr: float
    sign_agree: float


@dataclass(frozen=True)
class RankRow:
    """一只标的的 carry 排名数据。"""

    report: CarryReport
    recent_net: float
    recent_days: int


@dataclass(frozen=True)
class RiskRow:
    """一只标的近期窗口的风险调整指标与去膨胀夏普。"""

    stem: str
    days: int
    annual_vol: float
    annual_sharpe: float
    max_dd: float
    daily_sharpe: float
    dsr: float
    observations: int
    passed: bool


@dataclass(frozen=True)
class SegmentRow:
    """等权组合在一个日历分段上的表现，以及该段内正收益标的的比例。"""

    label: str
    days: int
    members: int
    positive: int
    annual: float
    sharpe: float


def cache_dir(exchange_id: str) -> Path:
    """资金费率缓存目录。"""
    return crypto_dataset_dir(
        exchange_id, MARKET_FUNDING, FUNDING_TIMEFRAME, root=CRYPTO_ROOT
    )


def list_stems(exchange_id: str) -> list[str]:
    """扫描该交易所缓存目录下的全部标的。"""
    directory = cache_dir(exchange_id)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def load_rates(exchange_id: str, stem: str) -> pd.Series:
    """读一只标的的费率序列（index 为 UTC 结算时刻）。"""
    frame = pd.read_parquet(cache_dir(exchange_id) / f"{stem}.parquet")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return cast(pd.Series, frame.sort_index()["rate"])


def load_daily(exchange_id: str, stems: list[str]) -> pd.DataFrame:
    """把多只标的的费率按日求和拼成宽表（index 为日，列为标的）。"""
    series = {stem: daily_rates(load_rates(exchange_id, stem)) for stem in stems}
    return pd.DataFrame(series).sort_index()


def net_recent_daily(daily: pd.Series, recent_days: int) -> pd.Series:
    """取近期窗口，并把该窗口的持有成本按日摊进收益（夏普与 DSR 用净口径）。"""
    recent = cast(pd.Series, daily.iloc[-recent_days:]).dropna()
    if recent.empty:
        return recent
    amortized, rebalance = cost_drag(len(recent) / ANNUAL_DAYS)
    return cast(pd.Series, recent - (amortized + rebalance) / ANNUAL_DAYS)


def cross_row(stem: str) -> CrossRow | None:
    """同期对比两所的费率；重叠期不足或任一缺失则返回 None。"""
    left = cast(
        pd.Series, daily_rates(load_rates(LONG_HISTORY_EXCHANGE, stem))
    ).dropna()
    right = cast(pd.Series, daily_rates(load_rates(COMPARISON_EXCHANGE, stem))).dropna()
    shared = left.index.intersection(right.index)
    if len(shared) < MIN_OVERLAP_DAYS:
        return None
    left = cast(pd.Series, left.loc[shared])
    right = cast(pd.Series, right.loc[shared])
    deviation = left - right
    return CrossRow(
        stem=stem,
        days=len(shared),
        binance_bp=float(left.mean()) * BP,
        okx_bp=float(right.mean()) * BP,
        diff_bp=float(deviation.mean()) * BP,
        annual_diff=float(deviation.mean()) * ANNUAL_DAYS,
        corr=float(cast(float, left.corr(right))),
        sign_agree=float((np.sign(left) == np.sign(right)).mean()),
    )


def rank_row(stem: str, recent_days: int) -> RankRow:
    """用长历史源算一只标的的 carry 报告与近期口径净年化。"""
    report = build_carry_report(stem, load_rates(LONG_HISTORY_EXCHANGE, stem))
    daily = cast(pd.Series, daily_rates(load_rates(LONG_HISTORY_EXCHANGE, stem)))
    recent = net_recent_daily(daily, recent_days)
    recent_net = (
        float((1 + recent).prod() ** (ANNUAL_DAYS / len(recent)) - 1) / CAPITAL_MULTIPLE
        if len(recent)
        else 0.0
    )
    return RankRow(report=report, recent_net=recent_net, recent_days=len(recent))


def risk_rows(
    wide: pd.DataFrame, recent_days: int, trials: int
) -> tuple[list[RiskRow], float]:
    """算全部标的的近期风险指标与去膨胀夏普，返回 (行, 零假设期望最大夏普)。

    trials 是「从多少只标的里挑过冠军」，不校正就等于拿最好看的那只当结论。
    """
    recent = {
        stem: net_recent_daily(cast(pd.Series, wide[stem]), recent_days)
        for stem in wide.columns
    }
    sharpes = [sharpe_ratio(series) for series in recent.values() if len(series)]
    observations = int(
        np.median([len(series) for series in recent.values() if len(series)])
    )
    expected_max = expected_max_sharpe(
        sharpe_estimator_variance(sharpes, observations), trials
    )
    rows: list[RiskRow] = []
    for stem, series in recent.items():
        if series.empty:
            continue
        test = deflated_sharpe(stem, series, expected_max)
        rows.append(
            RiskRow(
                stem=stem,
                days=len(series),
                annual_vol=float(series.std()) * float(np.sqrt(ANNUAL_DAYS)),
                annual_sharpe=annualized_sharpe(series, ANNUAL_DAYS),
                max_dd=max_drawdown(series),
                daily_sharpe=float(test.sharpe),
                dsr=float(test.dsr),
                observations=int(test.observations),
                passed=bool(test.passed),
            )
        )
    return rows, expected_max


def portfolio_daily(wide: pd.DataFrame) -> pd.Series:
    """等权组合日收益：当天有结算的标的等权，成分太少的日子剔除。"""
    members = wide.notna().sum(axis=1)
    mean = cast(pd.Series, wide.mean(axis=1, skipna=True))
    return cast(pd.Series, mean[members >= MIN_PORTFOLIO_MEMBERS])


def segment_rows(wide: pd.DataFrame, count: int = SEGMENTS) -> list[SegmentRow]:
    """把等权组合按日历等分，给出各段年化、夏普与正收益标的比例。"""
    portfolio = portfolio_daily(wide)
    if portfolio.empty:
        return []
    start = cast(pd.Timestamp, portfolio.index[0])
    end = cast(pd.Timestamp, portfolio.index[-1])
    step = (end - start) / count
    rows: list[SegmentRow] = []
    for i in range(count):
        low = cast(pd.Timestamp, start + step * i)
        high = cast(pd.Timestamp, start + step * (i + 1))
        mask = portfolio.index >= low
        if i < count - 1:
            mask = mask & (portfolio.index < high)
        window = cast(pd.Series, portfolio[mask])
        if window.empty:
            continue
        block = wide[(wide.index >= low) & (wide.index <= high)]
        rows.append(
            SegmentRow(
                label=f"{low:%Y-%m}~{high:%Y-%m}",
                days=len(window),
                members=int((block.notna().sum(axis=0) > 0).sum()),
                positive=int((cast(pd.Series, block.mean(axis=0)) > 0).sum()),
                annual=float((1 + window).prod() ** (ANNUAL_DAYS / len(window)) - 1)
                / CAPITAL_MULTIPLE,
                sharpe=annualized_sharpe(window, ANNUAL_DAYS),
            )
        )
    return rows


def ranking_frame(
    rank_rows: list[RankRow], risk_by_stem: dict[str, RiskRow]
) -> pd.DataFrame:
    """把排名与风险指标拼成一张可落盘的窄表。"""
    records: list[dict[str, object]] = []
    for row in rank_rows:
        report = row.report
        risk = risk_by_stem.get(report.symbol)
        records.append(
            {
                "symbol": report.symbol,
                "start": report.start,
                "end": report.end,
                "hist_days": report.days,
                "settlements": report.settlements,
                "recent_days": row.recent_days,
                "recent_net": row.recent_net,
                "hist_net": report.net_annual,
                "hist_net_stress": report.net_annual_stress,
                "hist_gross": report.gross_annual,
                "positive_share": report.positive_share,
                "rolling_p5": report.rolling_p5,
                "rolling_worst": report.rolling_worst,
                "negative_share": report.negative_share,
                "annual_vol": risk.annual_vol if risk else None,
                "annual_sharpe": risk.annual_sharpe if risk else None,
                "max_dd": risk.max_dd if risk else None,
                "daily_sharpe": risk.daily_sharpe if risk else None,
                "dsr": risk.dsr if risk else None,
                "dsr_passed": risk.passed if risk else None,
            }
        )
    return pd.DataFrame(records)


def save_outputs(
    wide: pd.DataFrame,
    rank_rows: list[RankRow],
    cross_rows: list[CrossRow],
    risk_by_stem: dict[str, RiskRow],
) -> list[Path]:
    """落盘三张表：日费率宽表（供组合复用）、排名窄表、跨源校验表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    daily_path = OUTPUT_DIR / "carry_daily.parquet"
    wide.to_parquet(daily_path, compression="snappy")
    written.append(daily_path)

    ranking_path = OUTPUT_DIR / "carry_ranking.parquet"
    ranking_frame(rank_rows, risk_by_stem).to_parquet(ranking_path, index=False)
    written.append(ranking_path)

    cross_path = OUTPUT_DIR / "carry_cross_check.parquet"
    pd.DataFrame([row.__dict__ for row in cross_rows]).to_parquet(
        cross_path, index=False
    )
    written.append(cross_path)
    return written
