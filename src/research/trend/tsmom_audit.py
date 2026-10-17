"""
src/research/trend/tsmom_audit.py
多标的时序动量的收尾体检：夏普的平稳块 bootstrap 区间、数据时效与幸存者偏差。

为什么单做这两件：

- DSR/PBO 回答「在这些臂里挑最好的算不算数」，但不回答「这条记录的夏普本身有多不确定」。
  平稳块 bootstrap（复用 `corrections.py` 的 Politits-Romano 实现）给夏普、以及「对基准的
  夏普差」的区间，且不假设 iid——金融序列自相关，iid 重采样会把区间造窄。
- 池子的缓存末端参差（各标的抓取时间不同），末段样本很薄；同时池子以今天的上币列表为
  主体，幸存者偏差方向未量化。这两件都可能把结论抬得太好看，必须单独量一次。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.metrics import sharpe_ratio
from src.research.trend.tsmom import (
    active_counts,
    cross_section_mean,
    portfolio_returns,
)
from src.research.trend.tsmom_report import (
    ArmRow,
    ArmRun,
    print_rows,
    summarize_stream,
)
from src.research.validation.corrections import (
    default_block,
    stationary_bootstrap_indices,
)

DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20261014
# 块长敏感性：默认 T^(1/3) 之外再看短块（近似 iid）与长块（更保守）
BLOCK_CHOICES: tuple[int, ...] = (5, 10, 20, 40)
# 宽度过滤门槛：组合标的数少于它的日期不计（末段缓存参差时样本很薄）
MIN_BREADTH = 30
# 存活判定：最后行情距面板末端在这个天数以内的算「仍然活着」
ALIVE_WINDOW_DAYS = 30
# 「退市」判定：最后行情距面板末端超过这个天数
STALE_WINDOW_DAYS = 90
# 长/短历史分档：看好处是不是只来自上市不久的新币
LONG_HISTORY_BARS = 2000
SHORT_HISTORY_BARS = 730
LOW_QUANTILE = 0.025
HIGH_QUANTILE = 0.975


@dataclass(frozen=True)
class SharpeInterval:
    """一次 bootstrap 的夏普读数：点估计 + 区间 + 非正比例。"""

    label: str
    sharpe: float
    low: float
    high: float
    share_nonpositive: float
    block: int
    replicates: int


def bootstrap_sharpe(
    returns: pd.Series,
    *,
    label: str,
    block: int | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> SharpeInterval:
    """逐根夏普的平稳块 bootstrap 区间；块长默认取 T^(1/3)（corrections 口径）。"""
    values = np.asarray(returns.dropna(), dtype=float)
    chosen = block or default_block(values.size)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = values[stationary_bootstrap_indices(values.size, chosen, rng)]
        error = float(sample.std(ddof=1))
        draws[index] = float(sample.mean()) / error if error > 0 else 0.0
    low, high = np.quantile(draws, [LOW_QUANTILE, HIGH_QUANTILE])
    return SharpeInterval(
        label=label,
        sharpe=sharpe_ratio(values),
        low=float(low),
        high=float(high),
        share_nonpositive=float((draws <= 0).mean()),
        block=chosen,
        replicates=replicates,
    )


def bootstrap_sharpe_diff(
    left: pd.Series,
    right: pd.Series,
    *,
    label: str,
    block: int | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> SharpeInterval:
    """配对重采样：两条序列用同一批下标，区间是（左夏普 − 右夏普）。

    配对是关键：两侧同一天同一批块，消掉共同的市场波动，否则区间会被 beta 撑大。
    """
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    left_values = frame["left"].to_numpy(dtype=float)
    right_values = frame["right"].to_numpy(dtype=float)
    chosen = block or default_block(len(frame))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        rows = stationary_bootstrap_indices(len(frame), chosen, rng)
        draws[index] = sharpe_ratio(left_values[rows]) - sharpe_ratio(
            right_values[rows]
        )
    low, high = np.quantile(draws, [LOW_QUANTILE, HIGH_QUANTILE])
    return SharpeInterval(
        label=label,
        sharpe=sharpe_ratio(left_values) - sharpe_ratio(right_values),
        low=float(low),
        high=float(high),
        share_nonpositive=float((draws <= 0).mean()),
        block=chosen,
        replicates=replicates,
    )


def coverage_table(bars_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """逐标的的根数与首末日期——看缓存末端参差到什么程度。"""
    rows = [
        {
            "stem": stem,
            "bars": len(bars),
            "first": bars.index[0],
            "last": bars.index[-1],
        }
        for stem, bars in bars_map.items()
    ]
    return pd.DataFrame(rows)


def alive_at_end(
    bars_map: dict[str, pd.DataFrame], *, window_days: int = ALIVE_WINDOW_DAYS
) -> list[str]:
    """面板末端仍在更新的标的（最后行情距末端不超过 window_days）。"""
    last = max(bars.index[-1] for bars in bars_map.values())
    threshold = last - np.timedelta64(window_days, "D")
    return sorted(
        stem for stem, bars in bars_map.items() if bars.index[-1] >= threshold
    )


def breadth_by_year(streams: dict[str, pd.Series]) -> pd.Series:
    """每年参与组合的标的数（年内中位）。"""
    counts = active_counts(streams)
    stamps = counts.index.to_numpy(dtype="datetime64[ns]")
    years = pd.Series(
        stamps.astype("datetime64[Y]").astype(int) + 1970, index=counts.index
    )
    return counts.groupby(years).median()


def print_intervals(title: str, rows: list[SharpeInterval], note: str = "") -> None:
    """打印 bootstrap 区间表。"""
    print(f"\n{title}")
    if note:
        print(note)
    header = (
        f"{'series':<28}{'sharpe/bar':>11}{'low':>9}{'high':>9}"
        f"{'P(<=0)':>9}{'block':>7}{'reps':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.label:<28}{row.sharpe:>11.4f}{row.low:>9.4f}{row.high:>9.4f}"
            f"{row.share_nonpositive:>9.3f}{row.block:>7}{row.replicates:>7}"
        )


def audit_slices(
    runs: dict[str, ArmRun],
    bars_map: dict[str, pd.DataFrame],
    *,
    arm: str,
    baseline: str,
    universe: str,
) -> list[ArmRow]:
    """体检切片：宽度过滤与存活/退市子池——都从同一批逐标的收益流取，不重跑引擎。"""
    breadth = active_counts(runs[baseline].net)
    long_symbols = [
        stem for stem, bars in bars_map.items() if len(bars) >= LONG_HISTORY_BARS
    ]
    short_symbols = [
        stem for stem, bars in bars_map.items() if len(bars) < SHORT_HISTORY_BARS
    ]
    plans: dict[str, tuple[list[str] | None, pd.Series | None]] = {
        f"breadth>={MIN_BREADTH}": (None, breadth >= MIN_BREADTH),
        "long_history": (long_symbols, None),
        "short_history": (short_symbols, None),
    }
    rows: list[ArmRow] = []
    for tag, (members, keep) in plans.items():
        for name in (arm, baseline):
            streams = runs[name].net
            if members is not None:
                streams = {stem: streams[stem] for stem in members}
            net = cross_section_mean(pd.DataFrame(streams))
            if keep is not None:
                net = net.where(keep)
            rows.append(summarize_stream(f"{universe}_{tag}", f"{name} @ {tag}", net))
    return rows


def print_audit(
    runs: dict[str, ArmRun],
    bars_map: dict[str, pd.DataFrame],
    *,
    arm: str,
    baseline: str,
    btc_stem: str,
    universe: str,
) -> list[ArmRow]:
    """一次跑完收尾体检：切片读数 + 数据时效 + 夏普 bootstrap 区间。"""
    slices = audit_slices(runs, bars_map, arm=arm, baseline=baseline, universe=universe)
    print_rows(
        "=== 收尾体检：宽度过滤与存活/退市子池（同一批收益流切片，不重跑引擎）===",
        slices,
        f"> 宽度过滤 = 组合标的数 ≥{MIN_BREADTH} 的日期；存活 = 末端仍在更新的标的",
    )
    print_coverage(
        "=== 数据时效：缓存末端与逐年宽度 ===",
        coverage_table(bars_map),
        breadth_by_year(runs[baseline].net),
        alive_at_end(bars_map),
    )
    book = portfolio_returns(runs[arm].net)
    bench = runs[baseline].net[btc_stem]
    intervals = [
        bootstrap_sharpe(book, label=f"{arm} @block={block or 'auto'}", block=block)
        for block in (None, *BLOCK_CHOICES)
    ]
    intervals.append(bootstrap_sharpe_diff(book, bench, label=f"{arm} − BTC 买入持有"))
    intervals.append(bootstrap_sharpe(bench, label="BTC 买入持有"))
    print_intervals(
        "=== 夏普的平稳块 bootstrap（配对重采样；块长 T^(1/3) 与 5/10/20/40）===",
        intervals,
        "> P(<=0) 是重采样里夏普非正的比例；配对差用同一批块，消掉共同的市场波动",
    )
    return slices


def print_coverage(
    title: str, table: pd.DataFrame, breadth: pd.Series, alive: list[str]
) -> None:
    """打印数据时效：末端分布、逐年宽度、存活/退市只数。"""
    print(f"\n{title}")
    last = table["last"].max()
    stale = table[table["last"] < last - np.timedelta64(STALE_WINDOW_DAYS, "D")]
    print(
        f"标的 {len(table)} 只｜面板末端 {last:%Y-%m-%d}｜末端仍在更新 {len(alive)} 只"
        f"｜末端落后超过 {STALE_WINDOW_DAYS} 天的 {len(stale)} 只"
    )
    if stale.empty:
        print(
            "→ 池子不含退市标的（已知事实：OKX 不保留退市历史，见 scripts/fetch_delisted_bars.py）；"
            "幸存者偏差另用 Binance 回填的 15 只跑一版 all_with_delisted"
        )
    print("逐年参与组合的标的数（中位）：")
    print("  " + "  ".join(f"{year}:{value:.0f}" for year, value in breadth.items()))
    if not stale.empty:
        stamps = np.asarray(stale["last"], dtype="datetime64[ns]")
        years = stamps.astype("datetime64[Y]").astype(int) + 1970
        counts = pd.Series(years).value_counts().sort_index()
        print(
            "末端落后标的的结束年份分布："
            + "  ".join(f"{year}:{count}" for year, count in counts.items())
        )
