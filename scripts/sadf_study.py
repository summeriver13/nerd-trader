"""SADF 爆炸期的信息增量研究（de Prado 第 17 章）。

**假设**：反转与动量都只看「过去涨了多少」，分不出**可持续的趋势**与**不可持续的
爆炸**。SADF 是统计意义上的爆炸性——如果它有信息，形态应当是**短期延续、长期
反转**（爆炸在继续，但终会回归），而不是单调的动量或反转。

**必须做的控制**：爆炸是罕见事件（实测占可交易日 1~3%），而且**几乎全落在过去
收益最高那一档**（最高档 518 个爆炸观测，其余四档只有 2/3/28/46）。所以「爆炸组
跑输」很可能只是「涨得更多 → 反转更多」的另一种说法。主表因此是
`double_sort_ic`——**控制过去 30 日收益分档之后**，SADF 在档内还有没有 IC。
不做这一步就会把已有的反转结论当成新发现。

**口径**：SADF 在 t 时刻只用 t 及之前的数据，无前视。门槛是随机游走的蒙特卡洛
分位（`breaks.monte_carlo_critical`），不是真实数据自己的分位——后者等于
「比过去更爆炸才算爆炸」，会把常态当爆炸。

SADF 包成 `FactorSpec` 之后直接复用现成的双重排序、分位组合与换手/成本口径，
不另写一套。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Final, cast

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.breaks import (
    SadfOptions,
    explosive_mask,
    monte_carlo_critical,
    sadf_series,
)
from src.research.factors.evaluation import forward_returns, rank_ic_series
from src.research.factors.library import DIRECTION_LONG, FactorSpec, momentum
from src.research.factors.orthogonal import double_sort_ic
from src.research.factors.panel import Panel, load_panel, tradable_mask
from src.research.factors.portfolio import PortfolioOptions, portfolio_stats
from src.utils.study_cli import add_market_argument, add_timeframe_argument

HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 15, 30)
CONTROL_WINDOW: Final[int] = 30
ONE_WAY_COST_BP: Final[float] = 5.0
SADF_ROOT: Final[Path] = RESULTS_ROOT / "breaks"


def sadf_panel(panel: Panel, options: SadfOptions) -> pd.DataFrame:
    """逐标的的 SADF 面板（列与 panel.close 对齐）。"""
    columns: dict[str, pd.Series] = {}
    for symbol in panel.symbols:
        close = cast(pd.Series, panel.close[symbol]).dropna()
        columns[symbol] = sadf_series(close, options).reindex(panel.index)
    return pd.DataFrame(columns, index=panel.index)


def cached_sadf(panel: Panel, options: SadfOptions, market: str) -> pd.DataFrame:
    """SADF 面板：有缓存就用缓存，避免每次重算两分钟。"""
    path = SADF_ROOT / f"sadf_{market}_1d.parquet"
    if path.is_file():
        cached = pd.read_parquet(path)
        if cached.shape == panel.close.shape:
            return cast(pd.DataFrame, cached)
    started = time.time()
    frame = sadf_panel(panel, options)
    SADF_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    print(f"  SADF 面板已算并落盘（{time.time() - started:.0f} 秒）→ {path}")
    return frame


def sadf_spec(panel: Panel, options: SadfOptions, market: str) -> FactorSpec:
    """把 SADF 面板包成 `FactorSpec`，接进现成的双重排序与组合层流水线。

    面板只算一次并缓存在闭包里：`compute` 会被流水线反复调用，每次重算两分钟。
    """
    cache: list[pd.DataFrame] = []

    def compute(target: Panel) -> pd.DataFrame:
        if not cache:
            cache.append(cached_sadf(target, options, market))
        return cache[0]

    return FactorSpec(
        key="sadf",
        label="SADF 爆炸性",
        family="breaks",
        direction=DIRECTION_LONG,
        warmup=options.min_window,
        compute=compute,
    )


def control_spec() -> FactorSpec:
    """控制变量：过去 30 日收益（双重排序的分档依据）。"""
    return FactorSpec(
        key="past_30",
        label="过去 30 日收益",
        family="momentum",
        direction=DIRECTION_LONG,
        warmup=CONTROL_WINDOW,
        compute=lambda panel: momentum(panel, CONTROL_WINDOW),
    )


def critical_value(panel: Panel, options: SadfOptions) -> float:
    """按面板长度取蒙特卡洛临界值（随机游走的 95% 分位）。"""
    length = int(panel.close.shape[0])
    started = time.time()
    value = monte_carlo_critical(length, options)
    print(
        f"  蒙特卡洛临界值（n={length}）= {value:.3f}（{time.time() - started:.0f} 秒）"
    )
    return value


def unconditional_ic(
    spec: FactorSpec, panel: Panel, mask: pd.DataFrame
) -> list[dict[str, object]]:
    """SADF 与前向收益的截面 IC，另附「过去收益」作对照。"""
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        forward = forward_returns(panel, horizon)
        for label, target in (("SADF", spec), ("过去收益", control_spec())):
            series = rank_ic_series(target.values(panel).where(mask), forward, mask)
            rows.append(
                {
                    "factor": label,
                    "horizon": horizon,
                    "mean_ic": float(series.mean()) if not series.empty else 0.0,
                    "periods": int(series.size),
                }
            )
    return rows


def controlled_ic(
    spec: FactorSpec, panel: Panel, mask: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    """控制过去收益分档后的档内 IC：SADF 有没有**增量**。"""
    control = control_spec()
    return {
        horizon: double_sort_ic(spec, control, panel, mask, horizon)
        for horizon in HORIZONS
    }


def portfolio_rows(
    spec: FactorSpec, panel: Panel, mask: pd.DataFrame
) -> list[dict[str, object]]:
    """组合层：分位价差、换手与平衡成本（复用现成口径，不另写）。"""
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        stats = portfolio_stats(
            spec,
            panel,
            mask,
            PortfolioOptions(horizon=horizon, cost_one_way=ONE_WAY_COST_BP / 1e4),
        )
        rows.append(
            {
                "horizon": horizon,
                "periods": stats.periods,
                "gross": stats.gross_spread * 100.0,
                "net": stats.net_spread * 100.0,
                "annual_gross": stats.annual_gross * 100.0,
                "turnover": stats.turnover,
                "break_even_bp": stats.break_even_cost * 1e4,
            }
        )
    return rows


def report_unconditional(rows: list[dict[str, object]]) -> str:
    """无条件 IC 表。"""
    frame = cast(pd.DataFrame, pd.DataFrame(rows))
    pivot = frame.pivot_table(index="horizon", columns="factor", values="mean_ic")
    lines = [
        "\n=== 无条件截面 IC（与过去 30 日收益对照）===",
        f"{'期限':>6}{'SADF':>12}{'过去收益':>12}",
    ]
    lines.extend(
        f"{int(horizon):>6}{float(pivot.loc[horizon, 'SADF']):>12.4f}"
        f"{float(pivot.loc[horizon, '过去收益']):>12.4f}"
        for horizon in pivot.index
    )
    return "\n".join(lines)


def report_controlled(tables: dict[int, pd.DataFrame]) -> str:
    """档内 IC 表：**这才是判断有没有增量的主表**。"""
    lines = ["\n=== 控制过去收益分档后的档内 IC（主表：有增量才会同号且显著）==="]
    for horizon, table in tables.items():
        if table.empty:
            lines.append(f"\n前向 {horizon} 日：无足够期次")
            continue
        lines.append(f"\n前向 {horizon} 日")
        lines.append(
            f"{'过去收益档':>10}{'档内 IC':>12}{'t 值':>10}{'期数':>8}{'档内中位收益':>14}"
        )
        for bucket, row in table.iterrows():
            lines.append(
                f"{int(cast(int, bucket)) + 1:>10}{row['ic']:>12.4f}{row['t_stat']:>10.2f}"
                f"{int(row['periods']):>8}{row['control_median']:>14.2%}"
            )
    return "\n".join(lines)


def report_portfolio(rows: list[dict[str, object]]) -> str:
    """组合层表：毛/净价差、年化、换手、平衡成本。"""
    lines = [
        f"\n=== 组合层（分位价差，往返 {ONE_WAY_COST_BP:.0f}bp/边）===",
        f"{'期限':>6}{'期数':>8}{'毛价差(%)':>12}{'净价差(%)':>12}{'年化毛(%)':>12}"
        f"{'换手':>8}{'平衡成本(bp)':>14}",
    ]
    lines.extend(
        f"{int(cast(int, row['horizon'])):>6}{int(cast(int, row['periods'])):>8}"
        f"{float(cast(float, row['gross'])):>12.3f}{float(cast(float, row['net'])):>12.3f}"
        f"{float(cast(float, row['annual_gross'])):>12.1f}"
        f"{float(cast(float, row['turnover'])):>8.3f}"
        f"{float(cast(float, row['break_even_bp'])):>14.1f}"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    """跑完整条链：SADF → 临界值 → 无条件 IC → 档内 IC → 组合层。"""
    parser = argparse.ArgumentParser(description="SADF 爆炸期研究")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser)
    args = parser.parse_args()

    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel)
    print(
        f"SADF 爆炸期 | {args.market} · {args.timeframe} | "
        f"{panel.close.shape[1]} 只，{panel.close.shape[0]} 天"
    )
    options = SadfOptions()
    spec = sadf_spec(panel, options, args.market)
    critical = critical_value(panel, options)
    flag = explosive_mask(spec.compute(panel), critical)
    tradable_days = int(mask.to_numpy().sum())
    boom_days = int((flag & mask).to_numpy().sum())
    print(
        f"  爆炸日占比 {boom_days / max(tradable_days, 1):.2%}"
        f"（{boom_days}/{tradable_days} 个可交易日）"
    )

    print(report_unconditional(unconditional_ic(spec, panel, mask)))
    print(report_controlled(controlled_ic(spec, panel, mask)))
    print(report_portfolio(portfolio_rows(spec, panel, mask)))


if __name__ == "__main__":
    main()
