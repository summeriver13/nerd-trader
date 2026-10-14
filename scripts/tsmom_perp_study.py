"""
scripts/tsmom_perp_study.py
完整版 TSMOM：永续多空 + 真实资金费（OKX 永续行情 × Binance 资金费率）。

与现货版（`scripts/tsmom_study.py`）的读数**不可直接比**，池子与口径都变了：

- **池子更短更窄**：永续行情 2020-01 起（现货 2018-01 起）；Binance 只有现货行情，
  永续没有退市回填，而永续退市率高于现货，所以这里的幸存者偏差比现货版更重。
- **多了空头**：下破 entry 窗口最低价开空、上破 exit 窗口最高价平空（镜像状态机）。
- **多了资金费**：正费率 = 多头付钱、空头收钱，按仓位逐日计入净收益；费率取 Binance
  （长历史唯一源）、行情取 OKX，属跨源代理——`docs/carry` 已验跨源替代「对大盘币
  成立、对小币不成立」，故池子限定在两源都有数据的标的。

臂是「只多 / 多空」×「建资金费 / 不建」的对照：前者量空头带来什么，后者量资金费
吃掉多少（趋势跟随的择时与资金费负相关：涨的时候做多付高费、跌的时候做空常遇负费）。
另加两条同速度的「收益符号」臂（经典 TSMOM 口径），用来分辨「是空头不行，还是突破式
空头不行」：两族池子、缩放、成本全同，只差信号构造。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.core.costs import PERP_FEES, FeeSchedule
from src.research.tsmom import (
    SIGNAL_SIGN,
    TsmomOptions,
    active_counts,
    cross_section_mean,
    portfolio_returns,
    run_symbol,
    tsmom_signal,
)
from src.research.tsmom_report import (
    ArmRow,
    ArmRun,
    hit_rate,
    print_corrections,
    print_hits,
    print_rows,
    print_segments,
    summarize,
    summarize_stream,
)
from src.utils.data_sources.funding import (
    FUNDING_TIMEFRAME,
    MARKET_FUNDING,
    daily_rates,
)

PERP_DIR = crypto_dataset_dir("okx", "swap", "1d")
FUNDING_DIR = crypto_dataset_dir("binance", MARKET_FUNDING, FUNDING_TIMEFRAME)
OUTPUT = Path("data/results/breaks/tsmom_perp_study_swap_1d.parquet")
DEFAULT_MIN_BARS = 365
BASELINE = "bh_basket"
NO_FUNDING_SUFFIX = " @ nofund"
# 主臂：只多（现货口径的永续对照）、多空、多空不缩放、多空带 ATR 止损
ARMS: dict[str, TsmomOptions] = {
    "tsmom_long": TsmomOptions(),
    "tsmom_ls": TsmomOptions(allow_short=True),
    "tsmom_ls_raw": TsmomOptions(allow_short=True, vol_target=None),
    "tsmom_ls_atr": TsmomOptions(allow_short=True, atr_mult=2.0),
    # 符号信号臂：与突破臂同速度、同缩放、同成本，只差信号构造
    "tsmom_sign_long": TsmomOptions(kind=SIGNAL_SIGN),
    "tsmom_sign_ls": TsmomOptions(kind=SIGNAL_SIGN, allow_short=True),
}
# 不建资金费的对照臂只对两个主口径做（量资金费自己吃掉多少）
NO_FUNDING_ARMS: tuple[str, ...] = ("tsmom_long", "tsmom_ls")


def load_universe(min_bars: int) -> dict[str, pd.DataFrame]:
    """读 OKX 永续日线，只留够长的标的。"""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(PERP_DIR.glob("*.parquet")):
        bars = pd.read_parquet(path)
        if len(bars) >= min_bars:
            out[path.stem] = bars
    return out


def load_funding_map(stems: list[str]) -> dict[str, pd.Series]:
    """读 Binance 8h 资金费缓存并压成日频；没有缓存的标的不进池。

    `daily_rates` 按日求和（结算间隔由交易所决定）且缺日给 NaN，引擎把缺日当 0。
    """
    out: dict[str, pd.Series] = {}
    for stem in stems:
        path = FUNDING_DIR / f"{stem}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        out[stem] = daily_rates(cast(pd.Series, frame["rate"]))
    return out


def run_arm(
    bars_map: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.Series],
    build_signal: Callable[[pd.DataFrame], pd.Series],
    *,
    use_funding: bool,
    cost: FeeSchedule | None = None,
) -> ArmRun:
    """对每个标的跑一次引擎（可带资金费），返回逐标的的净收益 / 仓位 / 成本。"""
    net: dict[str, pd.Series] = {}
    pos: dict[str, pd.Series] = {}
    spend: dict[str, pd.Series] = {}
    for stem, bars in bars_map.items():
        result = run_symbol(
            bars,
            build_signal(bars),
            cost=cost,
            funding=funding_map.get(stem) if use_funding else None,
        )
        net[stem], pos[stem], spend[stem] = result.net_ret, result.pos, result.cost
    return ArmRun(net=net, pos=pos, cost=spend)


def run_all(
    bars_map: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.Series],
    *,
    cost: FeeSchedule | None = None,
) -> dict[str, ArmRun]:
    """四个主臂建资金费、两个对照臂不建，外加等权买入持有篮子（永续多头，建费）。"""
    runs: dict[str, ArmRun] = {}
    for label, options in ARMS.items():
        runs[label] = run_arm(
            bars_map,
            funding_map,
            lambda bars, opt=options: tsmom_signal(bars, opt),
            use_funding=True,
            cost=cost,
        )
    for label in NO_FUNDING_ARMS:
        options = ARMS[label]
        runs[label + NO_FUNDING_SUFFIX] = run_arm(
            bars_map,
            funding_map,
            lambda bars, opt=options: tsmom_signal(bars, opt),
            use_funding=False,
            cost=cost,
        )
    runs[BASELINE] = run_arm(
        bars_map,
        funding_map,
        lambda bars: pd.Series(1.0, index=bars.index),
        use_funding=True,
        cost=cost,
    )
    return runs


def funding_drag_rows(
    runs: dict[str, ArmRun], universe: str
) -> list[tuple[str, float, float, float, float, float, float]]:
    """资金费拖累：同一信号「建费 − 不建费」的年化差、两边年化、净/毛暴露。

    年化走与绩效表同一条路（`summarize_stream` 的复利年化），才与上面那张表对得上；
    拖累取两者之差（收益点数）。「每单位毛暴露」= 拖累 / 平均毛暴露——池均费率很低
    不等于持仓时费率低，策略恰好选在资金费高的时刻做多，这一列才是真实承担。
    """
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for label in NO_FUNDING_ARMS:
        funded = summarize_stream(universe, label, portfolio_returns(runs[label].net))
        bare = summarize_stream(
            universe, label, portfolio_returns(runs[label + NO_FUNDING_SUFFIX].net)
        )
        pos = pd.DataFrame(runs[label].pos)
        net_expo = float(cross_section_mean(pos).mean())
        gross_expo = float(cross_section_mean(pos.abs()).mean())
        drag = funded.annual_ret - bare.annual_ret
        # 按毛暴露归一：多空臂的净暴露接近 0 甚至为负，除以它会翻号
        per_expo = drag / gross_expo if gross_expo else float("nan")
        rows.append(
            (
                f"{universe}::{label}",
                drag,
                bare.annual_ret,
                funded.annual_ret,
                per_expo,
                net_expo,
                gross_expo,
            )
        )
    return rows


def print_funding(
    rows: list[tuple[str, float, float, float, float, float, float]],
) -> None:
    """打印资金费拖累表：年化拖累（负 = 被吃掉）与建费前后的年化对照。"""
    header = (
        f"{'arm':<26}{'年化拖累':>10}{'不建费':>9}{'建费':>9}"
        f"{'每单位毛暴露':>13}{'净暴露':>8}{'毛暴露':>8}"
    )
    print("\n=== 资金费拖累（同信号建费 − 不建费）===")
    print("> 正费率 = 多头付钱；趋势跟随涨时做多、跌时做空，两头都可能付")
    print(header)
    print("-" * len(header))
    for label, drag, bare, funded, per_expo, net_expo, gross_expo in rows:
        print(
            f"{label:<26}{drag:>10.1%}{bare:>9.1%}{funded:>9.1%}"
            f"{per_expo:>13.1%}{net_expo:>8.1%}{gross_expo:>8.1%}"
        )


def analyze(
    universe: str,
    bars_map: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.Series],
    cost: FeeSchedule,
) -> tuple[list[ArmRow], dict[str, ArmRun]]:
    """跑一个池子的全部臂并打印读数，返回组合表行与逐标的输出。"""
    runs = run_all(bars_map, funding_map, cost=cost)
    counts = active_counts(runs[BASELINE].net)
    print(
        f"\n{'=' * 96}\n池子 {universe}：{len(bars_map)} 只 | 组合标的数 N_t "
        f"中位 {counts.median():.0f}（首 {int(counts.iloc[0])} → 末 {int(counts.iloc[-1])}）"
        f" | 成本 {(cost.taker + cost.slippage) * 1e4:.1f}bp/边\n{'=' * 96}"
    )
    rows = [summarize(universe, label, run) for label, run in runs.items()]
    print_rows(
        f"=== 组合绩效（等权 1/N_t，{universe}）===",
        rows,
        "> 主臂建资金费；@ nofund 是同信号不建费的对照",
    )
    print_segments(
        f"=== 分段（多空主口径 vs 篮子，{universe}）===",
        portfolio_returns(runs["tsmom_ls"].net),
        portfolio_returns(runs[BASELINE].net),
    )
    print_segments(
        f"=== 分段（符号信号多空 vs 篮子，{universe}）===",
        portfolio_returns(runs["tsmom_sign_ls"].net),
        portfolio_returns(runs[BASELINE].net),
    )
    print_hits([(label, *hit_rate(run, runs[BASELINE])) for label, run in runs.items()])
    print_corrections(
        {label: portfolio_returns(run.net) for label, run in runs.items()}
    )
    return rows, runs


def main() -> int:
    parser = argparse.ArgumentParser(description="完整版 TSMOM：永续多空 + 资金费")
    parser.add_argument(
        "--min-bars", type=int, default=DEFAULT_MIN_BARS, help="入池最少根数"
    )
    args = parser.parse_args()

    bars_all = load_universe(args.min_bars)
    funding_map = load_funding_map(list(bars_all))
    bars_map = {stem: bars for stem, bars in bars_all.items() if stem in funding_map}
    panel = pd.DataFrame({stem: bars["close"] for stem, bars in bars_map.items()})
    index = cast(pd.DatetimeIndex, panel.index)
    days = pd.Series([len(value) for value in funding_map.values()])
    print(
        f"OKX 永续 {len(bars_all)} 只（≥{args.min_bars} 根）｜两源都有 "
        f"{len(bars_map)} 只｜面板 {index[0]:%Y-%m-%d} → {index[-1]:%Y-%m-%d}"
        f"（{len(index)} 根）｜资金费中位覆盖 {days.median():.0f} 天"
    )
    rates = pd.DataFrame(funding_map).stack()
    mean_daily = float(rates.mean())
    print(
        f"池均日费率 {mean_daily * 1e4:.2f}bp（年化 {mean_daily * 365:.1%}，正 = 多头付钱）"
    )

    rows, runs = analyze("swap", bars_map, funding_map, PERP_FEES)
    drag_rows = funding_drag_rows(runs, "swap")
    print_funding(drag_rows)

    frame = pd.DataFrame([row.__dict__ for row in rows])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"\n落盘：{OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
