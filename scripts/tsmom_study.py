"""
scripts/tsmom_study.py
多标的时序动量（TSMOM）：把趋势线从 2 个标的扩到整个现货池。

问题：单标的趋势线（`docs/trend`、`docs/portfolio`）里唐奇安在 BTC/ETH 上都跑赢买入
持有，但都过不了多重检验校正（最好 DSR 0.875 / 0.821，门槛 0.95），PBO 在 ETH 上
65.1%。两种解释：样本太少（8.7 年 × 2 标的），或读数本来就是运气。本脚本做的是**唯一
能提高统计功效的动作**——把标的数从 2 扩到整个池子：标的是独立的样本维度，不是时间。

口径（与趋势线一致，便于对照）：
- 信号：唐奇安突破，多速度（20/55/100）**取平均**而不是挑窗口；
- 仓位：多速度信号 × 波动率缩放（目标 40%、上限 1、不加杠杆）；
- 组合：等权 1/N_t（每个日期按当时有数据的标的数等分）；
- 成本：永续 taker 5bp + 滑点 5bp = 10bp/边（与 `docs/trend` 同口径），另报现货口径；
- 基准：等权买入持有篮子（同 1/N_t）与 **BTC 买入持有**（单标的、不可再分散的那条线）。

用法：
    uv run python scripts/tsmom_study.py
    uv run python scripts/tsmom_study.py --min-bars 730 --liquid-top 50
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    annualize,
    annualized_sharpe,
    bars_per_year,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from src.config.paths import crypto_dataset_dir
from src.core.costs import PERP_FEES, SPOT_FEES, FeeSchedule
from src.research.overfitting import (
    DSR_THRESHOLD,
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_estimator_variance,
)
from src.research.robustness import segment_table
from src.research.track_record import track_record
from src.research.tsmom import (
    TsmomOptions,
    active_counts,
    cross_section_mean,
    portfolio_returns,
    run_symbol,
    tsmom_signal,
)

SPOT_DIR = crypto_dataset_dir("okx", "spot", "1d")
OUTPUT = Path("data/results/breaks/tsmom_study_spot_1d.parquet")
DEFAULT_MIN_BARS = 365
DEFAULT_LIQUID_TOP = 30
SEGMENTS = 4
# 声明的试验数三档：臂数（6）、臂 × 两个池子（12）、再加单速度与成本的组合（30）
TRIAL_BANDS: tuple[int, ...] = (6, 12, 30)
MIN_HIT_BARS = 365
BTC_STEM = "BTC-USDT"
BASELINE = "bh_basket"
# 臂：主口径 + 关缩放 + 三个单速度 + ATR 止损。全部报出，不挑最好的那个
ARMS: dict[str, TsmomOptions] = {
    "tsmom_ms": TsmomOptions(),
    "tsmom_ms_raw": TsmomOptions(vol_target=None),
    "tsmom_20": TsmomOptions(entry_periods=(20,)),
    "tsmom_55": TsmomOptions(entry_periods=(55,)),
    "tsmom_100": TsmomOptions(entry_periods=(100,)),
    "tsmom_atr": TsmomOptions(atr_mult=2.0),
}
LABEL_WIDTH = 22


@dataclass(frozen=True)
class ArmRun:
    """一个臂的逐标的引擎输出。"""

    net: dict[str, pd.Series]
    pos: dict[str, pd.Series]
    cost: dict[str, pd.Series]


@dataclass(frozen=True)
class ArmRow:
    """一个臂的组合级绩效读数。"""

    universe: str
    label: str
    total_ret: float
    annual_ret: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    psr: float
    min_trl: float
    exposure: float
    turnover: float
    cost: float


def load_universe(min_bars: int) -> dict[str, pd.DataFrame]:
    """读现货日线缓存，只留有足够历史的标的（退市标的也在池内）。"""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(SPOT_DIR.glob("*.parquet")):
        bars = pd.read_parquet(path)
        if len(bars) >= min_bars:
            out[path.stem] = bars
    return out


def liquidity_order(bars_map: dict[str, pd.DataFrame], top: int) -> list[str]:
    """按全历史中位日名义成交额取前 top 只（池子的定义，不是交易规则）。"""
    median: dict[str, float] = {}
    for stem, bars in bars_map.items():
        dollar = bars["close"] * bars["volume"]
        median[stem] = float(dollar.median())
    return sorted(median, key=lambda stem: median[stem], reverse=True)[:top]


def run_arm(
    bars_map: dict[str, pd.DataFrame],
    build_signal: Callable[[pd.DataFrame], pd.Series],
    *,
    cost: FeeSchedule | None = None,
) -> ArmRun:
    """对每个标的跑一次引擎，返回逐标的的净收益 / 仓位 / 成本。"""
    net: dict[str, pd.Series] = {}
    pos: dict[str, pd.Series] = {}
    spend: dict[str, pd.Series] = {}
    for stem, bars in bars_map.items():
        result = run_symbol(bars, build_signal(bars), cost=cost)
        net[stem], pos[stem], spend[stem] = result.net_ret, result.pos, result.cost
    return ArmRun(net=net, pos=pos, cost=spend)


def run_all(bars_map: dict[str, pd.DataFrame], *, cost: FeeSchedule | None = None) -> dict[str, ArmRun]:
    """跑全部臂 + 买入持有基准。"""
    runs: dict[str, ArmRun] = {}
    for label, options in ARMS.items():
        runs[label] = run_arm(
            bars_map, lambda bars, opt=options: tsmom_signal(bars, opt), cost=cost
        )
    runs[BASELINE] = run_arm(
        bars_map, lambda bars: pd.Series(1.0, index=bars.index), cost=cost
    )
    return runs


def summarize(universe: str, label: str, run: ArmRun) -> ArmRow:
    """组合级读数：等权 1/N_t 拼成组合，再算绩效与记录体检。"""
    net = portfolio_returns(run.net)
    pos = portfolio_returns(run.pos)
    periods = bars_per_year(net.index)
    total = total_return(net)
    drawdown = max_drawdown(net)
    annual = annualize(total, len(net), int(round(periods)))
    record = track_record(label, net)
    return ArmRow(
        universe=universe,
        label=label,
        total_ret=total,
        annual_ret=annual,
        sharpe=annualized_sharpe(net, periods),
        sortino=sortino_ratio(net, periods),
        calmar=calmar_ratio(annual, drawdown),
        max_dd=drawdown,
        psr=record.psr,
        min_trl=record.min_track_record,
        exposure=float(pos.mean()),
        turnover=float(cross_section_mean(pd.DataFrame(run.pos).diff().abs()).mean()),
        cost=float(portfolio_returns(run.cost).sum()),
    )


def print_rows(title: str, rows: list[ArmRow], note: str = "") -> None:
    """打印组合表。"""
    print(f"\n{title}")
    if note:
        print(note)
    header = (
        f"{'arm':<{LABEL_WIDTH}}{'total':>10}{'annual':>9}{'sharpe':>8}{'sortino':>9}"
        f"{'calmar':>8}{'maxDD':>9}{'PSR':>7}{'MinTRL':>9}{'expo':>7}{'turn':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.label:<{LABEL_WIDTH}}{row.total_ret:>9.1%}{row.annual_ret:>9.1%}"
            f"{row.sharpe:>8.2f}{row.sortino:>9.2f}{row.calmar:>8.2f}"
            f"{row.max_dd:>9.1%}{row.psr:>7.3f}{row.min_trl:>9.0f}"
            f"{row.exposure:>7.1%}{row.turnover:>7.3f}"
        )


def print_segments(title: str, net: pd.Series, bench: pd.Series) -> None:
    """四等分全历史：策略与基准的逐段收益（不年化）。"""
    print(f"\n{title}")
    header = f"{'segment':<26}{'bars':>6}{'strategy':>11}{'benchmark':>11}{'excess':>10}{'maxDD':>9}"
    print(header)
    print("-" * len(header))
    for seg in segment_table(net, bench, SEGMENTS):
        label = f"{seg.start:%Y-%m} -> {seg.end:%Y-%m}"
        print(
            f"{label:<26}{seg.bars:>6}{seg.total_ret:>10.1%}{seg.bench_ret:>11.1%}"
            f"{seg.total_ret - seg.bench_ret:>10.1%}{seg.max_drawdown:>9.1%}"
        )


def hit_rate(run: ArmRun, baseline: ArmRun) -> tuple[float, float, int]:
    """逐标的对比：策略夏普高于该标的买入持有的比例、超额中位、样本数。"""
    diffs = [
        sharpe_ratio(net) - sharpe_ratio(baseline.net[stem])
        for stem, net in run.net.items()
        if stem in baseline.net and len(net) >= MIN_HIT_BARS
    ]
    array = np.asarray(diffs, dtype=float)
    return float((array > 0).mean()), float(np.median(array)), int(array.size)


def print_hits(rows: list[tuple[str, float, float, int]]) -> None:
    """打印逐标的命中率。"""
    print("\n=== 逐标的对比（策略夏普 vs 同标的买入持有）===")
    header = f"{'arm':<{LABEL_WIDTH}}{'n':>6}{'win_rate':>10}{'median_diff':>13}"
    print(header)
    print("-" * len(header))
    for label, rate, median, count in rows:
        print(f"{label:<{LABEL_WIDTH}}{count:>6}{rate:>9.1%}{median:>13.4f}")


def print_corrections(streams: dict[str, pd.Series], bands: tuple[int, ...]) -> None:
    """多重检验校正：这些臂里挑最好的算不算数。"""
    labels = list(streams)
    sharpes = [sharpe_ratio(streams[label]) for label in labels]
    observations = len(streams[labels[0]])
    variance = sharpe_estimator_variance(sharpes, observations)
    print("\n=== 多重检验校正（组合级，逐根夏普口径）===")
    header = (
        f"{'arm':<{LABEL_WIDTH}}{'sharpe/bar':>11}{'obs':>7}"
        + "".join(f"{'DSR@' + str(t):>10}" for t in bands)
        + "  verdict"
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        tests = [
            deflated_sharpe(label, streams[label], expected_max_sharpe(variance, trials=t))
            for t in bands
        ]
        cells = "".join(f"{test.dsr:>10.3f}" for test in tests)
        passed = tests[-1].dsr >= DSR_THRESHOLD
        print(
            f"{label:<{LABEL_WIDTH}}{tests[0].sharpe:>11.4f}{observations:>7}{cells}"
            f"  {'PASS' if passed else 'FAIL'}"
        )
    pbo = probability_of_backtest_overfitting(streams)
    if pbo is None:
        print("PBO: 样本不足以分块，未定义")
        return
    verdict = "PASS" if pbo.passed else "FAIL"
    print(
        f"PBO CSCV: blocks={pbo.blocks} splits={pbo.splits} pbo={pbo.pbo:.1%} "
        f"winner_oos_sharpe_median={pbo.winner_oos_sharpe_median:.2f} -> {verdict}"
    )


def analyze(universe: str, bars_map: dict[str, pd.DataFrame], cost: FeeSchedule) -> list[ArmRow]:
    """跑一个池子的全部臂并打印读数，返回该池子的组合表行。"""
    runs = run_all(bars_map, cost=cost)
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
        "> PSR 是夏普显著性的概率；MinTRL 是达到 95% 置信所需期数（对可用期数看）",
    )
    main = portfolio_returns(runs["tsmom_ms"].net)
    bench = portfolio_returns(runs[BASELINE].net)
    print_segments(f"=== 分段（主口径 vs 篮子，{universe}）===", main, bench)
    hits = [
        (label, *hit_rate(runs[label], runs[BASELINE]))
        for label in list(ARMS) + [BASELINE]
    ]
    print_hits(hits)
    streams = {label: portfolio_returns(run.net) for label, run in runs.items()}
    print_corrections(streams, TRIAL_BANDS)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="多标的时序动量（TSMOM）组合研究")
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS, help="入池最少根数")
    parser.add_argument("--liquid-top", type=int, default=DEFAULT_LIQUID_TOP, help="流动性子池大小")
    args = parser.parse_args()

    bars_map = load_universe(args.min_bars)
    liquid = liquidity_order(bars_map, args.liquid_top)
    print(f"全池 {len(bars_map)} 只（≥{args.min_bars} 根）｜流动性子池 {len(liquid)} 只")

    rows = analyze("all", bars_map, PERP_FEES)
    rows += analyze("liquid", {stem: bars_map[stem] for stem in liquid}, PERP_FEES)
    rows += analyze(
        "all_spot_fee",
        bars_map,
        FeeSchedule(maker=SPOT_FEES.maker, taker=SPOT_FEES.taker, slippage=SPOT_FEES.slippage),
    )
    if BTC_STEM in bars_map:
        runs = run_all({BTC_STEM: bars_map[BTC_STEM]}, cost=PERP_FEES)
        rows += [summarize("btc_only", label, run) for label, run in runs.items()]

    frame = pd.DataFrame([row.__dict__ for row in rows])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"\n落盘：{OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
