"""
scripts/tsmom_study.py
多标的时序动量（TSMOM）：把趋势线从 2 个标的扩到整个现货池。

问题：单标的趋势线（`docs/trend`、`docs/portfolio`）里唐奇安在 BTC/ETH 上都跑赢买入
持有，但都过不了多重检验校正（最好 DSR 0.875 / 0.821，门槛 0.95）。两种解释：样本太少
（8.7 年 × 2 标的），或读数本来就是运气。本脚本做的是**唯一能提高统计功效的动作**——
把标的数从 2 扩到整个池子：标的是独立的样本维度，不是时间。

本脚本还做两件收尾：

1. **正交性检验**：把池子按每日滚动波动中位数切两半，看时序收益是不是截面低波动异象
   （`vol_30`）的时序版——如果 TSMOM 只在低波动那一半有效，它就不是新信息。
2. **账本级缩放**：逐标的 40% 目标在信号多数时间为零时只换来约 12% 的账本波动，所以
   按**账本**滚动波动把整体推到 BTC 的实测波动（上限 3 倍），让收益与 BTC 可比。

口径（与趋势线一致，便于对照）：
- 信号：唐奇安突破，多速度（20/55/100）**取平均**而不是挑窗口；
- 仓位：多速度信号 × 波动率缩放（目标 40%、上限 1、不加杠杆）；
- 组合：等权 1/N_t（每个日期按当时有数据的标的数等分）；
- 成本：永续 taker 5bp + 滑点 5bp = 10bp/边（与 `docs/trend` 同口径），另报现货口径；
- 基准：等权买入持有篮子（同 1/N_t）与 **BTC 买入持有**。

用法：
    uv run python scripts/tsmom_study.py
    uv run python scripts/tsmom_study.py --min-bars 730 --liquid-top 50
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.core.costs import PERP_FEES, SPOT_FEES, FeeSchedule
from src.research.study.universe import liquidity_order, load_bars_dir
from src.research.trend.tsmom import (
    TsmomOptions,
    active_counts,
    book_scale,
    multi_speed_signal,
    portfolio_returns,
    run_symbol,
    split_portfolio,
    trailing_volatility,
    tsmom_signal,
)
from src.research.trend.tsmom_audit import print_audit
from src.research.trend.tsmom_report import (
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

SPOT_DIR = crypto_dataset_dir("okx", "spot", "1d")
# OKX 不保留退市历史，退市标的由 Binance 回填（scripts/fetch_delisted_bars.py）
BINANCE_SPOT_DIR = crypto_dataset_dir("binance", "spot", "1d")
OUTPUT = Path("data/results/breaks/tsmom_study_spot_1d.parquet")
DEFAULT_MIN_BARS = 365
DEFAULT_LIQUID_TOP = 30
# 账本级缩放上限：3 倍以上强平风险不可忽略，且本仓没有保证金/资金费模型
BOOK_SCALE_CAP = 3.0
VOL_SPLIT_SHARE = 0.5
# 正交性检验的宽度门槛：半池成员少于 10 只的日期不计（池子早期只有 2 只标的）
MIN_SPLIT_MEMBERS = 10
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


def run_all(
    bars_map: dict[str, pd.DataFrame], *, cost: FeeSchedule | None = None
) -> dict[str, ArmRun]:
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


def vol_split_rows(
    runs: dict[str, ArmRun], bars_map: dict[str, pd.DataFrame], universe: str
) -> list[ArmRow]:
    """正交性检验：按每日滚动波动中位数切两半，看时序收益是不是低波动异象的时序版。

    波动滞后一根（用 t−1 已知的读数决定 t 的成员），故无前视。
    """
    vols = {
        stem: cast(pd.Series, trailing_volatility(bars).shift(1))
        for stem, bars in bars_map.items()
    }
    rows: list[ArmRow] = []
    for arm in ("tsmom_ms", BASELINE):
        for low, tag in ((True, "low_vol"), (False, "high_vol")):
            net = split_portfolio(
                runs[arm].net,
                vols,
                low=low,
                share=VOL_SPLIT_SHARE,
                min_members=MIN_SPLIT_MEMBERS,
            )
            rows.append(summarize_stream(f"{universe}_{tag}", f"{arm} @ {tag}", net))
    return rows


def book_scale_rows(
    bars_map: dict[str, pd.DataFrame],
    runs: dict[str, ArmRun],
    universe: str,
    target: float,
) -> list[ArmRow]:
    """账本级缩放：按账本滚动波动把整体推到目标波动（上限 BOOK_SCALE_CAP 倍）。

    第一遍用不缩放的账本量波动，第二遍把同一个系数乘到每个标的的信号上再跑引擎——
    这样成本按真实仓位变化计（缩放系数本身的变化也会产生换手）。
    """
    book = portfolio_returns(runs["tsmom_ms_raw"].net)
    scale = book_scale(book, target=target, cap=BOOK_SCALE_CAP)
    scaled = run_arm(
        bars_map,
        lambda bars: (
            multi_speed_signal(bars, TsmomOptions(vol_target=None))
            * scale.reindex(bars.index).fillna(0.0)
        ),
    )
    label = f"book_scale@{target:.0%} (cap {BOOK_SCALE_CAP:.0f}x)"
    return [summarize(universe, label, scaled)]


def analyze(
    universe: str, bars_map: dict[str, pd.DataFrame], cost: FeeSchedule
) -> tuple[list[ArmRow], dict[str, ArmRun]]:
    """跑一个池子的全部臂并打印读数，返回该池子的组合表行与逐标的输出。"""
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
    print_corrections(streams)
    return rows, runs


def main() -> int:
    parser = argparse.ArgumentParser(description="多标的时序动量（TSMOM）组合研究")
    parser.add_argument(
        "--min-bars", type=int, default=DEFAULT_MIN_BARS, help="入池最少根数"
    )
    parser.add_argument(
        "--liquid-top", type=int, default=DEFAULT_LIQUID_TOP, help="流动性子池大小"
    )
    args = parser.parse_args()

    bars_map = load_bars_dir(SPOT_DIR, args.min_bars)
    liquid = liquidity_order(bars_map, args.liquid_top)
    btc_vol = (
        float(trailing_volatility(bars_map[BTC_STEM]).median())
        if BTC_STEM in bars_map
        else 0.40
    )
    print(
        f"全池 {len(bars_map)} 只（≥{args.min_bars} 根）｜流动性子池 {len(liquid)} 只"
        f"｜账本缩放目标 = BTC 实测波动中位 {btc_vol:.1%}"
    )

    rows, runs = analyze("all", bars_map, PERP_FEES)
    scaled_rows = book_scale_rows(bars_map, runs, "all", btc_vol)
    print_rows(
        "=== 账本级缩放（把账本推到 BTC 的波动，收益才与 BTC 可比）===",
        [row for row in rows if row.label == "tsmom_ms_raw"] + scaled_rows,
    )
    rows += scaled_rows
    split_rows = vol_split_rows(runs, bars_map, "all")
    print_rows(
        "=== 正交性检验（按每日滚动波动中位数切两半，半池 ≥10 只才算）===",
        split_rows,
        "> 若 TSMOM 只在低波动那一半有效，它就只是截面低波动异象的时序版",
    )
    rows += split_rows
    delisted = load_bars_dir(BINANCE_SPOT_DIR, args.min_bars)
    if delisted:
        print(f"退市池（Binance 回填）：{len(delisted)} 只")
        delisted_rows, _ = analyze(
            "all_with_delisted", {**bars_map, **delisted}, PERP_FEES
        )
        rows += delisted_rows
    rows += print_audit(
        runs,
        bars_map,
        arm="tsmom_ms",
        baseline=BASELINE,
        btc_stem=BTC_STEM,
        universe="all",
    )
    liquid_rows, _ = analyze(
        "liquid", {stem: bars_map[stem] for stem in liquid}, PERP_FEES
    )
    rows += liquid_rows
    spot_rows, _ = analyze(
        "all_spot_fee",
        bars_map,
        FeeSchedule(
            maker=SPOT_FEES.maker, taker=SPOT_FEES.taker, slippage=SPOT_FEES.slippage
        ),
    )
    rows += spot_rows
    if BTC_STEM in bars_map:
        btc_runs = run_all({BTC_STEM: bars_map[BTC_STEM]}, cost=PERP_FEES)
        rows += [summarize("btc_only", label, run) for label, run in btc_runs.items()]

    frame = pd.DataFrame([row.__dict__ for row in rows])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"\n落盘：{OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
