"""
scripts/tsmom_capacity_study.py
容量口径：账本级 3 倍杠杆在多大账本规模下还成立。

问题（`docs/trend` §局限「杠杆与容量没走完」）：账本级缩放把账本整体推到目标波动
（上限 3 倍），但它假设每个标的都能同倍放大。真实约束不是「几倍」，而是「这笔钱占该
标的成交额多少」——细价币上 3 倍等于自己造出一个市场。本脚本按容量约束重跑同一策略：

    倍率_i,t = min(账本倍率_t, 参与率 × ADV_i,t × N_t / M, 3)

ADV 取滞后 30 根的中位日成交额（只用 t−1 及更早的读数），M 是账本名义规模。M 越小越
接近无约束 3 倍、M 越大越被池尾压住，这条曲线给出容量上限；「capped」列的读数回答
「扣掉拿不到额度的标的之后还剩多少」。

用法：
    uv run python scripts/tsmom_capacity_study.py
    uv run python scripts/tsmom_capacity_study.py --participation 0.02 --notionals 1000000,10000000
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.core.costs import PERP_FEES, FeeSchedule
from src.research.leverage_cap import (
    DEFAULT_NOTIONALS,
    MAX_MULTIPLIER,
    PARTICIPATION,
    Capacity,
    CapacitySpec,
    capacity_frame,
    run_capacity,
)
from src.research.tsmom import (
    SymbolRun,
    TsmomOptions,
    active_counts,
    book_scale,
    multi_speed_signal,
    portfolio_returns,
    run_symbol,
    trailing_volatility,
)
from src.research.tsmom_report import ArmRun, summarize
from src.research.universe import load_bars_dir

SPOT_DIR = crypto_dataset_dir("okx", "spot", "1d")
OUTPUT = Path("data/results/breaks/tsmom_capacity_spot_1d.parquet")
DEFAULT_MIN_BARS = 365
BTC_STEM = "BTC-USDT"
BOOK_SCALE_CAP = 3.0
BARS_PER_YEAR = 365
LABEL_WIDTH = 22
# 缩放口径的信号不带逐标的波动率目标（与 tsmom_study 的账本缩放同口径）
SCALED_OPTIONS = TsmomOptions(vol_target=None)


@dataclass(frozen=True)
class CapacityRow:
    """容量表的一行：绩效 + 容量诊断。"""

    label: str
    annual_ret: float
    vol: float
    sharpe: float
    max_dd: float
    exposure: float
    turnover: float
    mean_multiplier: float
    constrained_share: float


def engine_runs(
    bars_map: dict[str, pd.DataFrame], scale: pd.Series | None, cost: FeeSchedule
) -> dict[str, SymbolRun]:
    """跑一遍引擎：逐标的信号 × 当根缩放系数（``scale=None`` 即不缩放）。"""
    out: dict[str, SymbolRun] = {}
    for stem, bars in bars_map.items():
        signal = multi_speed_signal(bars, SCALED_OPTIONS)
        if scale is not None:
            signal = signal * scale.reindex(bars.index).fillna(0.0)
        out[stem] = run_symbol(bars, signal, cost=cost)
    return out


def to_arm_run(runs: dict[str, SymbolRun]) -> ArmRun:
    """逐标的引擎输出 → 组合读数需要的形状。"""
    return ArmRun(
        net={stem: run.net_ret for stem, run in runs.items()},
        pos={stem: run.pos for stem, run in runs.items()},
        cost={stem: run.cost for stem, run in runs.items()},
    )


def annual_vol(runs: dict[str, SymbolRun]) -> float:
    """账本年化波动：组合净收益的标准差年化。"""
    net = portfolio_returns({stem: run.net_ret for stem, run in runs.items()}).dropna()
    return float(net.std() * np.sqrt(BARS_PER_YEAR))


def record(
    label: str, runs: dict[str, SymbolRun], capacity: Capacity | None
) -> CapacityRow:
    """一行读数；无容量约束的行把两条诊断置 NaN。"""
    arm = summarize("capacity", label, to_arm_run(runs))
    return CapacityRow(
        label=label,
        annual_ret=arm.annual_ret,
        vol=annual_vol(runs),
        sharpe=arm.sharpe,
        max_dd=arm.max_dd,
        exposure=arm.exposure,
        turnover=arm.turnover,
        mean_multiplier=float("nan") if capacity is None else capacity.mean_multiplier,
        constrained_share=(
            float("nan") if capacity is None else capacity.constrained_share
        ),
    )


def print_rows(rows: list[CapacityRow]) -> None:
    """打印容量表：mult 是实现倍率，capped 是被容量压住的格子比重。"""
    header = (
        f"{'size':<{LABEL_WIDTH}}{'annual':>9}{'vol':>8}{'sharpe':>8}{'maxDD':>9}"
        f"{'expo':>7}{'turn':>7}{'mult':>7}{'capped':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        mult = "--" if pd.isna(row.mean_multiplier) else f"{row.mean_multiplier:.2f}"
        capped = (
            "--" if pd.isna(row.constrained_share) else f"{row.constrained_share:.1%}"
        )
        print(
            f"{row.label:<{LABEL_WIDTH}}{row.annual_ret:>8.1%}{row.vol:>8.1%}"
            f"{row.sharpe:>8.2f}{row.max_dd:>9.1%}{row.exposure:>7.1%}"
            f"{row.turnover:>7.3f}{mult:>7}{capped:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="账本缩放的容量口径（杠杆额度）研究")
    parser.add_argument(
        "--min-bars", type=int, default=DEFAULT_MIN_BARS, help="入池最少根数"
    )
    parser.add_argument(
        "--participation",
        type=float,
        default=PARTICIPATION,
        help="单标的仓位占滞后中位日成交额的比例",
    )
    parser.add_argument(
        "--notionals",
        type=str,
        default=",".join(str(int(value)) for value in DEFAULT_NOTIONALS),
        help="账本名义规模扫描档（美元，逗号分隔）",
    )
    args = parser.parse_args()

    bars_map = load_bars_dir(SPOT_DIR, args.min_bars)
    raw = engine_runs(bars_map, None, PERP_FEES)
    book = portfolio_returns({stem: run.net_ret for stem, run in raw.items()})
    counts = active_counts({stem: run.net_ret for stem, run in raw.items()})
    target = float(trailing_volatility(bars_map[BTC_STEM]).median())
    scale = book_scale(book, target=target, cap=BOOK_SCALE_CAP)
    print(
        f"池子 {len(bars_map)} 只（≥{args.min_bars} 根）｜缩放目标 = BTC 实测波动"
        f" 中位 {target:.1%}｜参与率 {args.participation:.1%}｜倍率上限"
        f" {MAX_MULTIPLIER:.0f}x"
    )

    rows = [
        record("no scale", raw, None),
        record(
            f"book_scale@{target:.0%} cap3x",
            engine_runs(bars_map, scale, PERP_FEES),
            None,
        ),
    ]
    for text in args.notionals.split(","):
        notional = float(text)
        spec = CapacitySpec(notional=notional, participation=args.participation)
        capacity = capacity_frame(bars_map, counts, scale, spec)
        runs = run_capacity(bars_map, SCALED_OPTIONS, capacity, cost=PERP_FEES)
        full, tiny = capacity.quota_counts()
        print(
            f"  ${notional:,.0f}：拿满 {MAX_MULTIPLIER:.0f}x 的标的 {full}/{len(bars_map)} 只"
            f"｜连 1x 都吃不下的 {tiny} 只"
        )
        rows.append(record(f"cap @ ${notional:,.0f}", runs, capacity))
    print_rows(rows)

    frame = pd.DataFrame([row.__dict__ for row in rows])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"\n落盘：{OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
