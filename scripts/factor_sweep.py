"""
scripts/factor_sweep.py
因子扫描器入口：按族扫窗口，用平台判据筛出对规格稳健的效应。

用法：
    uv run python scripts/factor_sweep.py
    uv run python scripts/factor_sweep.py --market swap --families mom
    uv run python scripts/factor_sweep.py --horizons 1,3,7 --no-save

只做信号层（IC + 平台判定），不算组合收益与成本——发现要便宜、验证才贵；
平台筛出的窗口再交给 factor_suite.py 做收益层。
只读 data/storage/crypto/okx/<market>/<timeframe>/ 下的缓存，不触发采集。
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.factors.panel import (
    Panel,
    UniverseRule,
    load_panel,
    tradable_counts,
    tradable_mask,
)
from src.research.factors.plateau import KIND_PLATEAU, KIND_SPIKE
from src.research.factors.sweep import (
    DEFAULT_ALPHA,
    DEFAULT_HORIZONS,
    SweepResult,
    resolve_families,
    sweep,
)

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "factors"
# 曲线抽样点：窗口 1..120 太宽，等距采点才看得见形状
CURVE_WINDOWS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10, 20, 30, 45, 60, 80, 100, 120)
WIDTH_FAMILY: Final[int] = 6
WIDTH_HORIZON: Final[int] = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="因子扫描器：按族扫窗口 + 平台判据")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument("--market", default="spot", help="市场层，默认 spot")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="前向期限，逗号分隔，默认 1,2,3,5,7,15,30",
    )
    parser.add_argument(
        "--families", default=None, help="只扫这些族，逗号分隔，默认全部"
    )
    parser.add_argument("--min-window", type=int, default=None, help="窗口下界覆盖")
    parser.add_argument("--max-window", type=int, default=None, help="窗口上界覆盖")
    parser.add_argument(
        "--min-plateau", type=int, default=3, help="连续多少窗口才算平台，默认 3"
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="显著性水平")
    parser.add_argument("--min-listed-days", type=int, default=180)
    parser.add_argument("--min-notional", type=float, default=1e6)
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def print_header(panel: Panel, mask: pd.DataFrame, args: argparse.Namespace) -> None:
    """池子口径先行：所有结论都建立在它上面。"""
    counts = tradable_counts(mask)
    print("=== 池子 ===")
    print(
        f"市场 {args.market} | 周期 {args.timeframe} | "
        f"标的 {len(panel.symbols)} | 期次 {len(panel.index)}"
    )
    print(f"截面宽度 中位 {counts.median():.0f} | 峰值 {counts.max():.0f}")


def print_thresholds(result: SweepResult) -> None:
    """门槛表：名义 Bonferroni 是诚实基线，有效试验数只是描述性参考。

    有效试验数由样本内 IC 相关矩阵导出，而 de Prado & Fabozzi (2026) 证明搜索-选择
    过程下的 FDR 无法从样本内统计量识别；它的作用方向是放松门槛，即低估假发现率。
    决策用 `src/research/overfitting.py` 的 PBO/DSR。
    """
    print("\n=== 门槛（名义 Bonferroni 为基线；有效门槛仅供参考，无理论保证）===")
    print(
        f"{'期限':>4}{'名义试验数':>12}{'有效试验数':>12}{'名义门槛':>10}{'有效门槛':>12}"
    )
    for row in result.tests.to_dict(orient="records"):
        print(
            f"{row['horizon']:>4}{row['nominal_tests']:>12}"
            f"{row['effective_tests']:>12.1f}"
            f"{row['bonferroni_t']:>10.2f}{row['critical_t']:>12.2f}"
        )


def print_curves(result: SweepResult, windows: tuple[int, ...] = CURVE_WINDOWS) -> None:
    """族 × 期限的 IC 曲线抽样：读的是整段形状，不是单点极值。"""
    print(f"\n=== IC 曲线（抽样 {len(windows)} 个窗口）===")
    print(
        f"{'族':<{WIDTH_FAMILY}}{'h':>{WIDTH_HORIZON}}"
        + "".join(f"{window:>8}" for window in windows)
    )
    for key, group in result.ic.groupby(["family", "horizon"]):
        family, horizon = cast(tuple[str, int], key)
        curve = cast(pd.Series, group.set_index("window")["ic"])
        cells = "".join(f"{_cell(curve, window):>8}" for window in windows)
        print(f"{family:<{WIDTH_FAMILY}}{horizon:>{WIDTH_HORIZON}}{cells}")


def _cell(curve: pd.Series, window: int) -> str:
    """曲线上的一个格子：没算到该窗口时留横杠，不留 0（0 和「没有」不是一回事）。"""
    value = curve.get(window)
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.3f}"


def print_plateaus(result: SweepResult) -> None:
    """平台与尖峰分开列：判据是「只认平台、尖峰一律丢」。"""
    frame = result.plateaus
    real = cast(pd.DataFrame, frame[frame["kind"] == KIND_PLATEAU])
    spikes = cast(pd.DataFrame, frame[frame["kind"] == KIND_SPIKE])
    print(f"\n=== 平台（连续 ≥3 窗口同号且过门槛）：{len(real)} 段 ===")
    if real.empty:
        print("  （无）——所有显著都是孤立尖峰，按判据一律丢弃")
    else:
        print(
            f"{'族':<{WIDTH_FAMILY}}{'h':>4}{'区间':>13}{'跨度':>6}"
            f"{'平台内IC':>11}{'最低|t|':>10}"
        )
        ordered = real.sort_values("windows", ascending=False)
        for row in ordered.to_dict(orient="records"):
            span = f"{row['start']}..{row['end']}"
            print(
                f"{row['family']:<{WIDTH_FAMILY}}{row['horizon']:>4}{span:>13}"
                f"{row['windows']:>6}{row['mean_ic']:>+11.4f}{row['min_abs_t']:>10.2f}"
            )
    print(f"\n=== 尖峰（孤立显著，丢弃）：{len(spikes)} 段 ===")
    if spikes.empty:
        print("  （无）")
    else:
        for row in spikes.to_dict(orient="records"):
            print(
                f"  {row['family']}_{row['start']}@{row['horizon']}"
                f"  IC {row['mean_ic']:+.4f}  |t| {row['min_abs_t']:.2f}"
            )


def print_verdicts(result: SweepResult) -> None:
    """逐族逐期限的裁决：没有平台就是不成立。"""
    print("\n=== 裁决（逐族逐期限）===")
    if result.verdicts.empty:
        print("  没有任何平台——这批族在这批期限上全部不成立")
        return
    print(
        f"{'族':<{WIDTH_FAMILY}}{'h':>4}{'平台段数':>10}{'最长跨度':>10}{'平均IC':>10}"
    )
    for row in result.verdicts.to_dict(orient="records"):
        print(
            f"{row['family']:<{WIDTH_FAMILY}}{row['horizon']:>4}{row['plateaus']:>10}"
            f"{row['longest']:>10}{row['mean_ic']:>+10.4f}"
        )


def save_outputs(
    result: SweepResult, tag: str, out_dir: Path = OUTPUT_DIR
) -> list[Path]:
    """把扫描表、平台表、门槛表写到 data/results/factors/。

    文件名带市场与周期：两个市场各跑一遍时不能互相覆盖（这是真踩过的坑）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        f"factor_sweep_{tag}": result.ic,
        f"factor_sweep_plateau_{tag}": result.plateaus,
        f"factor_sweep_tests_{tag}": result.tests,
    }
    paths: list[Path] = []
    for name, frame in tables.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    horizons = tuple(int(item) for item in str(args.horizons).split(","))
    families = resolve_families(args.families)
    if args.min_window is not None or args.max_window is not None:
        low = args.min_window if args.min_window is not None else 0
        high = args.max_window if args.max_window is not None else 10**9
        families = [
            replace(f, windows=tuple(w for w in f.windows if low <= w <= high))
            for f in families
        ]

    rule = UniverseRule(
        min_listed_days=args.min_listed_days, min_notional_usd=args.min_notional
    )
    panel = load_panel(args.timeframe, rule, market=args.market)
    mask = tradable_mask(panel, rule)
    print_header(panel, mask, args)

    started = time.time()
    result = sweep(
        panel,
        mask,
        horizons=horizons,
        families=families,
        min_plateau=args.min_plateau,
        alpha=args.alpha,
    )
    print_thresholds(result)
    print_curves(result)
    print_plateaus(result)
    print_verdicts(result)

    if not args.no_save:
        print("\n落盘：")
        for path in save_outputs(result, f"{args.market}_{args.timeframe}"):
            print(f"  {path}")
    print(f"\n耗时 {time.time() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
