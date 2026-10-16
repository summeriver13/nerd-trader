"""
scripts/factor_smoothing.py
平滑扫描：Alpha101 的两个算子能不能把现有因子推过 taker 成本线。

用法：
    uv run python scripts/factor_smoothing.py                        # 现货，h=1,3
    uv run python scripts/factor_smoothing.py --market swap
    uv run python scripts/factor_smoothing.py --factors rev_3 --operations decay5,decay20
    uv run python scripts/factor_smoothing.py --anchor none           # 复现文档单因子表
    uv run python scripts/factor_smoothing.py --single-anchor --verify  # 再跑 PBO/DSR 判定
口径：分位组合（5 档、多空价差、名义基准）与「平衡成本」全部走 portfolio.py 的同一套
实现，成本默认 10bp/边（OKX 永续 taker 5bp + 滑点 5bp）。锚点默认固定为 0（跨变体同相位），
h>1 再额外跑两个相位（0/h÷3/2h÷3）并把范围带进表里：固定锚点只解决可比性，解决不了相位
敏感性（实测 `ts_rank` 变体换锚点后只多腿净复利能从 −12% 跳到 +31%）；`--single-anchor`
可关掉，`--anchor none` 回到文档单因子表的口径（各自从预热起点起算）。

**这不是验证。** 本表是方法论筛查：40 个变体里挑最大值必然出好东西，判定要显式开
`--verify`——它用与筛选同一口径的序列（只多腿净收益、anchor=0 同一相位）跑 PBO（整组
候选）与 DSR（逐候选），N 默认取候选数 × 调仓档数，也可用 `--trials` 声明更大的挑选空间。
（sweep 空间用 scripts/factor_verify.py；这里管的是派生变体空间，账本不记变体，所以 N
必须由调用方声明，默认就是本次挑选空间的大小。）只读 data/storage/crypto/okx/<market>/<timeframe>/
下的缓存，不触发采集。
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Final, cast

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.factors.library import FactorSpec, resolve_factors
from src.research.factors.panel import UniverseRule, load_panel, tradable_mask
from src.research.factors.portfolio import DEFAULT_ONE_WAY_COST
from src.research.factors.smoothing import (
    GridVerdict,
    ScanConfig,
    scan,
    summarize,
    verify_grid,
)
from src.research.factors.variants import resolve_interaction, resolve_smoothing

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "factors"
KIND_BASE: Final[str] = "基准"
KIND_SMOOTH: Final[str] = "平滑"
KIND_INTERACTION: Final[str] = "交互"
KEY_WIDTH: Final[int] = 24
LABEL_WIDTH: Final[int] = 14
SUMMARY_WIDTH: Final[int] = 18
VERDICT_TOP: Final[int] = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="因子平滑扫描（Alpha101 算子）")
    parser.add_argument("--market", default="spot", help="市场层，默认 spot")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument(
        "--factors", default=None, help="基因子键（逗号分隔），省略取库内全部 8 个"
    )
    parser.add_argument(
        "--operations",
        default=None,
        help="平滑算子（逗号分隔），如 decay10,tsrank20；省略取全部 5 个",
    )
    parser.add_argument(
        "--horizons", default="1,3", help="分位组合调仓周期（逗号分隔），默认 1,3"
    )
    parser.add_argument("--quantiles", type=int, default=5, help="分位档数，默认 5")
    parser.add_argument(
        "--cost-bp",
        type=float,
        default=None,
        help=f"单边成本（bp），默认 {DEFAULT_ONE_WAY_COST * 1e4:.1f}",
    )
    parser.add_argument(
        "--anchor",
        default="0",
        help="none = 各自预热起点；数字 = 固定锚点（跨变体同相位）",
    )
    parser.add_argument(
        "--single-anchor",
        action="store_true",
        help="只跑一个相位（快一倍，但看不出相位敏感：默认 h>1 跑 0/h÷3/2h÷3）",
    )
    parser.add_argument("--min-notional", type=float, default=1_000_000.0)
    parser.add_argument("--min-listed-days", type=int, default=180)
    parser.add_argument(
        "--no-interaction", action="store_true", help="不跑量价交互族（corr_pv_*）"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="另跑收益层判定（PBO/DSR）：只跑 anchor=0，序列用只多腿净收益",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="DSR 声明的试验次数 N（默认 = 候选数 × 调仓档数，即本次挑选空间的大小）",
    )
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[tuple[str, FactorSpec]]:
    """报告行清单：基准 + 平滑变体（+ 量价交互族）。"""
    rows: list[tuple[str, FactorSpec]] = [
        (KIND_BASE, spec) for spec in resolve_factors(args.factors)
    ]
    rows += [
        (KIND_SMOOTH, spec) for spec in resolve_smoothing(args.factors, args.operations)
    ]
    if not args.no_interaction:
        rows += [(KIND_INTERACTION, spec) for spec in resolve_interaction()]
    return rows


def print_table(summary: pd.DataFrame, cost_bp: float) -> None:
    """主表：按调仓周期分组，组内按平衡成本降序；只多腿报中位（相位范围）。"""
    header = (
        f"{'类别':<6}{'因子':<{KEY_WIDTH}}{'h':>3}{'期限':>6}{'期数':>6}{'毛/期':>8}"
        f"{'净/期':>8}{'换手':>7}{'平衡成本':>10}{'只多净复利':>11}{'胜BTC':>6}"
    )
    for horizon in sorted({int(item) for item in summary["horizon"]}):
        group = cast(
            pd.DataFrame, summary[summary["horizon"] == horizon]
        ).sort_values("break_even_bp", ascending=False)
        btc = float(group["btc"].iloc[0])
        print(f"=== 调仓 {horizon} 根（成本线 {cost_bp:.1f}bp/边，BTC 复利 {btc:.1%}）===")
        print(header)
        for row in group.to_dict(orient="records"):
            flag = "是" if row["beats_all"] else ("部分" if row["beats"] else "-")
            print(
                f"{str(row['kind']):<6}{str(row['factor']):<{KEY_WIDTH}}"
                f"{horizon:>3}{int(row['phases']):>6}{int(row['frames']):>6}"
                f"{float(row['gross_bp']):>7.1f}b{float(row['net_bp']):>7.1f}b"
                f"{float(row['turnover']):>7.2f}{float(row['break_even_bp']):>9.1f}bp"
                f"{_leg_range(row):>11}{flag:>6}"
            )
        print()


def _leg_range(row: dict[str, object]) -> str:
    """只多腿净复利：单相位只报中位，多相位报中位（最低~最高）。"""
    median = float(cast(float, row["long_med"]))
    if int(cast(int, row["phases"])) == 1:
        return f"{median:.1%}"
    low, high = float(cast(float, row["long_low"])), float(cast(float, row["long_high"]))
    return f"{median:.1%}（{low:.0%}~{high:.0%}）"


def print_summary(summary: pd.DataFrame) -> None:
    """边际表：以可执行的只多腿净复利（相位中位）排序，基准 → 最好变体。"""
    bases = cast(pd.DataFrame, summary[summary["kind"] == KIND_BASE])
    smooth = cast(pd.DataFrame, summary[summary["kind"] == KIND_SMOOTH])
    print(
        f"=== 平滑的边际（基准 → 最好变体，按只多腿净复利中位）===\n"
        f"{'因子':<{LABEL_WIDTH}}{'h':>3}{'基准只多':>10}{'基准平衡':>10}"
        f"{'最好变体':<{SUMMARY_WIDTH}}{'只多':>10}{'平衡':>10}{'胜BTC':>7}{'相位':>6}"
    )
    for row in bases.to_dict(orient="records"):
        base_key, horizon = str(row["base"]), int(cast(int, row["horizon"]))
        candidates = cast(
            pd.DataFrame,
            smooth[(smooth["base"] == base_key) & (smooth["horizon"] == horizon)],
        )
        if candidates.empty:
            continue
        best = candidates.sort_values("long_med", ascending=False).iloc[0]
        span = f"{float(best['long_high']) - float(best['long_low']):.0%}"
        flag = "是" if best["beats_all"] else ("部分" if best["beats"] else "-")
        print(
            f"{base_key:<{LABEL_WIDTH}}{horizon:>3}"
            f"{float(cast(float, row['long_med'])):>10.1%}"
            f"{float(cast(float, row['break_even_bp'])):>9.1f}bp"
            f"{str(best['factor']):<{SUMMARY_WIDTH}}"
            f"{float(best['long_med']):>10.1%}"
            f"{float(best['break_even_bp']):>9.1f}bp{flag:>7}{span:>6}"
        )


def print_verdicts(
    verdicts: list[GridVerdict], summary: pd.DataFrame, trials: int
) -> None:
    """收益层判定表：逐调仓周期的 PBO，逐候选的 DSR，外加「胜 BTC 的行」的点名。"""
    print(
        f"\n=== 收益层判定（序列 = 只多腿净收益，anchor=0，声明的试验次数 N={trials}）==="
    )
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for verdict in verdicts:
        pbo = "不可算" if verdict.pbo is None else f"{verdict.pbo:.1%}"
        passed = int(cast(pd.Series, verdict.rows["passed"]).sum())
        print(
            f"\n调仓 {verdict.horizon} 根 | 候选 {verdict.candidates} | 观测 {verdict.observations}"
            f" | PBO {pbo} {'PASS' if verdict.pbo_passed else 'FAIL'}"
            f" | 过 DSR {passed}/{len(verdict.rows)}"
        )
        print(f"{'候选':<{KEY_WIDTH}}{'只多净复利':>11}{'夏普':>7}{'DSR':>7}{'裁决':>8}")
        ordered = verdict.rows.sort_values("dsr", ascending=False)
        for row in ordered.head(VERDICT_TOP).to_dict(orient="records"):
            lookup[(int(cast(int, row["horizon"])), str(row["factor"]))] = row
            print(
                f"{str(row['factor']):<{KEY_WIDTH}}{float(cast(float, row['annual'])):>11.1%}"
                f"{float(cast(float, row['sharpe'])):>7.2f}{float(cast(float, row['dsr'])):>7.3f}"
                f"{verdict_flag(row):>8}"
            )
    winners = cast(pd.DataFrame, summary[summary["beats_all"]])
    print("\n胜 BTC 的全部相位候选（相位中位口径）的 DSR：")
    if winners.empty:
        print("  没有。")
    for row in winners.to_dict(orient="records"):
        hit = lookup.get((int(cast(int, row["horizon"])), str(row["factor"])))
        verdict = (
            f"（在 DSR 前{VERDICT_TOP} 名之外）" if hit is None else verdict_flag(hit)
        )
        print(
            f"  {str(row['factor']):<{KEY_WIDTH}}h={int(cast(int, row['horizon']))}"
            f" 只多净复利中位 {float(cast(float, row['long_med'])):>7.1%}  {verdict}"
        )
    print(
        "\n判据：DSR ≥ 0.95 且 PBO < 50% 才算过。N 若小于真实的挑选空间，门槛还是低估。"
    )


def verdict_flag(row: dict[str, object]) -> str:
    """候选裁决：样本不足与未通过要分开——前者不是证据。"""
    if not bool(row["sufficient"]):
        return "样本不足"
    return "PASS" if bool(row["passed"]) else "FAIL"


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cost_one_way = (
        DEFAULT_ONE_WAY_COST if args.cost_bp is None else args.cost_bp / 1e4
    )
    anchor = None if str(args.anchor).lower() == "none" else int(args.anchor)
    config = ScanConfig(
        horizons=tuple(int(item) for item in str(args.horizons).split(",")),
        quantiles=args.quantiles,
        cost_one_way=cost_one_way,
        single_anchor=bool(args.single_anchor),
    )
    rule = UniverseRule(
        min_notional_usd=args.min_notional, min_listed_days=args.min_listed_days
    )
    panel = load_panel(args.timeframe, rule, market=args.market)
    mask = tradable_mask(panel, rule)

    print(
        f"\n平滑扫描 | {args.market} {args.timeframe} | 成本 {cost_one_way * 1e4:.1f}bp/边"
        f" | 锚点 {anchor} | 调仓 {config.horizons}"
    )
    print(
        f"时间 {cast(pd.Timestamp, panel.index[0]).date()}"
        f" -> {cast(pd.Timestamp, panel.index[-1]).date()}"
        f" | 面板 {len(panel.symbols)} 标的"
        f"，可交易中位 {mask.sum(axis=1).median():.0f} 只"
    )

    started = time.time()
    frame = scan(build_specs(args), panel, mask, config, anchor)
    summary = summarize(frame)
    print(f"耗时 {time.time() - started:.1f}s，{len(summary)} 行候选 / {len(frame)} 条读数")
    print_table(summary, cost_one_way * 1e4)
    print_summary(summary)

    if args.verify:
        specs = build_specs(args)
        trials = (
            int(args.trials)
            if args.trials is not None
            else len(specs) * len(config.horizons)
        )
        print(
            f"\n跑收益层判定（{len(specs)} 个候选 × {len(config.horizons)} 个调仓档，单相位）"
        )
        verify_started = time.time()
        verdicts = verify_grid(specs, panel, mask, config, trials)
        print_verdicts(verdicts, summary, trials)
        print(f"判定耗时 {time.time() - verify_started:.1f}s")

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"factor_smoothing_{args.market}_{args.timeframe}.parquet"
        frame.to_parquet(path)
        print(f"\n落盘 {path}")
    print("提醒：主表是筛查不是判定；过线候选要用 --verify 跑 PBO/DSR 才算结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
