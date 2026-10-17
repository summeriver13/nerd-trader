"""
scripts/factor_verify.py
收益层验证入口：把扫描出的候选接进分位组合，跑 PBO（CSCV）与 DSR。

用法：
    uv run python scripts/factor_verify.py                     # 平台代表窗口（快）
    uv run python scripts/factor_verify.py --pool all           # 全候选池（忠实 PBO，慢）
    uv run python scripts/factor_verify.py --pool all --horizons 1,2,3
    uv run python scripts/factor_verify.py --trials 3276 --no-save

读 factor_sweep_<市场>_<周期>.parquet 与 factor_sweep_plateau_<市场>_<周期>.parquet
（由 factor_sweep.py 产出），只读 data/storage 下的缓存，不触发采集。

两种候选池的区别：
- plateau：每个平台取一个代表窗口，回答「幸存者里最好的那个稳不稳」
- all：扫描空间里的全部候选，回答「这批候选本身是不是一片数据挖掘的沼泽」——
  后者才是忠实的 PBO，因为竞争就是在这几百个之间发生的

PBO 的可行性边界：分块数受「每块最少 60 根」限制，期限 ≥15 日时非重叠期数不够
分出 4 块，会报「不可算」。这不是缺陷——用日线做 30 日重叠检验本来就没有足够
的独立样本。

另外打印一组**记录体检**（de Prado 第 14/15 章）：PSR（夏普高于 0 的概率）、
MinTRL（达到 95% 置信所需的最小观测数，与期数同单位、可直接比大小）、回撤、
逐期价差的集中度，以及策略失败概率。DSR/PBO 问「试这么多次还算不算技能」，
这组数问「这条记录本身厚不厚」——实测两个市场里没有一个截面组合的记录长到
MinTRL，而市场等权买入持有的 PSR 反而更高。
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Final

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.factors.panel import UniverseRule, load_panel, tradable_mask
from src.research.factors.verification import (
    VerificationResult,
    picks_from_candidates,
    picks_from_plateaus,
    print_verification,
    verify,
)
from src.research.strategy_risk import strategy_failure_probability
from src.research.track_record import MIN_TRACK_OBSERVATIONS, track_record
from src.research.trials import ResearchScope
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "factors"
POOL_PLATEAU: Final[str] = "plateau"
POOL_ALL: Final[str] = "all"
# 逐候选跑分位组合，是分钟级的活，进度按这个间隔报一次
PROGRESS_STEP: Final[int] = 40
# 记录体检表列几个候选；失败概率用这个年化目标夏普（低于它就当策略没意义）
TRACK_RECORD_TOP: Final[int] = 10
TARGET_ANNUAL_SHARPE: Final[float] = 1.0
BARS_PER_YEAR: Final[float] = 365.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收益层验证：分位组合 + PBO/DSR")
    add_timeframe_argument(parser)
    add_market_argument(parser, default="spot")
    parser.add_argument(
        "--pool",
        default=POOL_PLATEAU,
        choices=[POOL_PLATEAU, POOL_ALL],
        help="候选池：plateau 只跑平台代表窗口，all 跑全部候选",
    )
    parser.add_argument("--horizons", default=None, help="只跑这些期限，逗号分隔")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="DSR 声明的历史试验次数 N；不声明则退化成本批候选数（会低估）",
    )
    parser.add_argument("--min-listed-days", type=int, default=180)
    parser.add_argument("--min-notional", type=float, default=1e6)
    add_no_save(parser)
    return parser.parse_args()


def load_inputs(market: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读扫描产物：IC 表与平台表。"""
    tag = f"{market}_{timeframe}"
    ic = pd.read_parquet(OUTPUT_DIR / f"factor_sweep_{tag}.parquet")
    plateau = pd.read_parquet(OUTPUT_DIR / f"factor_sweep_plateau_{tag}.parquet")
    return ic, plateau


def report_progress(position: int, total: int, label: str) -> None:
    if position % PROGRESS_STEP == 1 or position == total:
        print(f"  [{position}/{total}] {label}", flush=True)


def print_track_records(result: VerificationResult) -> None:
    """第 14/15 章体检：记录够不够长、收益集中不集中、策略失败概率。"""
    if result.picks.empty:
        return
    ordered = result.picks.sort_values("gross_per_period", ascending=False)
    print(f"\n=== 记录体检（第 14/15 章，按每期价差取前 {TRACK_RECORD_TOP}）===")
    print(
        f"{'候选':>20}{'期数':>7}{'夏普':>8}{'PSR':>7}{'MinTRL':>8}{'够长':>6}"
        f"{'最大回撤':>9}{'正/负集中度':>14}{'失败概率':>9}"
    )
    shown = 0
    for row in ordered.head(TRACK_RECORD_TOP).to_dict(orient="records"):
        series = result.series.get(str(row["key"]))
        if series is None or series.dropna().size < MIN_TRACK_OBSERVATIONS:
            continue
        record = track_record(str(row["key"]), series)
        track = (
            "inf"
            if math.isinf(record.min_track_record)
            else f"{record.min_track_record:.0f}"
        )
        enough = (
            "—"
            if math.isinf(record.min_track_record)
            else ("是" if record.observations >= record.min_track_record else "否")
        )
        bets_per_year = BARS_PER_YEAR / max(int(row["horizon"]), 1)
        risk = strategy_failure_probability(series, bets_per_year, TARGET_ANNUAL_SHARPE)
        print(
            f"{row['key']:>20}{record.observations:>7}{record.sharpe:>+8.3f}"
            f"{record.psr:>7.3f}{track:>8}{enough:>6}"
            f"{record.max_drawdown:>9.1f}"
            f"{record.positive_concentration:>7.3f}/{record.negative_concentration:<6.3f}"
            f"{risk:>9.4f}"
        )
        shown += 1
    if shown:
        print(
            "PSR 是「夏普>0」的概率；MinTRL 是达到 95% 置信所需的最小观测数（与期数同单位）；"
            "集中度接近 1 = 全部盈亏挤压在极少数期次上；失败概率按目标年化夏普 "
            f"{TARGET_ANNUAL_SHARPE:.1f} 算。"
        )


def save_outputs(result: VerificationResult, tag: str) -> list[Path]:
    """把逐候选读数与逐期限裁决写到 data/results/factors/。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        OUTPUT_DIR / f"factor_verify_{tag}.parquet": result.picks,
        OUTPUT_DIR / f"factor_verify_pbo_{tag}.parquet": result.by_horizon,
    }
    written: list[Path] = []
    for path, frame in targets.items():
        if frame.empty:
            continue
        frame.to_parquet(path)
        written.append(path)
    return written


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    rule = UniverseRule(
        min_listed_days=args.min_listed_days, min_notional_usd=args.min_notional
    )
    panel = load_panel(args.timeframe, rule, market=args.market)
    mask = tradable_mask(panel, rule)
    ic, plateau = load_inputs(args.market, args.timeframe)
    horizons = (
        None if not args.horizons else [int(item) for item in args.horizons.split(",")]
    )
    picks = (
        picks_from_candidates(ic, horizons)
        if args.pool == POOL_ALL
        else picks_from_plateaus(plateau)
    )
    print(
        f"市场 {args.market} | 周期 {args.timeframe} | 候选池 {args.pool} | "
        f"候选 {len(picks)} | 池子口径 {rule.describe()}"
    )

    started = time.time()
    # study 用 factor_sweep：平台池是全候选池的子集，同一个试验空间
    scope = ResearchScope(
        study="factor_sweep", market=args.market, timeframe=args.timeframe
    )
    result = verify(
        picks, panel, mask, trials=args.trials, progress=report_progress, scope=scope
    )
    print_verification(result)
    print_track_records(result)
    print(f"\n耗时 {time.time() - started:.0f} 秒")

    if not args.no_save:
        written = save_outputs(result, f"{args.market}_{args.timeframe}_{args.pool}")
        print("落盘：")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
