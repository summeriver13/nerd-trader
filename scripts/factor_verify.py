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
"""

from __future__ import annotations

import argparse
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

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "factors"
POOL_PLATEAU: Final[str] = "plateau"
POOL_ALL: Final[str] = "all"
# 逐候选跑分位组合，是分钟级的活，进度按这个间隔报一次
PROGRESS_STEP: Final[int] = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收益层验证：分位组合 + PBO/DSR")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument("--market", default="spot", help="市场层，默认 spot")
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
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
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
    result = verify(picks, panel, mask, trials=args.trials, progress=report_progress)
    print_verification(result)
    print(f"\n耗时 {time.time() - started:.0f} 秒")

    if not args.no_save:
        written = save_outputs(
            result, f"{args.market}_{args.timeframe}_{args.pool}"
        )
        print("落盘：")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
