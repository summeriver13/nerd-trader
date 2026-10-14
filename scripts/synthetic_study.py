"""端到端合成床的读数：真值已知时，整条管道各层判对没有。

用法：``uv run python scripts/synthetic_study.py [--seeds 10] [--days 2000] [--replications 199]``

与 ``scripts/corrections_study.py --stage bed`` 的分工：那一床造的是**候选收益序列**，只量校正
检验自己的尺寸与功效；这里造**面板**（close/volume/notional），因子构造、分位组合、成本、平台
判据都会真的被执行，于是检验的是整条管道。四档床的定义与真值写在 ``src/research/synthetic.py``。

单位口径：IC 是定向后的截面秩相关（正 = 高分位跑赢），钱的口径是 bp/期、默认成本（弱信号档用
30bp/边）。家族级检验的家族 = 七个主池因子（K=7），坐标同 anchor。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.research.corrections import (
    DEFAULT_ALPHA,
    benjamini_hochberg,
    hansen_spa,
    naive_p_values,
    reality_check,
)
from src.research.synthetic import BEDS, SyntheticSpec, run_bed

RESULTS_DIR: Final[Path] = Path("data/results/breaks")
DEFAULT_SEEDS: Final[int] = 10
DEFAULT_DAYS: Final[int] = 2000
DEFAULT_REPLICATIONS: Final[int] = 199
FAMILY_TESTS: Final[tuple[str, ...]] = ("naive", "BH", "RC", "SPA")
WATCH: Final[tuple[str, ...]] = ("rev_3", "mom_7", "vol_30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端合成床：真值已知时各层判对没有")
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="每档床的种子数"
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="面板根数")
    parser.add_argument(
        "--replications", type=int, default=DEFAULT_REPLICATIONS, help="bootstrap 次数"
    )
    parser.add_argument("--no-save", action="store_true", help="不落盘")
    return parser.parse_args()


def family_verdict(
    spreads: pd.DataFrame, replications: int, seed: int
) -> dict[str, float]:
    """家族级（K=7 个主池因子）四种做法是否拒绝：1 = 拒绝。"""
    p_values = naive_p_values(spreads)
    bh = benjamini_hochberg(p_values, alpha=DEFAULT_ALPHA)
    check = reality_check(spreads, replications=replications, seed=seed)
    spa = hansen_spa(spreads, replications=replications, seed=seed)
    return {
        "naive": float(bool(float(p_values.min()) <= DEFAULT_ALPHA)),
        "BH": float(len(bh.labels) > 0),
        "RC": float(bool(check.p_value <= DEFAULT_ALPHA)),
        "SPA": float(bool(spa.p_consistent <= DEFAULT_ALPHA)),
    }


def seed_row(
    spec: SyntheticSpec, seed: int, days: int, replications: int
) -> dict[str, object]:
    """一颗种子上的全部读数：家族级裁决 + 逐因子 IC/净 + 逐条真值核对。"""
    bed = run_bed(spec, seed, days=days)
    ic = bed.ic.set_index("因子")["IC"]
    money = bed.money.set_index("因子")
    row: dict[str, object] = {
        "床": spec.key,
        "种子": float(seed),
        "期数": float(money["期数"].iloc[0]),
        **family_verdict(bed.spreads, replications, seed),
    }
    for key in ic.index:
        row[f"IC:{key}"] = float(ic.loc[key])
        row[f"净:{key}"] = float(money.loc[key, "净(bp)"])
    for check in bed.checks():
        row[f"核对:{check.layer}:{check.subject}"] = float(check.hit)
    return row


def report(frame: pd.DataFrame, seeds: int, days: int) -> None:
    """逐档打印：真值核对通过率、家族级拒绝率、关注因子的读数中位。"""
    print(f"\n=== 端到端合成床（{seeds} 颗种子 × {days} 根 × 60 只）===")
    for key in frame["床"].unique():
        block = frame[frame["床"] == key]
        print(f"\n--- {key} ---")
        layers = sorted(
            {
                column.split(":")[1]
                for column in block.columns
                if column.startswith("核对:")
            }
        )
        for layer in layers:
            # 一档床只覆盖部分（层, 因子）组合，其余档的列在这里全 NaN——不打印它们，
            # 否则报告里会冒出一串没有意义的 nan%
            columns = [
                column
                for column in block.columns
                if column.startswith(f"核对:{layer}:")
                and bool(pd.Series(block[column]).notna().any())
            ]
            if not columns:
                continue
            hits = float(np.asarray(block[columns]).mean())
            detail = " ".join(
                f"{column.rsplit(':', 1)[1]}={float(block[column].mean()):.0%}"
                for column in columns
            )
            print(f"  真值核对·{layer:<4} 通过 {hits:.0%}   {detail}")
        rates = " ".join(
            f"{name}={float(block[name].mean()):.2f}" for name in FAMILY_TESTS
        )
        print(f"  家族级拒绝率（K=7）  {rates}")
        for subject in WATCH:
            column = f"IC:{subject}"
            if column in block:
                ic = float(pd.Series(block[column]).median())
                net = float(pd.Series(block[f"净:{subject}"]).median())
                print(f"  {subject:<7} IC 中位 {ic:+.4f}｜净中位 {net:+.1f}bp")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for spec in BEDS:
        for seed in range(args.seeds):
            rows.append(seed_row(spec, seed, args.days, args.replications))
            print(f"  完成 {spec.key} seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    report(frame, args.seeds, args.days)
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "synthetic_bed.parquet"
        frame.to_parquet(path, index=False)
        print(f"\n读数落盘 {path}")


if __name__ == "__main__":
    main()
