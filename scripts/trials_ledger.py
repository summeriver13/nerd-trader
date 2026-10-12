"""试验账本：查看诚实的 N，以及按可查证据回填建账之前的历史试验。

默认只打印账本；`--backfill` 先补录再打印。

**账本是下界不是真值。** 建账之前的历史只能回填有据可查的部分：

-「机械可推」的（因子扫描空间由 `SWEEP_FAMILIES` × `DEFAULT_HORIZONS` 定义）按代码枚举，
  这类是精确的；
-「文档记录了参数网格」的（SADF 阶梯、元标签的变体×阈值）按网格录入，出处写在 source 里；
- **更早的零星尝试一律不计**——carry、配对、maker、美元棒、反转那批研究各自试过的参数
  没有留下完整网格，事后重建必然偏小且无法验证。

所以真实 N ≥ 账本数。偏小的方向是安全的一侧：N 偏小只会让门槛偏松，不会让结论看起来
比实际更稳。报告里因此把「实时记录」与「事后补录」分开数。

用法：
    uv run python scripts/trials_ledger.py
    uv run python scripts/trials_ledger.py --backfill
"""

from __future__ import annotations

import argparse
import sys

from src.research.factors.sweep import DEFAULT_HORIZONS, SWEEP_FAMILIES
from src.research.trials import (
    ORIGIN_BACKFILL,
    ORIGIN_RUN,
    Trial,
    TrialLedger,
)

MARKETS: tuple[str, ...] = ("spot", "swap")
TIMEFRAME = "1d"


def sweep_trials() -> list[Trial]:
    """因子扫描空间：机械枚举，精确。"""
    return [
        Trial(
            study="factor_sweep",
            market=market,
            timeframe=TIMEFRAME,
            kind=family.name,
            params={"window": window, "horizon": horizon},
            origin=ORIGIN_BACKFILL,
            source="src/research/factors/sweep.py（SWEEP_FAMILIES × DEFAULT_HORIZONS）",
        )
        for market in MARKETS
        for family in SWEEP_FAMILIES.values()
        for window in family.windows
        for horizon in DEFAULT_HORIZONS
    ]


SADF_WINDOWS: tuple[int, ...] = (30, 40, 60, 90, 120, 180)
SADF_LAGS: tuple[int, ...] = (0, 1, 2)
SADF_HORIZONS: tuple[int, ...] = (1, 3, 5)
SADF_MAIN_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 7, 15, 30)

# 实际跑过的就是这三个口径组合（A 固定/等权、B 三重障碍/等权、C 三重障碍/唯一性）。
# 写成「标签 × 权重」的全叉乘会多算一个没跑过的组合——高估方向同样是缺陷。
META_COMBOS: tuple[tuple[str, str], ...] = (
    ("fixed_label", "weight_uniform"),
    ("barrier_label", "weight_uniform"),
    ("barrier_label", "weight_unique"),
)
META_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.55, 0.6, 0.7)
META_VERTICAL = 3
"""元标签研究的垂直障碍天数。

回填必须带上它，否则与脚本实时记录的身份不同、N 会重复计。
"""


def sadf_trials() -> list[Trial]:
    """SADF：主口径 + 18 变体阶梯。参数网格记在文档里，出处写在 source。"""
    main = {
        (60, 1): SADF_MAIN_HORIZONS,
    }
    ladder = {
        (window, lags): SADF_HORIZONS for window in SADF_WINDOWS for lags in SADF_LAGS
    }
    return [
        Trial(
            study="sadf",
            market="swap",
            timeframe=TIMEFRAME,
            kind="sadf",
            params={"min_window": window, "lags": lags, "horizon": horizon},
            origin=ORIGIN_BACKFILL,
            source="docs/factors/README.md 第十节（参数平台表）",
        )
        for grid in (main, ladder)
        for (window, lags), horizons in grid.items()
        for horizon in horizons
    ]


def meta_trials() -> list[Trial]:
    """元标签：标签口径 × 权重口径 × 阈值。"""
    return [
        Trial(
            study="meta_label",
            market="swap",
            timeframe=TIMEFRAME,
            kind=variant,
            params={
                "weight": weight,
                "threshold": threshold,
                "vertical": META_VERTICAL,
            },
            origin=ORIGIN_BACKFILL,
            source="docs/factors/README.md 第十一节（三条对照表）",
        )
        for variant, weight in META_COMBOS
        for threshold in META_THRESHOLDS
    ]


def backfill(ledger: TrialLedger) -> int:
    return ledger.record([*sweep_trials(), *sadf_trials(), *meta_trials()])


def report(ledger: TrialLedger) -> None:
    total = ledger.count()
    live = ledger.count(origin=ORIGIN_RUN)
    rebuilt = ledger.count(origin=ORIGIN_BACKFILL)
    print(f"\n试验账本 {ledger.path}")
    if total == 0:
        print(
            "  空。先跑 --backfill 建账，或在研究脚本里接线（scope=ResearchScope(...)）。"
        )
        return
    print(f"  累计 N = {total}（实时记录 {live} + 事后补录 {rebuilt}）")
    print("  这是**下界**：建账之前的零星尝试无从重建。\n")
    print(ledger.breakdown().to_string(index=False))
    print("\n按市场：")
    print(ledger.breakdown(("market", "timeframe")).to_string(index=False))
    print(
        "\nDSR/PBO 的 N 默认取这个累计数；窄作用域会让门槛变松，"
        "所以必须由调用方显式声明。\n"
        "只有 scope 为空且未声明 --trials 时才会退回「本批候选数」——那会打印警告。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="试验账本：诚实计数 N")
    parser.add_argument("--backfill", action="store_true", help="先补录已知历史试验")
    parser.add_argument("--path", default=None, help="账本路径（默认仓库内的账本）")
    args = parser.parse_args()

    ledger = TrialLedger() if args.path is None else TrialLedger(args.path)
    if args.backfill:
        added = backfill(ledger)
        print(f"补录完成：新增 {added} 条（重复的会跳过，所以可反复跑）")
    report(ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
