"""
美元棒 vs 时间棒：反转信号活在日历尺度还是信息尺度？

现有全部结论都建立在时间棒上——「3 根反转」是 3 个**日历日**。美元棒按成交额
采样，同一个窗口号是 3 个**信息单位**。实测每根美元棒的日历跨度 p5→p95 差
34.7 倍（0.07 天 → 3.8 天），所以两者是真正不同的尺度，而不是同一件事换个说法。

方法：两边用**同一份前向收益**（日线日历 h 日）与**同一个可交易掩码**，只有因子值
的度量方式不同——日线收盘算的 w 日收益 vs 美元棒收盘算的 w 根收益，后者再按日
采样。采样只取「当日收盘之前已收盘的最后一根美元棒」，无前视。

用法：
    uv run python scripts/dollar_bar_study.py [--market swap] [--horizon 5] [--no-save]

产物落 data/results/bars/dollar_vs_time.parquet。只读本地缓存，不联网。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import cast

import pandas as pd

from src.config.paths import CRYPTO_ROOT, RESULTS_ROOT
from src.research.bar_factors import (
    calendar_reversal,
    ic_row,
    keep_symbols,
    load_bar_closes,
    rescaled_reversal,
)
from src.research.factors.evaluation import forward_returns
from src.research.factors.panel import UniverseRule, load_panel, tradable_mask

DEFAULT_WINDOWS = (1, 2, 3, 5, 10, 20, 30)
DEFAULT_HORIZONS = (1, 3, 5, 10, 20, 30)
SCALE_CALENDAR = "日历全池"
SCALE_CALENDAR_SHARED = "日历同池"
SCALE_INFORMATION = "信息同池"
DOLLAR_TIMEFRAME = "dollar"


def main() -> int:
    parser = argparse.ArgumentParser(description="美元棒 vs 时间棒的反转尺度对比")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--min-notional", type=float, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    windows = [int(part) for part in args.windows.split(",") if part]
    horizons = [int(part) for part in args.horizons.split(",") if part]
    rule = (
        UniverseRule()
        if args.min_notional is None
        else UniverseRule(min_notional_usd=args.min_notional)
    )

    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel, rule)
    closes = load_bar_closes(CRYPTO_ROOT / "okx" / args.market / DOLLAR_TIMEFRAME)
    shared = [sym for sym in panel.symbols if sym in closes]
    print(
        f"美元棒 vs 时间棒 | {args.market} · {args.timeframe} | "
        f"日线 {len(panel.symbols)} 只，美元棒 {len(closes)} 只，交集 {len(shared)} 只"
    )

    started = time.time()
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        forward = forward_returns(panel, horizon)
        for window in windows:
            for scale, values in (
                (SCALE_CALENDAR, calendar_reversal(panel.close, window)),
                (
                    SCALE_CALENDAR_SHARED,
                    keep_symbols(calendar_reversal(panel.close, window), shared),
                ),
                (
                    SCALE_INFORMATION,
                    rescaled_reversal(closes, panel.index, window),
                ),
            ):
                rows.append(
                    ic_row(
                        scale,
                        window,
                        horizon,
                        values,
                        forward,
                        keep_symbols(mask, shared) if scale != SCALE_CALENDAR else mask,
                    )
                )
        print(
            f"  期限 {horizon:>3} 日完成（{time.time() - started:.0f} 秒）", flush=True
        )

    table = pd.DataFrame(rows)
    print(_report(table))
    if not args.no_save:
        target = RESULTS_ROOT / "bars"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "dollar_vs_time.parquet"
        table.to_parquet(path)
        print(f"\n落盘 {path}")
    return 0


def _report(table: pd.DataFrame) -> str:
    """并排打印三个口径：全池日历、同池日历、同池信息。

    信息尺度的池子只有 30 只（1m 缓存所限），所以必须多给一列**同池日历**——
    否则「信息 IC 更弱」可能只是池子换成了大币，与尺度无关。
    """
    lines = ["\n=== 反转因子的 IC（同前向收益、同掩码，只换度量尺度）==="]
    for horizon, group in table.groupby("horizon"):
        pivot = group.pivot_table(
            index="window", columns="scale", values="mean_ic", aggfunc="first"
        )
        counts = group.pivot_table(
            index="window", columns="scale", values="periods", aggfunc="first"
        )
        lines.append(f"\n前向 {cast(int, horizon)} 日")
        lines.append(
            f"{'窗口':>6}{'全池日历':>11}{'同池日历':>11}{'同池信息':>11}"
            f"{'信息−日历':>12}{'期数':>8}"
        )
        for window in pivot.index:
            full = float(pivot.loc[window, SCALE_CALENDAR])
            cal = float(pivot.loc[window, SCALE_CALENDAR_SHARED])
            info = float(pivot.loc[window, SCALE_INFORMATION])
            lines.append(
                f"{int(window):>6}{full:>11.4f}{cal:>11.4f}{info:>11.4f}"
                f"{info - cal:>12.4f}"
                f"{int(counts.loc[window, SCALE_INFORMATION]):>8}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
