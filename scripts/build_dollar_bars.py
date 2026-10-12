"""
用本地 1m 缓存构建美元棒（de Prado Ch 2）。

每根棒累积同样多的美元成交额，采样频率随活跃度变化。门槛按「每天一根」折算，
于是根数与日线时间棒可比——同一个窗口号在两套数据上分别是「信息尺度」与
「日历尺度」，IC 差异直接回答信号活在哪个尺度上。

用法：
    uv run python scripts/build_dollar_bars.py [--market swap] [--symbols A,B] [--no-save]

落盘 data/storage/crypto/okx/<market>/dollar/<SYMBOL>.parquet，列同 K 线外加 notional。
只读本地缓存，不联网。构建后跑：
    uv run python scripts/audit_crypto_bars.py --timeframe dollar --irregular
    uv run python scripts/build_crypto_manifests.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.config.paths import CRYPTO_ROOT, crypto_dataset_dir
from src.research.bars import (
    BAR_COLUMNS,
    MIN_BAR_FILL_RATIO,
    DollarBarOptions,
    dollar_bars,
    threshold_for_daily_bars,
)

SOURCE_TIMEFRAME = "dollar"
SYMBOL_WIDTH = 16


def _source_symbols(dataset_dir: Path, wanted: list[str] | None) -> list[str]:
    stems = sorted(path.stem for path in dataset_dir.glob("*.parquet"))
    if not wanted:
        return stems
    keep = {name.upper() for name in wanted}
    return [stem for stem in stems if stem in keep]


def build_symbol(source: Path, target: Path | None) -> tuple[int, int, float]:
    """单标的：1m → 美元棒。返回 (1m 根数, 棒数, 门槛)。target 为 None 时不落盘。"""
    frame = pd.read_parquet(source, columns=list(BAR_COLUMNS))
    threshold = threshold_for_daily_bars(frame)
    bars = dollar_bars(frame, DollarBarOptions(threshold=threshold))
    if bars.empty:
        raise ValueError(f"{source.stem}: 门槛 {threshold:.3e} 下没有产出任何棒")
    # 美元棒的定义性不变量：除末根外每根都要装满接近门槛的成交额
    floor = float(bars["notional"].iloc[:-1].min()) if len(bars) > 1 else threshold
    if floor < threshold * MIN_BAR_FILL_RATIO:
        raise ValueError(
            f"{source.stem}: 有棒只装了 {floor:.3e}，不足门槛 {threshold:.3e} 的"
            f" {MIN_BAR_FILL_RATIO:.0%}，分组算错了"
        )
    if target is not None:
        bars.to_parquet(target)
    return len(frame), len(bars), threshold


def main() -> int:
    parser = argparse.ArgumentParser(description="构建美元棒")
    parser.add_argument("--exchange", default="okx")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1m", help="源周期")
    parser.add_argument("--symbols", default="", help="逗号分隔，缺省全量")
    parser.add_argument("--no-save", action="store_true", help="只统计不落盘")
    args = parser.parse_args()

    wanted = [part for part in args.symbols.split(",") if part] or None
    source_dir = crypto_dataset_dir(args.exchange, args.market, args.timeframe)
    target_dir = CRYPTO_ROOT / args.exchange / args.market / SOURCE_TIMEFRAME
    if not args.no_save:
        target_dir.mkdir(parents=True, exist_ok=True)
    symbols = _source_symbols(source_dir, wanted)
    if not symbols:
        print(f"没有找到源数据：{source_dir}")
        return 1

    print(
        f"美元棒 | {args.exchange} · {args.market} | {len(symbols)} 只 | "
        f"源 {args.timeframe} → {SOURCE_TIMEFRAME}"
    )
    print(
        f"{'标的':<{SYMBOL_WIDTH}}{'1m 根数':>12}{'美元棒':>10}{'压缩比':>9}{'门槛($)':>14}"
    )
    started = time.time()
    total_minutes = total_bars = 0
    for stem in symbols:
        target = None if args.no_save else target_dir / f"{stem}.parquet"
        minutes, count, threshold = build_symbol(source_dir / f"{stem}.parquet", target)
        total_minutes += minutes
        total_bars += count
        print(
            f"{stem:<{SYMBOL_WIDTH}}{minutes:>12,}{count:>10,}"
            f"{minutes / max(count, 1):>8.0f}x{threshold:>14.3e}",
            flush=True,
        )

    if not args.no_save:
        print(f"\n合计 {total_bars:,} 根美元棒（源 {total_minutes:,} 根 1m）")
        print(f"落盘目录 {target_dir}")
    print(f"耗时 {time.time() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
