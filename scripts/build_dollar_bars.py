"""
用本地 1m 缓存构建「等量棒」：美元棒与量棒（de Prado Ch 2）。

每根棒累积同样多的累积变量（成交额或成交量），采样频率随活跃度变化。门槛按「每天一根」
折算，于是根数与日线时间棒可比——同一个窗口号在两套数据上分别是「信息尺度」与「日历
尺度」，IC 差异直接回答信号活在哪个尺度上。

用法：
    uv run python scripts/build_dollar_bars.py [--kind volume] [--market swap] [--symbols A,B]

落盘 data/storage/crypto/<exchange>/<market>/<kind>/<SYMBOL>.parquet，列同 K 线（美元棒
多一列 notional）。只读本地缓存，不联网。tick 棒的累积变量是成交笔数，要求源数据带
`trades` 列（OKX 的 K 线没有，见 scripts/fetch_binance_klines.py），故此处只支持美元
与量两种累积变量。构建后跑：
    uv run python scripts/audit_crypto_bars.py --timeframe dollar --irregular
    uv run python scripts/build_crypto_manifests.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.config.paths import CRYPTO_ROOT, crypto_dataset_dir
from src.research.sampling.bars import (
    ACCUMULATION_NOTIONAL,
    ACCUMULATION_VOLUME,
    BAR_COLUMNS,
    MIN_BAR_FILL_RATIO,
    BarOptions,
    accumulation_weights,
    dollar_bars,
    threshold_for_daily_bars,
    volume_bars,
)
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

KIND_DOLLAR = "dollar"
KIND_VOLUME = "volume"
# 每种棒：构造器 + 累积变量名。两者必须一致，否则「装满门槛」的不变量会看错列
BUILDERS: dict[str, tuple[Callable[[pd.DataFrame, BarOptions], pd.DataFrame], str]] = {
    KIND_DOLLAR: (dollar_bars, ACCUMULATION_NOTIONAL),
    KIND_VOLUME: (volume_bars, ACCUMULATION_VOLUME),
}
SYMBOL_WIDTH = 16


def _source_symbols(dataset_dir: Path, wanted: list[str] | None) -> list[str]:
    stems = sorted(path.stem for path in dataset_dir.glob("*.parquet"))
    if not wanted:
        return stems
    keep = {name.upper() for name in wanted}
    return [stem for stem in stems if stem in keep]


def build_symbol(
    source: Path, target: Path | None, kind: str
) -> tuple[int, int, float]:
    """单标的：源周期 → 等量棒。返回 (源根数, 棒数, 门槛)。target 为 None 时不落盘。"""
    builder, variable = BUILDERS[kind]
    frame = pd.read_parquet(source, columns=list(BAR_COLUMNS))
    threshold = threshold_for_daily_bars(frame, variable)
    bars = builder(frame, BarOptions(threshold=threshold))
    if bars.empty:
        raise ValueError(f"{source.stem}: 门槛 {threshold:.3e} 下没有产出任何棒")
    # 等量棒的定义性不变量：除末根外每根都要装满接近门槛的累积量。
    # 例外：单根源棒自己就超过门槛时，它会把下一组的起点抬过门槛，使前一组天然偏空
    # 且 id 被跳过——这不是分组错。成交量比成交额尖得多（SAND 有单分钟量达门槛
    # 1.55 倍的记录，全在价格跌到 0.07 之后的尾部），故此时只报告不报错。
    floor = float(bars[variable].iloc[:-1].min()) if len(bars) > 1 else threshold
    spike = float(accumulation_weights(frame, variable).max())
    if floor < threshold * MIN_BAR_FILL_RATIO and spike < threshold:
        raise ValueError(
            f"{source.stem}: 有棒只装了 {floor:.3e}，不足门槛 {threshold:.3e} 的"
            f" {MIN_BAR_FILL_RATIO:.0%}，分组算错了"
        )
    if floor < threshold * MIN_BAR_FILL_RATIO:
        print(
            f"  注意 {source.stem}: 最空一根只装 {floor / threshold:.1%}，"
            f"但单根源棒最大 {spike:.3e} ≥ 门槛——尖峰溢出，非分组错",
            flush=True,
        )
    if target is not None:
        bars.to_parquet(target)
    return len(frame), len(bars), threshold


def main() -> int:
    parser = argparse.ArgumentParser(description="构建等量棒（美元棒/量棒）")
    parser.add_argument("--kind", default=KIND_DOLLAR, choices=sorted(BUILDERS))
    parser.add_argument("--exchange", default="okx")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser, default="1m", help_text="源周期，默认 1m")
    parser.add_argument("--symbols", default="", help="逗号分隔，缺省全量")
    add_no_save(parser)
    args = parser.parse_args()

    wanted = [part for part in args.symbols.split(",") if part] or None
    _, variable = BUILDERS[args.kind]
    source_dir = crypto_dataset_dir(args.exchange, args.market, args.timeframe)
    target_dir = CRYPTO_ROOT / args.exchange / args.market / args.kind
    if not args.no_save:
        target_dir.mkdir(parents=True, exist_ok=True)
    symbols = _source_symbols(source_dir, wanted)
    if not symbols:
        print(f"没有找到源数据：{source_dir}")
        return 1

    print(
        f"{args.kind} 棒 | {args.exchange} · {args.market} | {len(symbols)} 只 | "
        f"源 {args.timeframe} → {args.kind} | 累积变量 {variable}"
    )
    print(
        f"{'标的':<{SYMBOL_WIDTH}}{'源根数':>12}{'棒数':>10}{'压缩比':>9}{'门槛':>14}"
    )
    started = time.time()
    total_minutes = total_bars = 0
    for stem in symbols:
        target = None if args.no_save else target_dir / f"{stem}.parquet"
        minutes, count, threshold = build_symbol(
            source_dir / f"{stem}.parquet", target, args.kind
        )
        total_minutes += minutes
        total_bars += count
        print(
            f"{stem:<{SYMBOL_WIDTH}}{minutes:>12,}{count:>10,}"
            f"{minutes / max(count, 1):>8.0f}x{threshold:>14.3e}",
            flush=True,
        )

    if not args.no_save:
        print(f"\n合计 {total_bars:,} 根 {args.kind} 棒（源 {total_minutes:,} 根源棒）")
        print(f"落盘目录 {target_dir}")
    print(f"耗时 {time.time() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
