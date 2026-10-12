"""K 线缓存验收闸：缺口、重复戳、包络、价格合法性的逐标的扫描。

用法：
    uv run python scripts/audit_crypto_bars.py [--exchange okx] [--market swap] [--timeframe 1m]
    uv run python scripts/audit_crypto_bars.py --timeframe dollar --irregular

致命项（有任意一条则退出码 1）：重复时间戳、时间缺口、包络违反、非正价格。
告警项（不影响退出码）：零成交量根数，1m 冷清时段无成交属正常。

`--irregular` 用于信息驱动采样（如美元棒）：间隔本来就不规则，缺口不是缺陷，
跳过缺口检查。跨源对账与同源复抓由 scripts/crosscheck_crypto_bars.py 负责，
本脚本只管落库结构。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.config.paths import CRYPTO_ROOT

PRICE_COLUMNS = ("open", "high", "low", "close")
SYMBOL_WIDTH = 16


@dataclass(frozen=True)
class AuditResult:
    """单个标的的验收结果。"""

    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    duplicates: int
    gaps: int
    envelope: int
    bad_price: int
    zero_volume: int

    @property
    def fatal(self) -> int:
        """致命项合计：非零即数据集不可用。"""
        return self.duplicates + self.gaps + self.envelope + self.bad_price


def audit_frame(frame: pd.DataFrame, irregular: bool = False) -> AuditResult:
    """结构检查：以步长众数为基准频率，超过一倍即为缺口。

    `irregular=True` 时跳过缺口检查（信息驱动采样的间隔本就不规则），
    但重复戳、包络、非正价格仍然是致命项。
    """
    index = pd.DatetimeIndex(frame.index)
    steps = index.to_series().diff().dropna()
    base = steps.mode().iloc[0] if not steps.empty else pd.Timedelta(0)
    high, low = frame["high"], frame["low"]
    envelope = (
        (high < low)
        | (high < frame["open"])
        | (high < frame["close"])
        | (low > frame["open"])
        | (low > frame["close"])
    )
    bad_price = (frame[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    return AuditResult(
        rows=len(frame),
        start=cast(pd.Timestamp, index[0]),
        end=cast(pd.Timestamp, index[-1]),
        duplicates=int(index.duplicated().sum()),
        gaps=0 if irregular else (int((steps > base).sum()) if base else 0),
        envelope=int(envelope.sum()),
        bad_price=int(bad_price.sum()),
        zero_volume=int((frame["volume"] == 0).sum()),
    )


def audit_file(path: Path, irregular: bool = False) -> AuditResult:
    return audit_frame(pd.read_parquet(path), irregular)


def report(path: Path, result: AuditResult) -> None:
    mark = "FAIL" if result.fatal else "OK  "
    span = f"{result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}"
    print(
        f"{mark} {path.stem:<{SYMBOL_WIDTH}} {result.rows:>11,} 根  {span}  "
        f"缺口 {result.gaps}  重复 {result.duplicates}  "
        f"包络 {result.envelope}  非正价 {result.bad_price}  零量 {result.zero_volume}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="K 线缓存验收闸")
    parser.add_argument("--exchange", default="okx")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument(
        "--irregular",
        action="store_true",
        help="信息驱动采样（如美元棒）：间隔本就不规则，跳过缺口检查",
    )
    args = parser.parse_args()

    dataset_dir = CRYPTO_ROOT / args.exchange / args.market / args.timeframe
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"无数据：{dataset_dir}")

    print(f"验收 {dataset_dir}：{len(files)} 个标的")
    rows_total = 0
    fatal_total = 0
    for path in files:
        result = audit_file(path, args.irregular)
        report(path, result)
        rows_total += result.rows
        fatal_total += result.fatal

    print(f"\n合计 {len(files)} 个标的 / {rows_total:,} 行；致命项 {fatal_total}")
    if fatal_total:
        print("致命项非零：坏行只能重建分区修，clean 层 merge 删不掉旧坏行")
        return 1
    print("结构验收通过；跨源对账另跑 scripts/crosscheck_crypto_bars.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
