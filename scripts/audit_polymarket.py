"""
Polymarket raw 层验收闸：结构 + 数值合法性的逐文件扫描。

用法：
    uv run python scripts/audit_polymarket.py
    uv run python scripts/audit_polymarket.py --stale-days 3

致命项（任意一条即退出码 1）：JSONL 解析失败、业务键重复、价格越界（不在 [0,1]）、
成交量非正、manifest 与磁盘不一致。
告警项（不影响退出码）：文件点数/成交数不足、价格序列过期、二元市场 YES+NO 偏离 1、
孤儿 slug（市场已结算或下架后不再出现在活跃枚举里是正常现象）、成交行数恰为页大小整数倍
（可能还有更早的历史）。

采集中断在这里是**告警而不是致命**：raw 层只增不改，缺口只能靠「当时有没有跑」追溯，
所以过期检查的意义是「提醒回补的机会正在关闭」（价格历史只有约 30 天窗口）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.paths import (
    MANIFEST_ROOT,
    POLYMARKET_EVENTS,
    POLYMARKET_MARKETS,
    POLYMARKET_PRICES,
    POLYMARKET_RAW_ROOT,
    POLYMARKET_TRADES,
    polymarket_dataset_dir,
)
from src.utils.jsonl_store import read_jsonl, row_key
from src.utils.polymarket_store import KEY_FIELDS

DEFAULT_STALE_DAYS = 2
DEFAULT_PAGE_LIMIT = 500
MIN_PRICE_ROWS = 2
DATE_WIDTH = 10
TITLE_WIDTH = 44


@dataclass
class AuditResult:
    """一个数据集的验收结果。"""

    dataset: str
    files: int = 0
    rows: int = 0
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal


def check_duplicates(path: Path, dataset: str) -> list[str]:
    """业务键重复：raw 层一旦重复就是追加逻辑坏了，必须致命。"""
    key_fields = KEY_FIELDS[dataset]
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for record in read_jsonl(path):
        key = row_key(record, key_fields)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return [f"{path.name}: 业务键重复 {duplicates} 行"] if duplicates else []


def check_prices_row(record: dict[str, Any], path: Path) -> str | None:
    price = record.get("p")
    if (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not 0.0 <= float(price) <= 1.0
    ):
        return f"{path.name}: 价格越界 p={price!r}"
    if not isinstance(record.get("t"), int):
        return f"{path.name}: 时间戳不是整数 t={record.get('t')!r}"
    return None


def check_trades_row(record: dict[str, Any], path: Path) -> str | None:
    price = record.get("price")
    if (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not 0.0 < float(price) <= 1.0
    ):
        return f"{path.name}: 成交价越界 price={price!r}"
    size = record.get("size")
    if (
        not isinstance(size, (int, float))
        or isinstance(size, bool)
        or float(size) <= 0.0
    ):
        return f"{path.name}: 成交量非正 size={size!r}"
    return None


def audit_dataset(
    dataset: str,
    *,
    raw_root: Path,
    stale_days: int,
    page_limit: int,
    now: int,
) -> AuditResult:
    """逐文件扫描一个数据集：解析、重复键、数值合法性、过期。"""
    result = AuditResult(dataset)
    directory = polymarket_dataset_dir(dataset, root=raw_root)
    checks = {POLYMARKET_PRICES: check_prices_row, POLYMARKET_TRADES: check_trades_row}
    for path in sorted(directory.glob("*.jsonl")):
        result.files += 1
        try:
            rows = list(read_jsonl(path))
        except ValueError as error:
            result.fatal.append(str(error))
            continue
        result.rows += len(rows)
        result.fatal.extend(check_duplicates(path, dataset))
        role = checks.get(dataset)
        if role is not None:
            problems = [message for record in rows if (message := role(record, path))]
            result.fatal.extend(sorted(set(problems))[:3])
        if dataset == POLYMARKET_PRICES:
            result.warnings.extend(check_binary_pair(path, rows))
        result.warnings.extend(check_file_size(dataset, path, rows, page_limit))
        result.warnings.extend(check_freshness(dataset, path, rows, stale_days, now))
    return result


def check_binary_pair(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    """二元市场语义：同一小时桶里 YES + NO 应当 ≈ 1（实测恒为 1.0000）。

    两个 token 的采样时刻错开十几秒，所以按小时桶对齐而不是按秒。
    """
    buckets: dict[str, dict[int, float]] = {}
    for record in rows:
        price = record.get("p")
        moment = record.get("t")
        if isinstance(price, (int, float)) and isinstance(moment, int):
            buckets.setdefault(str(record.get("outcome")), {})[moment // 3600] = float(
                price
            )
    if len(buckets) != 2:
        return [
            f"{path.name}: 只有 {len(buckets)} 个 outcome（单 token 抓取失败时会如此）"
        ]
    left, right = buckets.values()
    common = set(left) & set(right)
    if not common:
        return [f"{path.name}: 两个 outcome 没有重合的小时桶"]
    deviation = max(abs(left[bucket] + right[bucket] - 1.0) for bucket in common)
    if deviation > 0.02:
        return [f"{path.name}: YES+NO 偏离 1（最大 {deviation:.4f}，{len(common)} 桶）"]
    return []


def check_file_size(
    dataset: str, path: Path, rows: list[dict[str, Any]], page_limit: int
) -> list[str]:
    """样本量与分页截断迹象。"""
    if dataset == POLYMARKET_PRICES and len(rows) < MIN_PRICE_ROWS:
        return [f"{path.name}: 价格点不足（{len(rows)} 行）"]
    if dataset == POLYMARKET_TRADES and len(rows) % page_limit == 0 and rows:
        return [
            f"{path.name}: 成交行数 {len(rows)} 恰为页大小整数倍，可能还有更早的历史"
        ]
    return []


def check_freshness(
    dataset: str, path: Path, rows: list[dict[str, Any]], stale_days: int, now: int
) -> list[str]:
    """价格序列是否还在推进（采集中断会让 30 天窗口无声关闭）。"""
    if dataset != POLYMARKET_PRICES or not rows:
        return []
    latest = max(
        int(record["t"]) for record in rows if isinstance(record.get("t"), int)
    )
    age_days = (now - latest) / 86_400
    if age_days > stale_days:
        return [f"{path.name}: 最新价格点距今 {age_days:.1f} 天"]
    return []


def verify_manifest(
    dataset: str, result: AuditResult, manifest_root: Path, raw_root: Path
) -> None:
    """manifest 是交付物：文件数/行数与磁盘不符即致命。"""
    path = manifest_root / f"polymarket_raw_{dataset}.json"
    if not path.exists():
        result.fatal.append(f"缺少 manifest: {path.name}")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("files") != result.files:
        result.fatal.append(
            f"manifest 文件数 {manifest.get('files')} ≠ 磁盘 {result.files}"
        )
    if manifest.get("rows") != result.rows:
        result.fatal.append(
            f"manifest 行数 {manifest.get('rows')} ≠ 磁盘 {result.rows}"
        )
    if not manifest.get("coverage", {}).get("end"):
        result.warnings.append("manifest 覆盖区间为空（还没有数据）")


def audit_slugs(raw_root: Path, result: AuditResult) -> None:
    """孤儿检查：prices/trades 有文件、但 markets 快照里从未出现过的 slug。"""
    snapshot: set[str] = set()
    for path in sorted(
        polymarket_dataset_dir(POLYMARKET_MARKETS, root=raw_root).glob("*.jsonl")
    ):
        snapshot.update(str(record.get("slug")) for record in read_jsonl(path))
    for dataset in (POLYMARKET_PRICES, POLYMARKET_TRADES):
        orphans = [
            path.stem
            for path in sorted(
                polymarket_dataset_dir(dataset, root=raw_root).glob("*.jsonl")
            )
            if path.stem not in snapshot
        ]
        if orphans:
            result.warnings.append(
                f"{dataset} 里 {len(orphans)} 个 slug 不在任何市场快照中"
            )


def report(results: list[AuditResult]) -> int:
    """打印逐数据集结果，返回退出码。"""
    print(f"{'数据集':<12} {'文件':>5} {'行数':>9} 判定")
    for result in results:
        verdict = "OK" if result.ok else f"致命 {len(result.fatal)}"
        print(
            f"{result.dataset:<12} {result.files:>5} {result.rows:>9,} {verdict}（告警 {len(result.warnings)}）"
        )
        for message in result.fatal[:6]:
            print(f"    ✗ {message}")
        for message in result.warnings[:6]:
            print(f"    ! {message}")
    failed = [result for result in results if not result.ok]
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket raw 层验收闸")
    parser.add_argument("--raw-root", type=Path, default=POLYMARKET_RAW_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    parser.add_argument("--page-limit", type=int, default=DEFAULT_PAGE_LIMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = int(time.time())
    datasets = (
        POLYMARKET_EVENTS,
        POLYMARKET_MARKETS,
        POLYMARKET_PRICES,
        POLYMARKET_TRADES,
    )
    results = [
        audit_dataset(
            dataset,
            raw_root=args.raw_root,
            stale_days=args.stale_days,
            page_limit=args.page_limit,
            now=now,
        )
        for dataset in datasets
    ]
    for result in results:
        verify_manifest(result.dataset, result, args.manifest_root, args.raw_root)
    audit_slugs(args.raw_root, results[0])
    return report(results)


if __name__ == "__main__":
    raise SystemExit(main())
