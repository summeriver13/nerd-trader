"""OKX 强平事件流的验收闸：致命项退出码 1，告警项只打印。

致命项都是「数据不信」的量：JSONL 解析失败、业务键重复、缺核心字段、方向不在枚举、
`instId` 不是 USDT 永续、价格/张数/面值/名义非正、`notional ≠ sizeCcy × bkPx`、
时间戳在未来或早于 2020 年、manifest 与磁盘行数不符。

告警项是「数据可能不全」的量：缺口记录、最新事件过期（轮询器没在跑）、单笔名义异常
大、某日文件覆盖零点附近（分区边界需要人看一眼）。

用法：

    uv run python scripts/audit_liquidation.py
    uv run python scripts/audit_liquidation.py --stale-hours 2 --big-notional 20000000
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.paths import CRYPTO_ROOT, MANIFEST_ROOT, liquidation_dir
from src.utils.data_sources.okx_liquidation import POS_SIDES, SIDES
from src.utils.jsonl_store import read_jsonl, row_key
from src.utils.liquidation_store import DATASET, KEY_FIELDS

DEFAULT_STALE_HOURS = 4.0
DEFAULT_BIG_NOTIONAL = 20_000_000.0
NOTIONAL_TOLERANCE = 1e-6
MIN_TS_MS = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
FUTURE_SLACK_MS = 5 * 60 * 1000
REQUIRED_FIELDS = ("instId", "ts", "side", "posSide", "bkPx", "sz", "ctVal", "notional")


class Findings:
    """致命与告警分开记（致命计数决定退出码）。"""

    def __init__(self) -> None:
        self.fatal: list[str] = []
        self.warnings: list[str] = []

    def report(self) -> int:
        for message in self.fatal:
            print(f"  ! {message}")
        for message in self.warnings:
            print(f"  ~ {message}")
        print(f"\n致命 {len(self.fatal)} 项，告警 {len(self.warnings)} 项")
        return 1 if len(self.fatal) > 0 else 0


def check_row(record: dict[str, Any], label: str, now_ms: int) -> None:
    """单行检查：字段齐全、方向合法、数值物理可信、名义与「张数×面值×价格」自洽。"""
    for field in REQUIRED_FIELDS:
        if record.get(field) in (None, ""):
            raise ValueError(f"{label} 缺字段 {field}")
    if record["side"] not in SIDES:
        raise ValueError(f"{label} 未知 side={record['side']!r}")
    if record["posSide"] not in POS_SIDES:
        raise ValueError(f"{label} 未知 posSide={record['posSide']!r}")
    if not str(record["instId"]).endswith("-USDT-SWAP"):
        raise ValueError(f"{label} instId 不是 USDT 永续：{record['instId']!r}")
    for field in ("bkPx", "ctVal", "notional"):
        if not isinstance(record[field], (int, float)) or record[field] <= 0:
            raise ValueError(f"{label} {field} 非正：{record[field]!r}")
    size = float(record["sz"])
    if size <= 0:
        raise ValueError(f"{label} sz 非正：{record['sz']!r}")
    expected = size * float(record["ctVal"]) * float(record["bkPx"])
    if abs(expected - float(record["notional"])) > NOTIONAL_TOLERANCE * expected:
        raise ValueError(f"{label} notional 与 sz×ctVal×bkPx 不符：{expected:.6f}")
    ts = record["ts"]
    if not isinstance(ts, int) or ts < MIN_TS_MS:
        raise ValueError(f"{label} ts 异常：{ts!r}")
    if ts > now_ms + FUTURE_SLACK_MS:
        raise ValueError(f"{label} ts 在未来：{ts!r}")


def check_file(
    path: Path, findings: Findings, big: float
) -> tuple[int, int, int | None, int | None]:
    """单个日文件：逐行校验 + 去重键 + 覆盖区间；返回 (行数, 家数, 最早, 最晚)。"""
    rows = 0
    families: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    oldest: int | None = None
    newest: int | None = None
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    for record in read_jsonl(path):
        rows += 1
        key = row_key(record, KEY_FIELDS)
        if key in seen:
            findings.fatal.append(f"{path.name}: 业务键重复 {key}")
        seen.add(key)
        try:
            check_row(record, f"{path.name}:{record.get('ts')}", now_ms)
        except ValueError as error:
            findings.fatal.append(str(error))
            continue
        families.add(str(record["instId"]))
        if float(record["notional"]) > big:
            findings.warnings.append(
                f"{path.name}: 单笔名义 ${float(record['notional']):,.0f} 超过阈值"
                f"（{record['instId']} {record['posSide']}）"
            )
        oldest = record["ts"] if oldest is None else min(oldest, record["ts"])
        newest = record["ts"] if newest is None else max(newest, record["ts"])
    return rows, len(families), oldest, newest


def check_gaps(findings: Findings) -> int:
    """缺口是采集器的自检信号：有缺口说明某一轮轮询把环形缓冲绕过去了。"""
    gap_dir = liquidation_dir(root=CRYPTO_ROOT)
    total = 0
    for path in sorted((gap_dir / "gaps").glob("*.jsonl")):
        count = sum(1 for _ in read_jsonl(path))
        total += count
        if count:
            findings.warnings.append(
                f"缺口记录 {path.name}: {count} 条（对应事件已永久丢失）"
            )
    return total


def check_manifest(rows_on_disk: int, findings: Findings) -> None:
    """manifest 的行数/文件数必须与磁盘一致；不一致说明作业没跑完就验收了。"""
    path = MANIFEST_ROOT / f"{DATASET}.json"
    if not path.exists():
        findings.fatal.append(f"manifest 缺失（{path.name}）：先跑采集再验收")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if manifest.get("rows") != sum(entry["rows"] for entry in entries):
        findings.fatal.append("manifest rows 与 entries 行数和不符")
    if manifest.get("rows") != rows_on_disk:
        findings.fatal.append(
            f"manifest 记 {manifest.get('rows'):,} 行，磁盘实为 {rows_on_disk:,} 行"
        )


def audit(stale_hours: float, big: float) -> int:
    findings = Findings()
    files = sorted(liquidation_dir(root=CRYPTO_ROOT).glob("*.jsonl"))
    if not files:
        findings.fatal.append("没有任何事件文件：采集器还没成功落盘")
        return findings.report()
    rows_total = 0
    newest_all: int | None = None
    for path in files:
        rows, families, oldest, newest = check_file(path, findings, big)
        rows_total += rows
        print(
            f"  {path.name}: {rows:,} 行 / {families} 个标的"
            + (
                f" / {datetime.fromtimestamp(oldest / 1000, tz=UTC):%H:%M}"
                f"~{datetime.fromtimestamp(newest / 1000, tz=UTC):%H:%M}"
                if oldest and newest
                else ""
            )
        )
        newest_all = newest if newest_all is None else max(newest_all, newest or 0)
    check_manifest(rows_total, findings)
    gaps = check_gaps(findings)
    print(f"\n合计 {rows_total:,} 行 / {len(files)} 个日文件 / 缺口 {gaps} 条")
    if newest_all is None:
        findings.fatal.append("未读到任何事件时间")
    else:
        age_hours = (datetime.now(tz=UTC).timestamp() - newest_all / 1000) / 3600
        if age_hours > stale_hours:
            findings.warnings.append(
                f"最新事件距今 {age_hours:.1f} 小时（阈值 {stale_hours:.0f}）：轮询器可能没在跑"
            )
    return findings.report()


def main() -> int:
    parser = argparse.ArgumentParser(description="验收 OKX 强平事件流")
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS)
    parser.add_argument("--big-notional", type=float, default=DEFAULT_BIG_NOTIONAL)
    args: Any = parser.parse_args()
    return audit(args.stale_hours, args.big_notional)


if __name__ == "__main__":
    raise SystemExit(main())
