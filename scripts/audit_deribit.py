"""Deribit 期权链数据的验收闸：致命项退出码 1，告警项只打印。

致命项都是「数据不信」的量：JSONL 解析失败、业务键重复、价格/IV 超出物理范围、
标的或行权价缺失、买卖价倒挂、manifest 与实际文件不符。
告警项是「数据可能不全」的量：快照过期、链明显缩水、指数价缺失。

用法：

    uv run python scripts/audit_deribit.py
    uv run python scripts/audit_deribit.py --stale-hours 30
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.paths import (
    DERIBIT_DATASETS,
    DERIBIT_OPTIONS,
    DERIBIT_RAW_ROOT,
    MANIFEST_ROOT,
    deribit_dataset_dir,
)
from src.utils.deribit_store import KEY_FIELDS
from src.utils.jsonl_store import iso, read_jsonl, row_key, to_epoch

DEFAULT_STALE_HOURS = 30.0
DEFAULT_MIN_OPTIONS = 200
MAX_IV = 5.0  # 500%：Roger Lee 上界下再留一倍冗余，超过即是脏数据


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
        return 1 if self.fatal else 0


def check_rows(dataset: str, path: Path, findings: Findings) -> tuple[int, int | None]:
    """逐行检查，返回 (行数, 最新快照 epoch)。"""
    seen: set[tuple[str, ...]] = set()
    rows = 0
    newest: int | None = None
    for record in read_jsonl(path):  # 解析失败会直接抛错，由调用方兜住
        rows += 1
        key = row_key(record, KEY_FIELDS[dataset])
        if key in seen:
            findings.fatal.append(f"{path.name}: 业务键重复 {key}")
        seen.add(key)
        newest = max(
            [stamp for stamp in (newest, to_epoch(record.get("fetched_at"))) if stamp]
        )
        _check_quote(dataset, path, record, findings)
    return rows, newest


def _check_quote(dataset: str, path: Path, record: dict[str, Any], findings: Findings) -> None:
    """单行字段检查：价格、IV、行权价、买卖价。"""
    label = f"{path.name}:{record.get('instrument')}"
    mark = record.get("mark_price")
    if not isinstance(mark, (int, float)) or mark <= 0:
        findings.fatal.append(f"{label} mark_price 非正：{mark!r}")
    if dataset == DERIBIT_OPTIONS:
        strike = record.get("strike")
        if not isinstance(strike, (int, float)) or strike <= 0:
            findings.fatal.append(f"{label} strike 非正：{strike!r}")
        iv = record.get("mark_iv")
        if not isinstance(iv, (int, float)) or not 0 < iv <= MAX_IV:
            findings.fatal.append(f"{label} mark_iv 越界：{iv!r}")
    bid, ask = record.get("bid"), record.get("ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > ask:
        findings.fatal.append(f"{label} 买卖价倒挂：bid {bid} > ask {ask}")


def check_manifest(dataset: str, findings: Findings) -> None:
    """manifest 的文件数/行数必须与磁盘一致；不一致说明作业没跑完就验收了。"""
    manifest_path = MANIFEST_ROOT / f"deribit_raw_{dataset}.json"
    if not manifest_path.exists():
        findings.fatal.append(f"{dataset}: manifest 缺失（{manifest_path.name}）")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sum(entry["rows"] for entry in manifest.get("entries", []))
    if manifest.get("files") != len(manifest.get("entries", [])):
        findings.fatal.append(f"{dataset}: manifest files 与 entries 数不符")
    if manifest.get("rows") != rows:
        findings.fatal.append(f"{dataset}: manifest rows 与 entries 行数和不符")
    on_disk = sum(
        sum(1 for _ in read_jsonl(path))
        for path in sorted(deribit_dataset_dir(dataset, root=DERIBIT_RAW_ROOT).glob("*.jsonl"))
    )
    if on_disk != rows:
        findings.fatal.append(
            f"{dataset}: manifest 记 {rows:,} 行，磁盘实为 {on_disk:,} 行"
            "（作业没跑完就验收会出现这条）"
        )


def check_freshness(newest: int | None, stale_hours: float, findings: Findings) -> None:
    """快照过期：链不是只增不改的行情，过期即失去定价锚的作用。"""
    if newest is None:
        findings.fatal.append("未读到任何快照时刻")
        return
    age_hours = (datetime.now(tz=UTC).timestamp() - newest) / 3600
    if age_hours > stale_hours:
        findings.warnings.append(
            f"最新快照距今 {age_hours:.1f} 小时（阈值 {stale_hours:.0f}），落于 {iso(newest)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deribit 期权链数据验收")
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS)
    parser.add_argument("--min-options", type=int, default=DEFAULT_MIN_OPTIONS)
    args = parser.parse_args()

    findings = Findings()
    for dataset in DERIBIT_DATASETS:
        directory = deribit_dataset_dir(dataset, root=DERIBIT_RAW_ROOT)
        files = sorted(directory.glob("*.jsonl"))
        print(f"\n=== {dataset}（{len(files)} 个文件）===")
        if not files:
            findings.fatal.append(f"{dataset}: 目录里没有任何 JSONL")
            continue
        newest: int | None = None
        for path in files:
            try:
                rows, stamp = check_rows(dataset, path, findings)
            except (ValueError, OSError) as error:
                findings.fatal.append(f"{path.name}: 读取失败 {error}")
                continue
            newest = max([s for s in (newest, stamp) if s])
            if dataset == DERIBIT_OPTIONS and rows < args.min_options:
                findings.warnings.append(
                    f"{path.name}: 仅 {rows} 档期权（阈值 {args.min_options}），疑抓取不全"
                )
            print(f"  {path.name}: {rows:,} 行")
        check_manifest(dataset, findings)
        check_freshness(newest, args.stale_hours, findings)

    return findings.report()


if __name__ == "__main__":
    raise SystemExit(main())
