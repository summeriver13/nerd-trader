"""OKX 跟单榜单数据的验收闸：致命项退出码 1，告警项只打印。

致命项都是「数据不信」的量：JSONL 解析失败、业务键重复、uniqueCode/dataVer/fetched_at
缺失、核心数值（aum/pnl/pnlRatio/winRatio）非数字或越界、aum 与跟单人数为负、
manifest 与实际文件不符。
告警项是「数据可能不全或口径漂移」的量：快照过期、当日人数明显缩水（抓了一半）、
跟单统计缺失、`pnlRatio ≈ pnl/investAmt` 不成立的样本占比过高、history 时间未递增。

用法：

    uv run python scripts/audit_okx_copy.py
    uv run python scripts/audit_okx_copy.py --stale-hours 48 --min-traders 100
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.paths import MANIFEST_ROOT, OKX_COPY_RAW_ROOT, okx_copy_dataset_dir
from src.utils.copytrade_store import KEY_FIELDS, MANIFEST_NAME
from src.utils.jsonl_store import iso, read_jsonl, row_key, to_epoch

DEFAULT_STALE_HOURS = 30.0
DEFAULT_MIN_TRADERS = 100
MIN_HISTORY_FOR_IDENTITY = 1e-4
MIN_PNL_FOR_IDENTITY = 1_000.0
IDENTITY_TOLERANCE = 0.05
SHRINK_RATIO = 0.5


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


def check_row(record: dict[str, Any], path: Path, counts: dict[str, int]) -> None:
    """单行检查：核心字段可读、数值在物理范围内；口径漂移记进计数（不逐行告警）。"""
    label = f"{path.name}:{record.get('uniqueCode') or '-'}"
    for field in ("uniqueCode", "dataVer", "fetched_at"):
        if not record.get(field):
            raise ValueError(f"{label} 缺字段 {field}")
    for field in ("aum", "pnl", "pnlRatio", "winRatio"):
        if not isinstance(record.get(field), (int, float)):
            raise ValueError(f"{label} {field} 非数字：{record.get(field)!r}")
    if record["aum"] < 0 or record.get("copyTraderNum", 0) < 0:
        raise ValueError(f"{label} aum/跟单人数为负")
    if not 0.0 <= record["winRatio"] <= 1.0:
        raise ValueError(f"{label} winRatio 越界：{record['winRatio']}")
    _check_identity(record, counts)
    _check_history(record, counts)


def _check_identity(record: dict[str, Any], counts: dict[str, int]) -> None:
    """`pnlRatio ≈ pnl / investAmt` 是实测核对过的口径，漂了说明上游改了分母。"""
    invested = record.get("investAmt")
    if not isinstance(invested, (int, float)) or invested <= 0:
        counts["no_stats"] += 1
        return
    if abs(record["pnlRatio"]) < MIN_HISTORY_FOR_IDENTITY:
        return
    if abs(record["pnl"]) < MIN_PNL_FOR_IDENTITY:
        # 小额盈亏受展示取整支配（0.0004 这类四位数比率），不构成口径漂移的证据
        return
    drift = abs(record["pnl"] / invested - record["pnlRatio"]) / abs(record["pnlRatio"])
    if drift > IDENTITY_TOLERANCE:
        counts["identity_bad"] += 1


def _check_history(record: dict[str, Any], counts: dict[str, int]) -> None:
    """日度收益序列必须按开始时刻递增（上游乱序会让下游取错最新点）。"""
    history = record.get("history")
    if not isinstance(history, list):
        return
    stamps = [point[0] for point in history if isinstance(point, list) and point]
    if stamps != sorted(stamps):
        counts["history_unsorted"] += 1


def check_file(
    path: Path, findings: Findings
) -> tuple[int, int | None, dict[str, int]]:
    """逐行检查，返回 (行数, 最新快照 epoch, 计数)。解析失败由调用方兜住。"""
    seen: set[tuple[str, ...]] = set()
    counts = {"no_stats": 0, "identity_bad": 0, "history_unsorted": 0}
    rows = 0
    newest: int | None = None
    for record in read_jsonl(path):
        rows += 1
        key = row_key(record, KEY_FIELDS)
        if key in seen:
            findings.fatal.append(f"{path.name}: 业务键重复 {key}")
        seen.add(key)
        try:
            check_row(record, path, counts)
        except ValueError as error:
            findings.fatal.append(str(error))
        newest = max(
            [stamp for stamp in (newest, to_epoch(record.get("fetched_at"))) if stamp]
        )
    return rows, newest, counts


def report_counts(path: Path, counts: dict[str, int], rows: int) -> list[str]:
    """把逐行计数压成按文件一行的告警。"""
    messages: list[str] = []
    if counts["no_stats"]:
        messages.append(
            f"{path.name}: {counts['no_stats']}/{rows} 行缺跟单统计（investAmt）"
        )
    if counts["identity_bad"]:
        messages.append(
            f"{path.name}: {counts['identity_bad']}/{rows} 行 pnlRatio ≠ pnl/investAmt"
            f"（容差 {IDENTITY_TOLERANCE:.0%}），上游口径可能改了"
        )
    if counts["history_unsorted"]:
        messages.append(
            f"{path.name}: {counts['history_unsorted']} 行 history 时间未递增"
        )
    return messages


def check_manifest(findings: Findings) -> None:
    """manifest 的文件数/行数必须与磁盘一致；不一致说明作业没跑完就验收了。"""
    manifest_path = MANIFEST_ROOT / f"{MANIFEST_NAME}.json"
    if not manifest_path.exists():
        findings.fatal.append(f"manifest 缺失（{manifest_path.name}）")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    rows = sum(entry["rows"] for entry in entries)
    if manifest.get("files") != len(entries):
        findings.fatal.append("manifest files 与 entries 数不符")
    if manifest.get("rows") != rows:
        findings.fatal.append("manifest rows 与 entries 行数和不符")
    on_disk = sum(
        sum(1 for _ in read_jsonl(path))
        for path in sorted(okx_copy_dataset_dir(root=OKX_COPY_RAW_ROOT).glob("*.jsonl"))
    )
    if on_disk != rows:
        findings.fatal.append(
            f"manifest 记 {rows:,} 行，磁盘实为 {on_disk:,} 行（作业没跑完就验收会出现这条）"
        )


def check_freshness(newest: int | None, stale_hours: float, findings: Findings) -> None:
    """榜单是每天刷新的快照：过期说明计划任务没跑或跑失败。"""
    if newest is None:
        findings.fatal.append("未读到任何快照时刻")
        return
    age_hours = (datetime.now(tz=UTC).timestamp() - newest) / 3600
    if age_hours > stale_hours:
        findings.warnings.append(
            f"最新快照距今 {age_hours:.1f} 小时（阈值 {stale_hours:.0f}），落于 {iso(newest)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX 跟单榜单数据验收")
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS)
    parser.add_argument("--min-traders", type=int, default=DEFAULT_MIN_TRADERS)
    args = parser.parse_args()

    findings = Findings()
    directory = okx_copy_dataset_dir(root=OKX_COPY_RAW_ROOT)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        findings.fatal.append(f"{directory} 里没有任何 JSONL")
        return findings.report()

    newest: int | None = None
    counts_per_file: list[tuple[Path, int, dict[str, int]]] = []
    for path in files:
        try:
            rows, stamp, counts = check_file(path, findings)
        except (ValueError, OSError) as error:
            findings.fatal.append(f"{path.name}: 读取失败 {error}")
            continue
        counts_per_file.append((path, rows, counts))
        newest = max([s for s in (newest, stamp) if s])
        print(f"  {path.name}: {rows:,} 行")
    for path, rows, counts in counts_per_file:
        findings.warnings.extend(report_counts(path, counts, rows))

    latest_rows = counts_per_file[-1][1] if counts_per_file else 0
    if latest_rows < args.min_traders:
        findings.warnings.append(
            f"最新快照仅 {latest_rows:,} 人（阈值 {args.min_traders}），疑抓取不全"
        )
    if len(counts_per_file) >= 2:
        previous_rows = counts_per_file[-2][1]
        if previous_rows and latest_rows < previous_rows * SHRINK_RATIO:
            findings.warnings.append(
                f"最新快照 {latest_rows:,} 人，较上一份 {previous_rows:,} 人缩水超"
                f" {1 - SHRINK_RATIO:.0%}"
            )
    if not counts_per_file:
        findings.fatal.append("没有任何可读的 JSONL 文件")

    check_manifest(findings)
    check_freshness(newest, args.stale_hours, findings)
    return findings.report()


if __name__ == "__main__":
    raise SystemExit(main())
