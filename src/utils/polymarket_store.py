"""
预测市场 raw 层的 JSONL 存储：去重追加 + manifest。

raw 只增不改（容器 ADR-0007）：`append_jsonl` 先读已有业务键，去重后追加，
重复运行不产生重复行；不提供任何改写接口，坏行修正只能重建分区。

数据集与文件名约定见 `src/config/paths.py`：

    data/storage/polymarket/raw/<dataset>/<name>.jsonl

manifest 落 `data/manifests/polymarket_raw_<dataset>.json`，字段对齐数据湖
（覆盖区间 / 行数 / 来源 / 每文件 sha256）。sha256 是重活，故只对本次触及的
文件重算，其余沿用上一版 manifest 的条目。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
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
    polymarket_jsonl_path,
)

SCHEMA_VERSION = 1

# 各数据集的去重业务键
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    POLYMARKET_EVENTS: ("id", "updatedAt"),
    POLYMARKET_MARKETS: ("conditionId", "updatedAt"),
    POLYMARKET_PRICES: ("token", "t"),
    POLYMARKET_TRADES: ("transactionHash", "asset", "timestamp", "price", "size"),
}
# 各数据集用于取覆盖区间的时间字段（epoch 秒或 ISO 字符串）；None 表示不提取
TIME_FIELDS: dict[str, str | None] = {
    POLYMARKET_EVENTS: "fetched_at",
    POLYMARKET_MARKETS: "updatedAt",
    POLYMARKET_PRICES: "t",
    POLYMARKET_TRADES: "timestamp",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """流式读 JSONL；损坏的行直接抛错（raw 层不允许静默丢行）。"""
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} 不是合法 JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number} 不是 JSON 对象")
            yield record


def row_key(record: dict[str, Any], key_fields: Sequence[str]) -> tuple[str, ...]:
    """业务键：缺字段用空串占位，保证同一文件内的键长度一致。"""
    return tuple(str(record.get(field, "")) for field in key_fields)


def existing_keys(path: Path, key_fields: Sequence[str]) -> set[tuple[str, ...]]:
    """已落盘行的业务键集合。"""
    return {row_key(record, key_fields) for record in read_jsonl(path)}


def append_jsonl(
    path: Path, records: Iterable[dict[str, Any]], key_fields: Sequence[str]
) -> tuple[int, int]:
    """按业务键去重后追加，返回 (新增, 跳过)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys(path, key_fields) if path.exists() else set()
    added = 0
    skipped = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            key = row_key(record, key_fields)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            added += 1
    return added, skipped


def to_epoch(value: Any) -> int | None:
    """覆盖区间统一按 epoch 秒：成交/价格点直接是秒，ISO 字符串转过去。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return int(moment.timestamp())
    return None


def iso(epoch: int | None) -> str | None:
    """epoch 秒 → ISO 字符串（manifest 里的人读形式）。"""
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC).isoformat()


def day_stamp(moment: datetime) -> str:
    """每日快照的文件名（UTC 日期）。"""
    return moment.astimezone(UTC).strftime("%Y-%m-%d")


def _scan_file(path: Path, time_field: str | None) -> dict[str, Any]:
    """单文件统计：行数、字节数、sha256、时间覆盖（取扫描到的极值）。"""
    digest = hashlib.sha256()
    rows = 0
    start: int | None = None
    end: int | None = None
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            digest.update(raw)
            rows += 1
            if time_field is None:
                continue
            moment = to_epoch(json.loads(raw).get(time_field))
            if moment is None:
                continue
            start = moment if start is None else min(start, moment)
            end = moment if end is None else max(end, moment)
    stat = path.stat()
    return {
        "path": path.name,
        "rows": rows,
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": digest.hexdigest(),
        "coverage": {"start": iso(start), "end": iso(end)},
    }


def latest_market_snapshot(
    day: str, *, raw_root: Path = POLYMARKET_RAW_ROOT
) -> list[dict[str, Any]]:
    """当天市场快照里每个市场的最新一行（重取明细、不想再枚举时用）。"""
    path = polymarket_jsonl_path(POLYMARKET_MARKETS, day, root=raw_root)
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        key = str(record.get("conditionId") or record.get("slug"))
        known = latest.get(key)
        if known is None or str(record.get("fetched_at")) > str(
            known.get("fetched_at")
        ):
            latest[key] = record
    return list(latest.values())


def update_manifest(
    dataset: str,
    *,
    touched: Iterable[str] = (),
    raw_root: Path = POLYMARKET_RAW_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> Path:
    """重写数据集 manifest；只对本次触及的文件重算，其余沿用旧条目。"""
    directory = polymarket_dataset_dir(dataset, root=raw_root)
    manifest_path = manifest_root / f"polymarket_raw_{dataset}.json"
    previous: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {entry["path"]: entry for entry in cached.get("entries", [])}

    time_field = TIME_FIELDS.get(dataset)
    dirty = set(touched)
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        cached = previous.get(path.name)
        if cached is not None and path.name not in dirty:
            entries.append(cached)
            continue
        entries.append(_scan_file(path, time_field))

    starts = [
        entry["coverage"]["start"] for entry in entries if entry["coverage"]["start"]
    ]
    ends = [entry["coverage"]["end"] for entry in entries if entry["coverage"]["end"]]
    manifest = {
        "dataset": f"polymarket_raw_{dataset}",
        "schema_version": SCHEMA_VERSION,
        "layer": "raw",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": _source(dataset),
        "coverage": {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        },
        "files": len(entries),
        "rows": sum(entry["rows"] for entry in entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def _source(dataset: str) -> str:
    """数据集的来源主机，写进 manifest 便于追溯。"""
    if dataset in (POLYMARKET_EVENTS, POLYMARKET_MARKETS):
        return "gamma-api.polymarket.com"
    if dataset == POLYMARKET_PRICES:
        return "clob.polymarket.com"
    return "data-api.polymarket.com"
