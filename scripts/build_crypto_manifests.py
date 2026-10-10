"""
scripts/build_crypto_manifests.py
为自持的 crypto 数据集生成 manifest（覆盖区间/行数/来源/hash）。

容器数据湖的约定照搬过来：每个数据集一份 JSON，放在 data/manifests/ 下，
数据集名 = crypto_<exchange>_<market>_<timeframe>。crypto 不入湖由容器层 ADR-0006 豁免，
但"数据集必须有 manifest"这条不放宽。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import CRYPTO_ROOT, MANIFEST_ROOT

SCHEMA_VERSION = 1
# 时间列名候选：不同来源的 parquet 索引/列名不一致
TIME_COLUMNS = ("datetime", "time", "Index", "timestamp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(frame: pd.DataFrame, path: Path) -> tuple[str | None, str | None]:
    """数据集的时间覆盖区间：优先用索引，其次用时间列。"""
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) and not index.empty:
        return str(index[0]), str(index[-1])
    for column in TIME_COLUMNS:
        if column in frame.columns:
            series = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            if not series.empty:
                return str(series.iloc[0]), str(series.iloc[-1])
    return None, None


def build_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    """扫描一个数据集目录（exchange/timeframe）下的所有 parquet。"""
    files: list[dict[str, object]] = []
    rows = 0
    starts: list[str] = []
    ends: list[str] = []

    for path in sorted(dataset_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        start, end = _coverage(frame, path)
        if start:
            starts.append(start)
        if end:
            ends.append(end)
        rows += len(frame)
        files.append(
            {
                "path": path.name,
                "rows": len(frame),
                "sha256": _sha256(path),
                "coverage": {"start": start, "end": end},
            }
        )

    exchange, market, timeframe = dataset_dir.parts[-3:]
    return {
        "dataset": f"crypto_{exchange}_{market}_{timeframe}",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "coverage": {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        },
        "symbols": len(files),
        "rows": rows,
        "sources": [exchange],
        "files": files,
    }


def main() -> None:
    """遍历 data/storage/crypto/<exchange>/<market>/<timeframe>/，各写一份 manifest。"""
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_dirs = sorted(
        d for d in CRYPTO_ROOT.glob("*/*/*") if d.is_dir() and any(d.glob("*.parquet"))
    )
    for dataset_dir in dataset_dirs:
        manifest = build_dataset_manifest(dataset_dir)
        out = MANIFEST_ROOT / f"{manifest['dataset']}.json"
        out.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{manifest['dataset']}: {manifest['symbols']} 个文件 / "
            f"{manifest['rows']} 行 / {manifest['coverage']['start']} → "
            f"{manifest['coverage']['end']} → {out}"
        )


if __name__ == "__main__":
    main()
