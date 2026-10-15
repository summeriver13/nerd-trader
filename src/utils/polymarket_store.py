"""
预测市场 raw 层的存储契约：业务键、来源主机与当天快照读取。

通用逻辑（去重追加、manifest 生成）在 `src/utils/jsonl_store.py`；
这里只声明 Polymarket 自己的东西。数据集与文件名约定见 `src/config/paths.py`：

    data/storage/polymarket/raw/<dataset>/<name>.jsonl

manifest 落 `data/manifests/polymarket_raw_<dataset>.json`（容器 ADR-0007）。
"""

from __future__ import annotations

from collections.abc import Iterable
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
from src.utils.jsonl_store import read_jsonl, rewrite_manifest

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
        if known is None or str(record.get("fetched_at")) > str(known.get("fetched_at")):
            latest[key] = record
    return list(latest.values())


def update_manifest(
    dataset: str,
    *,
    touched: Iterable[str] = (),
    raw_root: Path = POLYMARKET_RAW_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> Path:
    """重写该数据集 manifest；只对本次触及的文件重算，其余沿用旧条目。"""
    return rewrite_manifest(
        f"polymarket_raw_{dataset}",
        directory=polymarket_dataset_dir(dataset, root=raw_root),
        source=_source(dataset),
        time_field=TIME_FIELDS.get(dataset),
        touched=touched,
        manifest_root=manifest_root,
    )


def _source(dataset: str) -> str:
    """数据集的来源主机，写进 manifest 便于追溯。"""
    if dataset in (POLYMARKET_EVENTS, POLYMARKET_MARKETS):
        return "gamma-api.polymarket.com"
    if dataset == POLYMARKET_PRICES:
        return "clob.polymarket.com"
    return "data-api.polymarket.com"
