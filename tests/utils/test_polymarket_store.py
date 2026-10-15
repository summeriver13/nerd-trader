"""raw 层存储的不变量：去重追加、只增不改、manifest 与磁盘一致。

这里断言的都是「以后想改必须先推翻」的口径：同一业务键只落一次、坏行必须炸、
manifest 只对触及的文件重算（sha256 是重活）。
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.paths import POLYMARKET_MARKETS, POLYMARKET_PRICES, POLYMARKET_TRADES
from src.utils.jsonl_store import (
    append_jsonl,
    day_stamp,
    existing_keys,
    read_jsonl,
    row_key,
    to_epoch,
)
from src.utils.polymarket_store import (
    KEY_FIELDS,
    latest_market_snapshot,
    update_manifest,
)


def _price(price: float, moment: int, token: str = "t1") -> dict[str, object]:
    return {"token": token, "t": moment, "p": price}


def test_append_jsonl_dedupes_by_business_key(tmp_path: Path) -> None:
    """重复运行不产生重复行：同一 (token, t) 第二次进来只算跳过。"""
    path = tmp_path / "prices" / "m.jsonl"
    rows = [_price(0.4, 100), _price(0.5, 200)]
    assert append_jsonl(path, rows, KEY_FIELDS[POLYMARKET_PRICES]) == (2, 0)
    assert append_jsonl(path, rows, KEY_FIELDS[POLYMARKET_PRICES]) == (0, 2)
    assert len(list(read_jsonl(path))) == 2


def test_append_jsonl_keeps_existing_lines(tmp_path: Path) -> None:
    """只增不改：追加后旧行原样还在，且顺序是「先到先落」。"""
    path = tmp_path / "prices" / "m.jsonl"
    append_jsonl(path, [_price(0.4, 100)], KEY_FIELDS[POLYMARKET_PRICES])
    append_jsonl(path, [_price(0.6, 50)], KEY_FIELDS[POLYMARKET_PRICES])
    assert [record["t"] for record in read_jsonl(path)] == [100, 50]
    assert existing_keys(path, KEY_FIELDS[POLYMARKET_PRICES]) == {
        ("t1", "100"),
        ("t1", "50"),
    }


def test_row_key_fills_missing_fields(tmp_path: Path) -> None:
    """缺字段用空串占位：键长度必须恒定，否则去重会漏。"""
    assert row_key({}, KEY_FIELDS[POLYMARKET_TRADES]) == ("", "", "", "", "")


def test_read_jsonl_rejects_broken_line(tmp_path: Path) -> None:
    """坏行不能静默跳过：raw 层的价值就是「行数即事实」。"""
    path = tmp_path / "prices" / "bad.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"p": 0.4}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_jsonl(path))


def test_to_epoch_handles_iso_and_seconds() -> None:
    assert to_epoch(1730000000) == 1730000000
    assert to_epoch("2026-10-14T04:22:00Z") == int(
        datetime(2026, 10, 14, 4, 22, tzinfo=UTC).timestamp()
    )
    assert to_epoch("not-a-date") is None
    assert to_epoch(True) is None


def test_day_stamp_is_utc_date() -> None:
    assert day_stamp(datetime(2026, 10, 14, 23, 30, tzinfo=UTC)) == "2026-10-14"


def test_update_manifest_counts_and_coverage(tmp_path: Path) -> None:
    """manifest 的覆盖区间按极值算，与追加顺序无关（成交是新到旧抓的）。"""
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    path = raw_root / POLYMARKET_PRICES / "m.jsonl"
    append_jsonl(
        path, [_price(0.4, 200), _price(0.5, 100)], KEY_FIELDS[POLYMARKET_PRICES]
    )
    update_manifest(POLYMARKET_PRICES, raw_root=raw_root, manifest_root=manifest_root)
    manifest = json.loads(
        (manifest_root / f"polymarket_raw_{POLYMARKET_PRICES}.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["files"] == 1
    assert manifest["rows"] == 2
    assert manifest["coverage"]["start"] < manifest["coverage"]["end"]
    assert manifest["source"] == "clob.polymarket.com"


def test_update_manifest_reuses_untouched_entries(tmp_path: Path) -> None:
    """未触及的文件不重算 sha256：被改动但没上报的文件会沿用旧条目（已知代价）。"""
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    path = raw_root / POLYMARKET_PRICES / "m.jsonl"
    append_jsonl(path, [_price(0.4, 100)], KEY_FIELDS[POLYMARKET_PRICES])
    update_manifest(POLYMARKET_PRICES, raw_root=raw_root, manifest_root=manifest_root)
    first = json.loads(
        (manifest_root / f"polymarket_raw_{POLYMARKET_PRICES}.json").read_text(
            encoding="utf-8"
        )
    )
    append_jsonl(path, [_price(0.6, 200)], KEY_FIELDS[POLYMARKET_PRICES])
    update_manifest(POLYMARKET_PRICES, raw_root=raw_root, manifest_root=manifest_root)
    cached = json.loads(
        (manifest_root / f"polymarket_raw_{POLYMARKET_PRICES}.json").read_text(
            encoding="utf-8"
        )
    )
    assert cached["entries"] == first["entries"]
    update_manifest(
        POLYMARKET_PRICES,
        touched=["m.jsonl"],
        raw_root=raw_root,
        manifest_root=manifest_root,
    )
    refreshed = json.loads(
        (manifest_root / f"polymarket_raw_{POLYMARKET_PRICES}.json").read_text(
            encoding="utf-8"
        )
    )
    assert refreshed["rows"] == 2
    assert refreshed["entries"] != first["entries"]


def test_latest_market_snapshot_keeps_newest_row_per_market(tmp_path: Path) -> None:
    """同一 market 每轮一行快照，重取明细时要取最新那行（fetched_at 最晚）。"""
    path = tmp_path / POLYMARKET_MARKETS / "2026-10-14.jsonl"
    rows = [
        {
            "conditionId": "0xa",
            "slug": "a",
            "updatedAt": "2026-10-14T01:00:00+00:00",
            "fetched_at": "2026-10-14T01:10:00+00:00",
        },
        {
            "conditionId": "0xa",
            "slug": "a",
            "updatedAt": "2026-10-14T02:00:00+00:00",
            "fetched_at": "2026-10-14T02:10:00+00:00",
        },
        {
            "conditionId": "0xb",
            "slug": "b",
            "updatedAt": "2026-10-14T01:30:00+00:00",
            "fetched_at": "2026-10-14T01:40:00+00:00",
        },
    ]
    append_jsonl(path, rows, KEY_FIELDS[POLYMARKET_MARKETS])
    latest = latest_market_snapshot("2026-10-14", raw_root=tmp_path)
    assert {record["conditionId"]: record["fetched_at"] for record in latest} == {
        "0xa": "2026-10-14T02:10:00+00:00",
        "0xb": "2026-10-14T01:40:00+00:00",
    }


def test_latest_market_snapshot_missing_day(tmp_path: Path) -> None:
    assert latest_market_snapshot("2026-01-01", raw_root=tmp_path) == []
