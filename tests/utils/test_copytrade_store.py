"""OKX 跟单榜单落盘的契约测试（合成响应，不联网、不碰真实数据湖）。

抓的是会静默出错的地方：上游把数字写成字符串、同版本重跑不能写重复行、新版本必须
另留一行、按天分区、manifest 的行数与覆盖区间必须真是这批文件的。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.utils.copytrade_store import (
    KEY_FIELDS,
    append_snapshot,
    read_snapshot,
    snapshot_rows,
    update_manifest,
)
from src.utils.data_sources.base import parse_instant
from src.utils.data_sources.okx_copytrading import (
    Leaderboard,
    TraderStats,
    parse_rank,
    parse_stats,
)
from src.utils.jsonl_store import read_jsonl

FETCHED_AT = "2026-10-17T15:40:00+00:00"
DAY = "2026-10-17"


def raw_rank(code: str, *, ratio: str = "5.5222", aum: str = "688066.06") -> dict[str, Any]:
    """榜单行：上游把数字全写成字符串，history 是 3 个日度点（故意乱序）。"""
    return {
        "uniqueCode": code,
        "nickName": f"trader-{code}",
        "ccy": "USDT",
        "aum": aum,
        "pnl": "245506.49",
        "pnlRatio": ratio,
        "winRatio": "0.5281",
        "leadDays": "228",
        "copyTraderNum": "533",
        "accCopyTraderNum": "1329",
        "maxCopyTraderNum": "600",
        "traderInsts": ["BTC-USDT-SWAP"],
        "pnlRatios": [
            {"beginTs": "1792080000000", "pnlRatio": ratio},
            {"beginTs": "1791216000000", "pnlRatio": "4.5053"},
            {"beginTs": "1791648000000", "pnlRatio": "5.4391"},
        ],
    }


def leaderboard(data_ver: str, *codes: str) -> Leaderboard:
    return Leaderboard(
        data_ver=data_ver,
        inst_type="SWAP",
        fetched_at=parse_instant(FETCHED_AT, "fetched_at"),
        ranks=tuple(parse_rank(raw_rank(code)) for code in codes),
    )


def stats_for(code: str) -> TraderStats:
    return parse_stats(
        code,
        {
            "ccy": "USDT",
            "investAmt": "44458.138",
            "curCopyTraderPnl": "52028.365",
            "avgSubPosNotional": "121948.1335",
            "profitDays": "42",
            "lossDays": "47",
            "winRatio": "0.5281",
        },
    )


def test_parse_rank_keeps_strings_typed_and_sorts_history() -> None:
    rank = parse_rank(raw_rank("CA1C48E543EF7A95"))
    assert rank.aum == 688066.06
    assert rank.lead_days == 228
    assert rank.instrument_num == 1  # 上游给的是一串标的，落盘只留个数
    assert rank.history[0].begin_ts == 1791216000000  # 乱序输入按时间排好
    assert [point.pnl_ratio for point in rank.history] == [4.5053, 5.4391, 5.5222]


def test_parse_rank_rejects_bad_rows() -> None:
    """缺业务键或数字解析失败必须抛错，不能静默补 0 污染 raw 层。"""
    with pytest.raises(ValueError):
        parse_rank({**raw_rank("X"), "uniqueCode": ""})
    with pytest.raises(ValueError):
        parse_rank({**raw_rank("X"), "aum": "n/a"})


def test_snapshot_row_carries_both_roi_scopes() -> None:
    """交易员自有收益率与跟单者盈亏同排存放，且字段名保持上游拼写。"""
    board = leaderboard("20261016230001", "AAA")
    rows = snapshot_rows(board, {"AAA": stats_for("AAA")}, fetched_at=FETCHED_AT)
    row = rows[0]
    assert row["pnlRatio"] == 5.5222 and row["investAmt"] == 44458.138
    assert row["curCopyTraderPnl"] == 52028.365
    assert row["dataVer"] == "20261016230001" and row["fetched_at"] == FETCHED_AT
    assert row["history"][0] == [1791216000000, 4.5053]


def test_snapshot_row_without_stats_keeps_rank_fields() -> None:
    """单人统计失败不该丢掉榜单标量（落盘后再由验收闸报缺统计）。"""
    rows = snapshot_rows(leaderboard("v1", "AAA"), {}, fetched_at=FETCHED_AT)
    assert rows[0]["aum"] == 688066.06
    assert "investAmt" not in rows[0]


def test_append_snapshot_dedupes_same_version_and_keeps_new_one(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    first = snapshot_rows(leaderboard("v1", "AAA", "BBB"), {}, fetched_at=FETCHED_AT)
    assert append_snapshot(first, day=DAY, root=root) == (2, 0)
    # 同一天同一版本重跑：一行都不新增
    assert append_snapshot(first, day=DAY, root=root) == (0, 2)
    # 平台刷新出 v2：同一个人再留一行（raw 只增不改）
    second = snapshot_rows(leaderboard("v2", "AAA", "BBB"), {}, fetched_at=FETCHED_AT)
    assert append_snapshot(second, day=DAY, root=root) == (2, 0)
    keys = [tuple(record[field] for field in KEY_FIELDS) for record in read_jsonl(root / "lead_traders" / f"{DAY}.jsonl")]
    assert keys == [("AAA", "v1"), ("BBB", "v1"), ("AAA", "v2"), ("BBB", "v2")]


def test_append_snapshot_partitions_by_day(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    append_snapshot(
        snapshot_rows(leaderboard("v1", "AAA"), {}, fetched_at=FETCHED_AT), day=DAY, root=root
    )
    append_snapshot(
        snapshot_rows(leaderboard("v2", "AAA"), {}, fetched_at=FETCHED_AT),
        day="2026-10-18",
        root=root,
    )
    assert len(read_snapshot(DAY, root=root)) == 1
    assert len(read_snapshot("2026-10-18", root=root)) == 1
    assert read_snapshot("2026-10-19", root=root) == []


def test_update_manifest_counts_disk_rows_and_coverage(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    append_snapshot(
        snapshot_rows(leaderboard("v1", "AAA", "BBB"), {}, fetched_at=FETCHED_AT),
        day=DAY,
        root=root,
    )
    path = update_manifest(
        touched=[f"{DAY}.jsonl"], raw_root=root, manifest_root=manifest_root
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["rows"] == 2 and manifest["files"] == 1
    assert manifest["entries"][0]["path"] == f"{DAY}.jsonl"
    assert manifest["coverage"]["start"] == FETCHED_AT
    on_disk = sum(1 for _ in read_jsonl(root / "lead_traders" / f"{DAY}.jsonl"))
    assert manifest["rows"] == on_disk
