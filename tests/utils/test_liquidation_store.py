"""强平事件流的契约测试：面值换算、分日、去重、缺口检测。

夹具是 2026-10-17 实测响应（BTC-USDT 两个 details）的裁剪版，数值原样保留，
所以 `notional = sz × ctVal × bkPx` 的断言就是线上口径。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.paths import liquidation_dir
from src.utils.data_sources.okx_liquidation import (
    ContractSpec,
    LiquidationEvent,
    parse_contracts,
    parse_events,
)
from src.utils.jsonl_store import read_jsonl
from src.utils.liquidation_store import (
    append_events,
    append_gaps,
    detect_gaps,
    event_row,
    group_by_day,
    load_last_seen,
    update_manifest,
)

CT_VAL = 0.01
CONTRACTS = {
    "BTC-USDT-SWAP": ContractSpec(
        inst_id="BTC-USDT-SWAP",
        inst_family="BTC-USDT",
        ct_val=CT_VAL,
        ct_val_ccy="BTC",
        ct_type="linear",
    )
}
PAYLOAD = {
    "code": "0",
    "data": [
        {
            "instFamily": "BTC-USDT",
            "instId": "BTC-USDT-SWAP",
            "details": [
                {
                    "bkPx": "81440.7",
                    "posSide": "short",
                    "side": "buy",
                    "sz": "0.09",
                    "ts": "1792205280547",
                },
                {
                    "bkPx": "81473.4",
                    "posSide": "long",
                    "side": "sell",
                    "sz": "1.19",
                    "ts": "1792198264660",
                },
            ],
        }
    ],
    "msg": "",
}


def make_event(
    inst: str, ts_ms: int, size: float = 1.0, price: float = 100.0
) -> LiquidationEvent:
    """构造一条事件，默认参数不影响任何断言。"""
    return LiquidationEvent(
        inst_id=inst,
        inst_family=inst.removesuffix("-SWAP"),
        side="sell",
        pos_side="long",
        price=price,
        size_contracts=size,
        ct_val=CT_VAL,
        ts_ms=ts_ms,
    )


def test_parse_events_converts_contracts_to_notional() -> None:
    events = parse_events(PAYLOAD, CONTRACTS)
    assert len(events) == 2
    first = events[0]
    assert first.inst_family == "BTC-USDT"
    assert first.pos_side == "short" and first.side == "buy"
    assert first.size_ccy == pytest.approx(0.09 * CT_VAL)
    assert first.notional == pytest.approx(0.09 * CT_VAL * 81440.7)
    assert first.key[0] == "BTC-USDT-SWAP" and first.key[1] == 1792205280547


def test_parse_events_rejects_unknown_side_and_missing_contract() -> None:
    bad = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "details": [
                    {
                        "bkPx": "1",
                        "posSide": "flat",
                        "side": "buy",
                        "sz": "1",
                        "ts": "1",
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="未知方向"):
        parse_events(bad, CONTRACTS)
    assert parse_events(PAYLOAD, {}) == []


def test_parse_contracts_keeps_only_linear() -> None:
    payload = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "instFamily": "BTC-USDT",
                "ctVal": "0.01",
                "ctValCcy": "BTC",
                "ctType": "linear",
            },
            {
                "instId": "BTC-USD-SWAP",
                "instFamily": "BTC-USD",
                "ctVal": "100",
                "ctValCcy": "USD",
                "ctType": "inverse",
            },
        ],
    }
    specs = parse_contracts(payload)
    assert list(specs) == ["BTC-USDT-SWAP"]


def test_partition_uses_event_time_not_wall_clock() -> None:
    day_one = 1792198264660
    day_two = day_one + 86_400_000
    buckets = group_by_day(
        [make_event("BTC-USDT-SWAP", day_one), make_event("BTC-USDT-SWAP", day_two)],
        "now",
    )
    assert len(buckets) == 2


def test_append_events_is_idempotent(tmp_path: Path) -> None:
    events = [
        make_event("BTC-USDT-SWAP", 1792198264660),
        make_event("ETH-USDT-SWAP", 1792198264661),
    ]
    added, skipped, touched = append_events(events, "t0", root=tmp_path)
    assert (added, skipped) == (2, 0) and len(touched) == 1
    added, skipped, _ = append_events(events, "t1", root=tmp_path)
    assert (added, skipped) == (0, 2)
    path = next(liquidation_dir(root=tmp_path).glob("*.jsonl"))
    assert len(list(read_jsonl(path))) == 2


def test_event_row_keeps_raw_size_for_recheck() -> None:
    row = event_row(
        make_event("BTC-USDT-SWAP", 1792198264660, size=1.19, price=81473.4),
        "2026-10-17T00:00:00.000Z",
    )
    assert row["sz"] == "1.19"
    assert row["notional"] == pytest.approx(1.19 * CT_VAL * 81473.4)
    assert row["tsIso"].endswith("Z") and row["fetchedAt"] == "2026-10-17T00:00:00.000Z"


def test_load_last_seen_takes_max_per_instrument(tmp_path: Path) -> None:
    append_events(
        [
            make_event("BTC-USDT-SWAP", 100),
            make_event("BTC-USDT-SWAP", 300),
            make_event("ETH-USDT-SWAP", 200),
        ],
        "t0",
        root=tmp_path,
    )
    day = next(liquidation_dir(root=tmp_path).glob("*.jsonl")).stem
    assert load_last_seen([day], root=tmp_path) == {
        "BTC-USDT-SWAP": 300,
        "ETH-USDT-SWAP": 200,
    }


def test_detect_gaps_skips_first_sighting_and_respects_slack() -> None:
    events = [make_event("BTC-USDT-SWAP", 10_000), make_event("ETH-USDT-SWAP", 10_000)]
    assert detect_gaps(events, {}, slack_ms=60_000) == []
    near = detect_gaps(
        [make_event("BTC-USDT-SWAP", 50_000)],
        {"BTC-USDT-SWAP": 10_000},
        slack_ms=60_000,
    )
    assert near == []
    far = detect_gaps(
        [make_event("BTC-USDT-SWAP", 200_000)],
        {"BTC-USDT-SWAP": 10_000},
        slack_ms=60_000,
    )
    assert far[0]["lostMs"] == 190_000


def test_append_gaps_dedupes_same_gap(tmp_path: Path) -> None:
    gap = {
        "instId": "BTC-USDT-SWAP",
        "prevTs": 1,
        "nextTs": 2,
        "lostMs": 1,
        "detectedAt": "t",
    }
    assert append_gaps([gap], root=tmp_path) == 1
    assert append_gaps([gap], root=tmp_path) == 0
    assert sum(1 for _ in liquidation_dir(root=tmp_path).glob("gaps/*.jsonl")) == 1


def test_manifest_coverage_is_seconds_though_rows_store_ms(tmp_path: Path) -> None:
    """`ts` 存毫秒，manifest 覆盖区间按 epoch 秒解析，所以只能走 `tsIso`。

    2026-10-17 实采时踩到：直接拿毫秒当秒喂 `fromtimestamp` 会抛
    `OSError: [Errno 22]`（Windows 上越界），manifest 写不成、整轮采集白跑。
    """
    append_events([make_event("BTC-USDT-SWAP", 1_792_198_264_660)], "t0", root=tmp_path)
    manifest = json.loads(
        update_manifest(root=tmp_path, manifest_root=tmp_path / "manifests").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["rows"] == 1
    assert manifest["coverage"] == {
        "start": "2026-10-17T00:51:04+00:00",
        "end": "2026-10-17T00:51:04+00:00",
    }
