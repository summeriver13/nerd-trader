"""Deribit 链落盘的契约测试（合成链，不联网、不碰真实数据湖）。

抓的是会静默出错的地方：同一天多轮快照不能互相顶掉、同一快照重跑不能写重复行、
期货里的永续不能带到期码、manifest 的行数与覆盖区间必须真是这批文件的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.paths import DERIBIT_FUTURES, DERIBIT_OPTIONS, deribit_jsonl_path
from src.utils.data_sources.base import parse_instant
from src.utils.data_sources.deribit import DeribitChain, OptionQuote
from src.utils.deribit_store import (
    KEY_FIELDS,
    append_chain,
    future_rows,
    option_rows,
    update_manifest,
)
from src.utils.jsonl_store import read_jsonl


def quote(strike: float, kind: str, iv: float, *, bid: float | None, ask: float | None):
    return OptionQuote(
        instrument=f"BTC-25DEC26-{strike:.0f}-{kind}",
        expiry="25DEC26",
        kind=kind,
        strike=strike,
        mark_price=0.05,
        mark_iv=iv,
        bid=bid,
        ask=ask,
        open_interest=12.0,
        underlying=100_000.0,
    )


def chain(fetched: str) -> DeribitChain:
    return DeribitChain(
        fetched_at=parse_instant(fetched, "fetched_at"),
        currency="BTC",
        index_price=100_000.0,
        futures={"BTC-PERPETUAL": 100_100.0, "BTC-25DEC26": 105_000.0},
        options=(quote(110_000.0, "C", 0.55, bid=0.02, ask=0.03), quote(90_000.0, "P", 0.6, bid=None, ask=None)),
    )


def test_option_rows_keep_pricing_fields_and_anchor() -> None:
    """每行自带指数价：下游定价不必再去取一次 S0。"""
    rows = option_rows(chain("2026-10-15T08:00:00Z"))
    assert set(rows[0]) == {
        "instrument",
        "expiry",
        "kind",
        "strike",
        "mark_price",
        "mark_iv",
        "bid",
        "ask",
        "open_interest",
        "underlying",
        "currency",
        "index_price",
        "fetched_at",
    }
    assert rows[0]["index_price"] == 100_000.0
    assert rows[1]["bid"] is None and rows[1]["ask"] is None
    assert rows[0]["fetched_at"] == "2026-10-15T08:00:00+00:00"


def test_option_rows_drop_unquoted_strikes() -> None:
    """mark 为 0 的档（近到期无人报价）不入库：零价零 IV 对定价无用，留着只会让验收闸报。"""
    subject = DeribitChain(
        fetched_at=parse_instant("2026-10-15T08:00:00Z", "fetched_at"),
        currency="BTC",
        index_price=100_000.0,
        futures={"BTC-25DEC26": 105_000.0, "BTC-PERPETUAL": 0.0},
        options=(
            quote(110_000.0, "C", 0.55, bid=0.02, ask=0.03),
            OptionQuote(
                instrument="BTC-16OCT26-95000-C",
                expiry="16OCT26",
                kind="C",
                strike=95_000.0,
                mark_price=0.0,
                mark_iv=0.0,
                bid=0.0,
                ask=0.0,
                open_interest=0.0,
                underlying=100_000.0,
            ),
        ),
    )
    assert [row["instrument"] for row in option_rows(subject)] == ["BTC-25DEC26-110000-C"]
    assert [row["instrument"] for row in future_rows(subject)] == ["BTC-25DEC26"]


def test_future_rows_mark_perpetual_without_expiry() -> None:
    """永续记空到期码：写成 PERPETUAL 会让 carry 外推时把 T 当有效期限。"""
    rows = {row["instrument"]: row for row in future_rows(chain("2026-10-15T08:00:00Z"))}
    assert rows["BTC-PERPETUAL"]["expiry"] is None
    assert rows["BTC-25DEC26"]["expiry"] == "25DEC26"


def test_append_chain_dedupes_identical_snapshot(tmp_path: Path) -> None:
    """同一快照重跑一次：新增 0 行，全靠业务键拦住。"""
    subject = chain("2026-10-15T08:00:00Z")
    first = append_chain(subject, root=tmp_path)
    second = append_chain(subject, root=tmp_path)
    assert first[DERIBIT_OPTIONS] == (2, 0) and first[DERIBIT_FUTURES] == (2, 0)
    assert second[DERIBIT_OPTIONS] == (0, 2) and second[DERIBIT_FUTURES] == (0, 2)
    path = deribit_jsonl_path(DERIBIT_OPTIONS, "2026-10-15", root=tmp_path)
    assert len(list(read_jsonl(path))) == 2


def test_append_chain_keeps_second_snapshot_of_same_day(tmp_path: Path) -> None:
    """同一天第二轮快照要各留一行；否则日内多轮采集等于白采。"""
    append_chain(chain("2026-10-15T08:00:00Z"), root=tmp_path)
    added = append_chain(chain("2026-10-15T20:00:00Z"), root=tmp_path)[DERIBIT_OPTIONS][0]
    path = deribit_jsonl_path(DERIBIT_OPTIONS, "2026-10-15", root=tmp_path)
    stamps = {row["fetched_at"] for row in read_jsonl(path)}
    assert added == 2 and len(stamps) == 2


def test_append_chain_accepts_explicit_day(tmp_path: Path) -> None:
    """跨午夜补采：文件名由调用方给，不能一律用「现在」。"""
    append_chain(chain("2026-10-15T23:59:00Z"), day="2026-10-16", root=tmp_path)
    assert deribit_jsonl_path(DERIBIT_OPTIONS, "2026-10-16", root=tmp_path).exists()


def test_manifest_counts_rows_and_coverage(tmp_path: Path) -> None:
    """manifest 是验收与下游对齐的唯一凭据：行数、覆盖区间、来源都要对。"""
    append_chain(chain("2026-10-15T08:00:00Z"), root=tmp_path)
    manifest_root = tmp_path / "manifests"
    path = update_manifest(DERIBIT_OPTIONS, touched={"2026-10-15.jsonl"},
                           raw_root=tmp_path, manifest_root=manifest_root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "deribit_raw_options"
    assert manifest["source"] == "www.deribit.com"
    assert manifest["rows"] == 2 and manifest["files"] == 1
    assert manifest["coverage"] == {
        "start": "2026-10-15T08:00:00+00:00",
        "end": "2026-10-15T08:00:00+00:00",
    }
    assert manifest["entries"][0]["sha256"]


def test_manifest_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未知数据集"):
        update_manifest("spot", raw_root=tmp_path, manifest_root=tmp_path)
    assert set(KEY_FIELDS) == {DERIBIT_OPTIONS, DERIBIT_FUTURES}
