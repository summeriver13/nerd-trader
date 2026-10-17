"""采集层口径：白名单落盘、整只重试、增量水位。

这些断言护的是「数据能不能重来」——写盘去重让重试无害，白名单让 900B/条的成交
变成 250B/条，而增量水位决定每天重复运行的成本。
"""

from pathlib import Path
from typing import Any, cast

import pytest
import requests

from src.utils import polymarket_collect as pc
from src.utils.jsonl_store import read_jsonl

PRICE_HISTORY = [{"t": 100, "p": 0.25}, {"t": 200, "p": 0.5}]
TRADE = {
    "transactionHash": "0xtx",
    "asset": "tok-a",
    "timestamp": 150,
    "side": "BUY",
    "price": 0.3,
    "size": 10.0,
    "outcome": "Yes",
    "outcomeIndex": 0,
    "slug": "m",
    "conditionId": "0xcond",
    "proxyWallet": "0xwallet",
    "icon": "https://example.com/a.png",
    "profileImage": "https://example.com/b.png",
    "bio": "x" * 200,
}


def _market() -> dict[str, Any]:
    return {
        "slug": "m",
        "conditionId": "0xcond",
        "clobTokenIds": '["tok-a", "tok-b"]',
        "outcomes": '["Yes", "No"]',
        "fetched_at": "2026-10-14T00:00:00+00:00",
    }


@pytest.fixture
def raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把落盘重定向到临时目录：测试不碰仓库里的真数据。"""

    def fake_path(dataset: str, name: str, **_: Any) -> Path:
        return tmp_path / dataset / f"{name}.jsonl"

    monkeypatch.setattr(pc, "polymarket_jsonl_path", fake_path)
    monkeypatch.setattr(pc, "create_session", lambda: cast(requests.Session, object()))
    monkeypatch.setattr(pc.time, "sleep", lambda _: None)
    return tmp_path


def test_collect_prices_tags_outcome_and_dedupes(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_points(session: Any, token: str, **_: Any) -> list[dict[str, Any]]:
        calls.append(token)
        return PRICE_HISTORY

    monkeypatch.setattr(pc.pm, "price_points", fake_points)
    first = pc.collect_prices(
        cast(requests.Session, object()), _market(), pc.CollectSpec()
    )
    second = pc.collect_prices(
        cast(requests.Session, object()), _market(), pc.CollectSpec()
    )
    assert (first.prices_added, second.prices_added) == (4, 0)
    assert second.prices_skipped == 4
    assert calls == ["tok-a", "tok-b", "tok-a", "tok-b"]
    rows = list(read_jsonl(raw_root / "prices" / "m.jsonl"))
    assert {(row["outcome"], row["t"], row["p"]) for row in rows} == {
        ("Yes", 100, 0.25),
        ("Yes", 200, 0.5),
        ("No", 100, 0.25),
        ("No", 200, 0.5),
    }


def test_collect_trades_whitelists_fields_and_stops_at_duplicate_page(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages: list[int] = []

    def fake_trades(
        session: Any, condition: str, *, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        pages.append(offset)
        return [TRADE] if offset == 0 else []

    monkeypatch.setattr(pc.pm, "trades_page", fake_trades)
    spec = pc.CollectSpec(trade_pages=3, trade_limit=1)
    result = pc.collect_trades(
        cast(requests.Session, object()),
        _market(),
        spec,
        pc.MarketResult("m", 0, 0, 0, 0),
    )
    assert result.trades_added == 1
    assert pages == [0, 1]  # 第二页空 → 停
    stored = (raw_root / "trades" / "m.jsonl").read_text(encoding="utf-8")
    assert "profileImage" not in stored and "bio" not in stored
    assert '"proxyWallet": "0xwallet"' in stored


def test_collect_market_retries_once_then_keeps_notes(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第一遍丢一条腿、第二遍补齐：重试只补缺口，读数记录「曾经失败过」。"""
    attempts = {"prices": 0}

    def flaky_points(session: Any, token: str, **_: Any) -> list[dict[str, Any]]:
        attempts["prices"] += 1
        if attempts["prices"] == 1:
            raise RuntimeError("SSL 断了")
        return PRICE_HISTORY

    monkeypatch.setattr(pc.pm, "price_points", flaky_points)
    result = pc.collect_market(_market(), pc.CollectSpec(skip_trades=True))
    assert result.prices_added == 4
    assert result.prices_skipped == 2  # 重试把第一遍已落的 tok-b 判定为重复
    assert attempts["prices"] == 4  # 两遍各打两次（两个 token）
    assert any("价格失败" in note for note in result.notes)


def test_collect_market_skips_retry_when_clean(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        pc.pm,
        "price_points",
        lambda session, token, **_: calls.append(1) or PRICE_HISTORY,
    )
    result = pc.collect_market(_market(), pc.CollectSpec(skip_trades=True))
    assert result.notes == []
    assert len(calls) == 2  # 两个 token，各一次，没有重试
