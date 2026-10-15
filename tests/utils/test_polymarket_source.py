"""Polymarket 端点封装的解析与降级口径（全部离线，用假 session）。

关键不变量：只在 tag=crypto 的活跃市场里选池、`clobTokenIds`/`outcomes` 的两种形态
都要认、价格窗口被服务端拒绝时逐档收缩而不是整段丢。
"""

from typing import Any, cast

import pytest
import requests

from src.utils.data_sources import base as dsb
from src.utils.data_sources import polymarket as pm


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status
        self.text = str(payload)[:200]

    def json(self) -> Any:
        return self.payload


class _FakeSession:
    """按调用顺序返回预设响应；可注入异常。"""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _Response:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self.responses.pop(0) if self.responses else _Response({})
        if isinstance(item, Exception):
            raise item
        return cast(_Response, item)


def _session(*responses: Any) -> requests.Session:
    return cast(requests.Session, _FakeSession(*responses))


def _market(
    slug: str,
    *,
    volume: float = 1000.0,
    condition: str | None = None,
    closed: bool = False,
    order_book: bool = True,
    tokens: str = '["tok-a", "tok-b"]',
) -> dict[str, Any]:
    market = {
        "slug": slug,
        "conditionId": condition or f"0x{slug}",
        "volumeNum": volume,
        "closed": closed,
        "enableOrderBook": order_book,
        "clobTokenIds": tokens,
        "outcomes": '["Yes", "No"]',
        "description": "x" * 5000,
        "resolutionSource": "https://www.binance.com/en/trade/BTC_USDT",
        "events": [{"id": 1}],
    }
    return market


def test_active_crypto_markets_filters_dedupes_and_sorts() -> None:
    """过滤掉已结算/不可下单/无 token 的；同一市场挂两个事件时保留成交额大的。"""
    events = [
        {
            "id": 1,
            "slug": "ev-1",
            "tags": [{"slug": "crypto"}, {"slug": "bitcoin"}],
            "markets": [
                _market("keep-big", volume=5000),
                _market("closed-one", closed=True),
                _market("no-book", order_book=False),
                _market("no-token", tokens="[]"),
            ],
        },
        {
            "id": 2,
            "slug": "ev-2",
            "tags": [{"slug": "crypto"}],
            "markets": [_market("keep-big", volume=9000, condition="0xkeep-big")],
        },
    ]
    picked = pm.active_crypto_markets(events, limit=10)
    assert [market["slug"] for market in picked] == ["keep-big"]
    assert picked[0]["volumeNum"] == 9000
    assert picked[0]["eventSlug"] == "ev-2"
    assert picked[0]["tags"] == ["crypto"]
    # 规则字段必须留下：它是 price → payoff 口径（one-touch、1 分钟 High/Low、窗口起点）的凭证
    assert picked[0]["resolutionSource"] == "https://www.binance.com/en/trade/BTC_USDT"
    assert picked[0]["description"]
    assert "events" not in picked[0]


def test_active_crypto_markets_respects_limit() -> None:
    events = [
        {
            "id": 1,
            "slug": "ev",
            "tags": [],
            "markets": [_market(f"m{i}", volume=float(i)) for i in range(5)],
        }
    ]
    picked = pm.active_crypto_markets(events, limit=3)
    assert [market["slug"] for market in picked] == ["m4", "m3", "m2"]


def test_clob_token_ids_and_outcome_names_accept_both_shapes() -> None:
    assert pm.clob_token_ids({"clobTokenIds": '["a", "b"]'}) == ["a", "b"]
    assert pm.clob_token_ids({"clobTokenIds": ["a", "b"]}) == ["a", "b"]
    assert pm.clob_token_ids({"clobTokenIds": "not-json"}) == []
    assert pm.outcome_names({"outcomes": '["Yes", "No"]'}) == ["Yes", "No"]
    assert pm.outcome_names({"outcomes": ["Up", "Down"]}) == ["Up", "Down"]
    assert pm.outcome_names({}) == []


def test_price_points_shrinks_window_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """服务端嫌窗口太长（400）时降到 3 天窗口，而不是放弃这个市场。"""
    monkeypatch.setattr(dsb.time, "sleep", lambda _: None)
    session = _session(
        _Response({"error": "interval is too long"}, status=400),
        _Response({"history": [{"t": 100, "p": 0.25}, {"t": 200, "p": 0.5}]}),
    )
    points = pm.price_points(session, "tok", now=1_000_000)
    assert points == [{"t": 100, "p": 0.25}, {"t": 200, "p": 0.5}]


def test_price_points_raises_when_all_windows_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dsb.time, "sleep", lambda _: None)
    session = _session(
        *[_Response({"error": "too long"}, status=400) for _ in range(3)]
    )
    with pytest.raises(RuntimeError):
        pm.price_points(session, "tok", now=1_000_000)


def test_price_points_reraises_non_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """404 之类的错误不是「窗口太长」，不该被降级逻辑吞掉。"""
    monkeypatch.setattr(dsb.time, "sleep", lambda _: None)
    session = _session(_Response({"error": "not found"}, status=404))
    with pytest.raises(pm.SourceHTTPError):
        pm.price_points(session, "tok", now=1_000_000)


def test_get_json_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """代理切节点会让 SSL 直接断，重试是常态而不是异常处理。"""
    monkeypatch.setattr(dsb.time, "sleep", lambda _: None)
    session = _session(
        requests.ConnectionError("proxy down"),
        _Response({"ok": True}),
    )
    assert pm.get_json(session, pm.GAMMA_EVENTS, {}) == {"ok": True}


def test_trades_page_ignores_non_list_payload() -> None:
    session = _session(_Response({"error": "nope"}))
    assert pm.trades_page(session, "0xabc", limit=10, offset=0) == []
