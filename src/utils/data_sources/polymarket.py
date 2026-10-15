"""
Polymarket 数据源：Gamma（市场/事件）、CLOB（价格历史）、Data API（成交明细）。

历史回填已被实测封死（容器 ADR-0007）：已结算市场价格取不到、subgraph 已关、`/trades`
无时间窗分页，所以这里只有前向采集用的三个端点；落盘见 `src/utils/polymarket_store.py`。

字段名保留上游原样（camelCase），只做白名单裁剪：`icon` 与 `events` 嵌套在一条市场里
合计 50KB+，而研究只用到白名单里那些；`description`/`resolutionSource` 例外保留，
因为结算规则（判定源、1 分钟 High/Low、窗口起点）是定价口径的唯一凭证。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import requests

from src.utils.data_sources.base import SourceHTTPError, get_json

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_PRICE_HISTORY = "https://clob.polymarket.com/prices-history"
DATA_API_TRADES = "https://data-api.polymarket.com/trades"
CRYPTO_TAG_SLUG = "crypto"

REQUEST_TIMEOUT = 60.0
RETRY_ATTEMPTS = 5
RETRY_BACKOFF = 1.5
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
SECONDS_PER_DAY = 86_400
# 价格历史只在活跃市场存活期内可取（约 30 天），单请求窗口给 7 天；
# 遇到 400 `interval is too long` 逐档收缩——宁可少抓几天，也不能整段丢。
PRICE_WINDOW_DAYS = 7
PRICE_WINDOW_FALLBACKS = (3, 1)
DEFAULT_FIDELITY = 60

# 快照白名单：Gamma 的市场/事件对象里 icon、events 嵌套是体积大头，裁掉。
# description 与 resolutionSource 必须留：结算规则（判定源、1 分钟 High/Low、窗口起点）
# 只写在这里，是「按 one-touch 口径定价」合法性的唯一凭证；各约 1.2KB/市场/天，
# 相对 daily trades 50MB/天 可忽略。
MARKET_FIELDS = (
    "id",
    "slug",
    "question",
    "description",
    "resolutionSource",
    "conditionId",
    "questionID",
    "outcomes",
    "outcomePrices",
    "clobTokenIds",
    "lastTradePrice",
    "bestBid",
    "bestAsk",
    "spread",
    "volumeNum",
    "volumeClob",
    "volume1wk",
    "volume1mo",
    "liquidity",
    "liquidityClob",
    "startDate",
    "endDate",
    "closed",
    "closedTime",
    "active",
    "acceptingOrders",
    "enableOrderBook",
    "archived",
    "restricted",
    "negRisk",
    "negRiskMarketID",
    "groupItemTitle",
    "marketType",
    "umaResolutionStatus",
    "umaEndDate",
    "resolvedBy",
    "orderPriceMinTickSize",
    "orderMinSize",
    "feesEnabled",
    "takerBaseFee",
    "makerBaseFee",
    "clobRewards",
    "rewardsMinSize",
    "rewardsMaxSpread",
    "oneHourPriceChange",
    "oneDayPriceChange",
    "oneWeekPriceChange",
    "oneMonthPriceChange",
    "createdAt",
    "updatedAt",
)
# 成交明细白名单：上游一条带 icon/profileImage/bio/pseudonym 等展示字段，约 900B，
# 40 只市场一天就能写成 50MB；下面这些是校准研究实际要用的（钱包地址保留：不可重取）。
TRADE_FIELDS = (
    "transactionHash",
    "timestamp",
    "conditionId",
    "asset",
    "outcome",
    "outcomeIndex",
    "side",
    "price",
    "size",
    "slug",
    "proxyWallet",
)
EVENT_FIELDS = (
    "id",
    "slug",
    "ticker",
    "title",
    "startDate",
    "endDate",
    "closed",
    "active",
    "volume",
    "liquidity",
    "openInterest",
    "negRisk",
    "commentCount",
)


def pick(source: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """按白名单裁剪字段，缺字段不补（上游删字段要能被看见）。"""
    return {name: source[name] for name in fields if name in source}


def tag_slugs(event: dict[str, Any]) -> list[str]:
    """事件的标签 slug 列表（`tag_slug=crypto` 枚举出来的事件都带这一族标签）。"""
    return [
        str(tag["slug"])
        for tag in event.get("tags") or []
        if isinstance(tag, dict) and tag.get("slug")
    ]


def volume_of(market: dict[str, Any]) -> float:
    """市场成交额（美元）；字段缺失按 0 处理，排序与选池都靠它。"""
    return float(market.get("volumeNum") or 0.0)


def fetch_crypto_events(
    session: requests.Session, *, pages: int, page_size: int
) -> list[dict[str, Any]]:
    """按 `tag_slug=crypto` 枚举事件（含内嵌市场），按成交额降序分页。"""
    events: list[dict[str, Any]] = []
    for page in range(pages):
        batch = get_json(
            session,
            GAMMA_EVENTS,
            {
                "tag_slug": CRYPTO_TAG_SLUG,
                "closed": "false",
                "limit": page_size,
                "offset": page * page_size,
                "order": "volume",
                "ascending": "false",
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        events.extend(item for item in batch if isinstance(item, dict))
    return events


def active_crypto_markets(
    events: Sequence[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """事件树 → 活跃且可下单的市场，按成交额降序取前 limit 只。

    同一市场可能挂在多个事件下，按 conditionId 去重并保留成交额最大的那份。
    """
    best: dict[str, dict[str, Any]] = {}
    for event in events:
        for market in event.get("markets") or []:
            if market.get("closed") or not market.get("enableOrderBook"):
                continue
            if not clob_token_ids(market):
                continue
            record = pick(market, MARKET_FIELDS)
            record["eventId"] = event.get("id")
            record["eventSlug"] = event.get("slug")
            record["tags"] = tag_slugs(event)
            key = str(record.get("conditionId") or record.get("slug"))
            if key not in best or volume_of(record) > volume_of(best[key]):
                best[key] = record
    return sorted(best.values(), key=volume_of, reverse=True)[:limit]


def _as_list(raw: Any) -> list[str]:
    """Gamma 的列表字段有时是 JSON 字符串、有时是列表，统一成字符串列表。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in raw] if isinstance(raw, list) else []


def clob_token_ids(market: dict[str, Any]) -> list[str]:
    """两个 outcome 对应的 ERC1155 token id（价格历史按 token 取）。"""
    return _as_list(market.get("clobTokenIds"))


def outcome_names(market: dict[str, Any]) -> list[str]:
    """outcome 名称，与 `clob_token_ids` 同序（用于给价格点打标签）。"""
    return _as_list(market.get("outcomes"))


def price_points(
    session: requests.Session,
    token: str,
    *,
    now: int,
    window_days: int = PRICE_WINDOW_DAYS,
    fidelity: int = DEFAULT_FIDELITY,
) -> list[dict[str, Any]]:
    """单个 token 的成交价格历史（CLOB），窗口过长时逐档收缩。"""
    windows = (window_days, *PRICE_WINDOW_FALLBACKS)
    last_error: SourceHTTPError | None = None
    for days in windows:
        try:
            payload = get_json(
                session,
                CLOB_PRICE_HISTORY,
                {
                    "market": token,
                    "startTs": now - days * SECONDS_PER_DAY,
                    "endTs": now,
                    "fidelity": fidelity,
                },
            )
        except SourceHTTPError as error:
            if error.status != HTTP_BAD_REQUEST:
                raise
            last_error = error
            continue
        history = payload.get("history") if isinstance(payload, dict) else None
        return [
            {"t": int(point["t"]), "p": float(point["p"])}
            for point in history or []
            if isinstance(point, dict) and "t" in point and "p" in point
        ]
    raise RuntimeError(f"价格窗口 {windows} 全被拒（{last_error}）")


def trades_page(
    session: requests.Session, condition_id: str, *, limit: int, offset: int
) -> list[dict[str, Any]]:
    """成交明细一页（Data API，limit ≤ 1000；单市场 offset 上限约 1~2 万笔）。"""
    payload = get_json(
        session,
        DATA_API_TRADES,
        {"market": condition_id, "limit": limit, "offset": offset},
    )
    if not isinstance(payload, list):
        return []
    return [trade for trade in payload if isinstance(trade, dict)]
