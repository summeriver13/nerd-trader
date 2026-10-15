"""
逐市场采集：价格历史 + 成交明细，带一次整只重试。

单独成模块是为了让 CLI 保持薄壳：`scripts/fetch_polymarket.py` 只管选池、并发与落盘，
这里只管「怎么把一个市场抓全」。

重试的理由是实测的：本机代理每轮会让约 5% 的市场丢一条腿（SSL 断开、5 次重试全挂），
而价格历史一旦错过窗口就永久缺失；写盘靠业务键去重，所以重试只会补缺口、不会写重复行。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import requests

from src.config.paths import POLYMARKET_PRICES, POLYMARKET_TRADES, polymarket_jsonl_path
from src.utils.data_sources import polymarket as pm
from src.utils.data_sources.base import create_session
from src.utils.jsonl_store import append_jsonl
from src.utils.polymarket_store import KEY_FIELDS

RETRY_PAUSE_SECONDS = 3.0


@dataclass(frozen=True)
class CollectSpec:
    """一轮采集的口径。"""

    price_window_days: int = pm.PRICE_WINDOW_DAYS
    fidelity: int = pm.DEFAULT_FIDELITY
    trade_pages: int = 8
    trade_limit: int = 500
    skip_prices: bool = False
    skip_trades: bool = False


@dataclass(frozen=True)
class MarketResult:
    """单个市场的采集结果（写盘在 worker 内完成，主线程只收统计）。"""

    slug: str
    prices_added: int
    trades_added: int
    prices_skipped: int
    trades_skipped: int
    notes: list[str] = field(default_factory=list)


def _failure(error: Exception) -> str:
    """失败备注只留前 60 字：代理报错带整条 URL，日志里不需要全文。"""
    return str(error)[:60]


def collect_prices(
    session: requests.Session, market: dict[str, Any], spec: CollectSpec
) -> MarketResult:
    """两个 token 各取一次价格历史，按 (token, t) 去重后追加。"""
    tokens = pm.clob_token_ids(market)
    outcomes = pm.outcome_names(market)
    rows: list[dict[str, Any]] = []
    now = int(time.time())
    notes: list[str] = []
    for index, token in enumerate(tokens):
        try:
            points = pm.price_points(
                session,
                token,
                now=now,
                window_days=spec.price_window_days,
                fidelity=spec.fidelity,
            )
        except (RuntimeError, requests.RequestException) as error:
            notes.append(f"价格失败[{index}]: {_failure(error)}")
            continue
        outcome = outcomes[index] if index < len(outcomes) else f"outcome{index}"
        rows.extend(
            {
                "slug": market.get("slug"),
                "conditionId": market.get("conditionId"),
                "token": token,
                "outcome": outcome,
                "outcomeIndex": index,
                "t": point["t"],
                "p": point["p"],
            }
            for point in points
        )
    path = polymarket_jsonl_path(POLYMARKET_PRICES, str(market.get("slug")))
    added, skipped = append_jsonl(path, rows, KEY_FIELDS[POLYMARKET_PRICES])
    return MarketResult(str(market.get("slug")), added, 0, skipped, 0, notes)


def collect_trades(
    session: requests.Session,
    market: dict[str, Any],
    spec: CollectSpec,
    result: MarketResult,
) -> MarketResult:
    """成交明细从最新往前翻，翻到整页都已落盘即停（增量水位）。"""
    path = polymarket_jsonl_path(POLYMARKET_TRADES, str(market.get("slug")))
    added_total = 0
    skipped_total = 0
    notes = list(result.notes)
    for page in range(spec.trade_pages):
        offset = page * spec.trade_limit
        try:
            trades = pm.trades_page(
                session,
                str(market.get("conditionId") or ""),
                limit=spec.trade_limit,
                offset=offset,
            )
        except (RuntimeError, requests.RequestException) as error:
            notes.append(f"成交失败[offset={offset}]: {_failure(error)}")
            break
        if not trades:
            break
        added, skipped = append_jsonl(
            path, _trade_rows(trades, market), KEY_FIELDS[POLYMARKET_TRADES]
        )
        added_total += added
        skipped_total += skipped
        if added == 0 or len(trades) < spec.trade_limit:
            break
    return replace(
        result, trades_added=added_total, trades_skipped=skipped_total, notes=notes
    )


def _trade_rows(
    trades: Sequence[dict[str, Any]], market: dict[str, Any]
) -> list[dict[str, Any]]:
    """成交明细白名单 + 本轮采集时刻（上游一条约 900B，白名单后约 250B）。"""
    fetched_at = market.get("fetched_at")
    return [
        {**pm.pick(trade, pm.TRADE_FIELDS), "fetched_at": fetched_at}
        for trade in trades
    ]


def _collect_once(market: dict[str, Any], spec: CollectSpec) -> MarketResult:
    """一个市场的完整采集；每个 worker 自建 session（Session 不保证线程安全）。"""
    session = create_session()
    result = MarketResult(str(market.get("slug")), 0, 0, 0, 0)
    if not spec.skip_prices:
        result = collect_prices(session, market, spec)
    if not spec.skip_trades:
        result = collect_trades(session, market, spec, result)
    return result


def collect_market(market: dict[str, Any], spec: CollectSpec) -> MarketResult:
    """带一次整只重试的采集：第一遍有备注（多半是代理抖动）就再跑一遍。

    统计量取两遍之和：报告的是「这一轮实际写入了多少行」，而不是最后一遍的增量。
    """
    result = _collect_once(market, spec)
    if not result.notes:
        return result
    time.sleep(RETRY_PAUSE_SECONDS)
    retry = _collect_once(market, spec)
    return replace(
        retry,
        prices_added=result.prices_added + retry.prices_added,
        trades_added=result.trades_added + retry.trades_added,
        prices_skipped=result.prices_skipped + retry.prices_skipped,
        trades_skipped=result.trades_skipped + retry.trades_skipped,
        notes=[*result.notes, *retry.notes],
    )
