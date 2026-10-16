"""
src/utils/data_sources/okx_copytrading.py
OKX 跟单（copy trading）公开端点的 transport 层：榜单分页 + 跟单统计解析。

端点与字段语义均是 2026-10-16 实测核对（无需鉴权；本机直连不通，统一走
`create_session()` 的代理开关）：

- `public-lead-traders` 每页固定 10 名带单员，`page` 从 1 起，越界页返回空；
  当天全榜 246 人（第 26 页起为空）。响应里的 `dataVer` 是榜单版本号，平台在
  本地时间 23:00 前后刷新，同版本重跑拿到的是同一份快照，落盘去重靠它。
- `public-stats` 的 `lastDays` 只接受 1/2/3，分别是 7/30/90 天滚动口径（实测
  传 7/14/30/90 都报 51000 参数错）；取 3 时 `winRatio` 与榜单字段逐位一致。
- `pnlRatio ≈ pnl / investAmt`：**交易员自有投入资金**的累计收益率（217 个非退化
  样本里 97% 相对误差 < 1%），不是跟单者收益；跟单者盈亏看 `curCopyTraderPnl`，
  跟单资金规模看 `aum`，累计跟单人次看 `accCopyTraderNum`。
- 上游把数字全写成字符串、空值给 ""；这里解析失败一律抛错，不静默补 0。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from src.utils.data_sources.base import get_json

BASE_URL = "https://www.okx.com/api/v5/copytrading"
LEAD_TRADERS_PATH = "/public-lead-traders"
STATS_PATH = "/public-stats"
INST_TYPE_SWAP = "SWAP"
PAGE_SIZE = 10
MAX_PAGES = 60
REQUEST_PAUSE = 0.12
STATS_WINDOW_90D = 3


@dataclass(frozen=True)
class PnlPoint:
    """榜单附带的日度收益点：`beginTs`（epoch 毫秒）当天的累计收益率。"""

    begin_ts: int
    pnl_ratio: float


@dataclass(frozen=True)
class TraderRank:
    """榜单里的一行（标量字段 + 上游给的日度收益序列）。"""

    unique_code: str
    nick_name: str
    ccy: str
    aum: float
    pnl: float
    pnl_ratio: float
    win_ratio: float
    lead_days: int
    copy_trader_num: int
    acc_copy_trader_num: int
    max_copy_trader_num: int
    instrument_num: int
    history: tuple[PnlPoint, ...]


@dataclass(frozen=True)
class TraderStats:
    """`public-stats` 的 90 天滚动口径统计（跟单者盈亏与交易员自有投入资金）。"""

    unique_code: str
    ccy: str
    invest_amt: float
    cur_copy_trader_pnl: float
    avg_sub_pos_notional: float
    profit_days: int
    loss_days: int
    win_ratio: float


@dataclass(frozen=True)
class Leaderboard:
    """一次榜单快照：版本号 + 抓取时刻 + 全部带单员。"""

    data_ver: str
    inst_type: str
    fetched_at: pd.Timestamp
    ranks: tuple[TraderRank, ...]


def parse_rank(raw: Mapping[str, Any]) -> TraderRank:
    """榜单行 → `TraderRank`；缺 uniqueCode 或数字解析失败当场抛错。"""
    code = str(raw.get("uniqueCode") or "")
    if not code:
        raise ValueError(f"榜单行缺 uniqueCode：{dict(raw)!r}")
    history = tuple(
        sorted(
            (parse_point(item) for item in raw.get("pnlRatios") or []),
            key=lambda point: point.begin_ts,
        )
    )
    return TraderRank(
        unique_code=code,
        nick_name=str(raw.get("nickName") or ""),
        ccy=str(raw.get("ccy") or ""),
        aum=_number(raw.get("aum"), f"{code}.aum"),
        pnl=_number(raw.get("pnl"), f"{code}.pnl"),
        pnl_ratio=_number(raw.get("pnlRatio"), f"{code}.pnlRatio"),
        win_ratio=_number(raw.get("winRatio"), f"{code}.winRatio"),
        lead_days=_int(raw.get("leadDays"), f"{code}.leadDays"),
        copy_trader_num=_int(raw.get("copyTraderNum"), f"{code}.copyTraderNum"),
        acc_copy_trader_num=_int(
            raw.get("accCopyTraderNum"), f"{code}.accCopyTraderNum"
        ),
        max_copy_trader_num=_int(
            raw.get("maxCopyTraderNum"), f"{code}.maxCopyTraderNum"
        ),
        instrument_num=len(raw.get("traderInsts") or []),
        history=history,
    )


def parse_point(raw: Mapping[str, Any]) -> PnlPoint:
    """日度收益点 → `PnlPoint`。"""
    return PnlPoint(
        begin_ts=_int(raw.get("beginTs"), "beginTs"),
        pnl_ratio=_number(raw.get("pnlRatio"), "pnlRatio"),
    )


def parse_stats(unique_code: str, raw: Mapping[str, Any]) -> TraderStats:
    """跟单统计 → `TraderStats`。"""
    return TraderStats(
        unique_code=unique_code,
        ccy=str(raw.get("ccy") or ""),
        invest_amt=_number(raw.get("investAmt"), f"{unique_code}.investAmt"),
        cur_copy_trader_pnl=_number(
            raw.get("curCopyTraderPnl"), f"{unique_code}.curCopyTraderPnl"
        ),
        avg_sub_pos_notional=_number(
            raw.get("avgSubPosNotional"), f"{unique_code}.avgSubPosNotional"
        ),
        profit_days=_int(raw.get("profitDays"), f"{unique_code}.profitDays"),
        loss_days=_int(raw.get("lossDays"), f"{unique_code}.lossDays"),
        win_ratio=_number(raw.get("winRatio"), f"{unique_code}.winRatio"),
    )


def fetch_leaderboard(
    session: requests.Session,
    *,
    inst_type: str = INST_TYPE_SWAP,
    max_pages: int = MAX_PAGES,
) -> Leaderboard:
    """翻完整个榜单；出现重复 uniqueCode（分页竞态）或空榜即抛错，拒绝落盘半份。"""
    ranks: list[TraderRank] = []
    data_ver = ""
    page = 1
    while page <= max_pages:
        entry = _first(session, LEAD_TRADERS_PATH, {"instType": inst_type, "page": page})
        if entry is None:
            break
        data_ver = str(entry.get("dataVer") or data_ver)
        page_ranks = list(entry.get("ranks") or [])
        if not page_ranks:
            break
        ranks.extend(parse_rank(item) for item in page_ranks)
        if len(page_ranks) < PAGE_SIZE:
            break
        page += 1
        time.sleep(REQUEST_PAUSE)
    if not ranks:
        raise ValueError(f"{inst_type} 榜单为空（接口变更或代理异常）")
    if not data_ver:
        raise ValueError("榜单缺 dataVer，无法给快照去重")
    codes = {rank.unique_code for rank in ranks}
    if len(codes) != len(ranks):
        raise ValueError("榜单出现重复 uniqueCode（分页竞态），拒绝落盘")
    return Leaderboard(
        data_ver=data_ver,
        inst_type=inst_type,
        fetched_at=pd.Timestamp.now(tz="UTC"),
        ranks=tuple(ranks),
    )


def fetch_stats(
    session: requests.Session,
    unique_code: str,
    *,
    inst_type: str = INST_TYPE_SWAP,
    last_days: int = STATS_WINDOW_90D,
) -> TraderStats:
    """单个带单员的跟单统计（默认 90 天滚动口径）。"""
    entry = _first(
        session,
        STATS_PATH,
        {"instType": inst_type, "uniqueCode": unique_code, "lastDays": last_days},
    )
    if entry is None:
        raise ValueError(f"带单员 {unique_code} 的统计为空")
    return parse_stats(unique_code, entry)


def _first(
    session: requests.Session, path: str, params: dict[str, Any]
) -> Mapping[str, Any] | None:
    """取 `data[0]`；OKX 的业务错（code != "0"）当作异常抛出，不当成空数据。"""
    payload = get_json(session, f"{BASE_URL}{path}", params)
    if not isinstance(payload, Mapping) or payload.get("code") != "0":
        raise ValueError(f"{path} 返回异常：{str(payload)[:200]}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    return first if isinstance(first, Mapping) else None


def _number(raw: Any, label: str) -> float:
    """上游数字字段（字符串形态）→ float；缺失或非法当场报错。"""
    if isinstance(raw, bool):
        raise ValueError(f"{label} 不是数字：{raw!r}")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except ValueError as error:
            raise ValueError(f"{label} 不是数字：{raw!r}") from error
    raise ValueError(f"{label} 缺失或非法：{raw!r}")


def _int(raw: Any, label: str) -> int:
    """整数口径字段；浮点字符串（"228.0"）也算合法。"""
    return int(_number(raw, label))
