"""Deribit 公开行情：期权链、期货与指数价，阈值市场定价的风险中性参照。

只在研究链路里用（`scripts/polymarket_digital_study.py`），不进数据湖：Deribit 期权是
**反向（币本位）**合约——权利金按币计、行权价按美元计，换美元要乘指数价；深度尾部的
`mark_iv` 与买卖报价都很宽（140k/150k/200k 档相对价差 46~86%），只能当参照不能当成交价。

历史教训：经本机代理可达，但 curl 直连返回 000（必须走 `base.get_json` 的 requests + 重试）；
`public/get_book_summary_by_currency` 偶发 ReadTimeout + SSLError，所以一次读回整条链
（BTC 约 894 只），不要按到期日拆请求。
"""

from __future__ import annotations

import json
import math
import re
from calendar import month_abbr
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.data_sources.base import get_json, parse_instant

DERIBIT_API = "https://www.deribit.com/api/v2"
INDEX_NAMES = {"BTC": "btc_usd", "ETH": "eth_usd"}
KIND_OPTION = "option"
KIND_FUTURE = "future"
OPTION_PARTS = 4  # BTC-25DEC26-82000-C
FUTURE_PARTS = 2  # BTC-25DEC26
EXPIRY_HOUR_UTC = 8  # Deribit 一律 08:00 UTC 到期
YEAR_BASE = 2000  # 到期码里的两位年份
MONTHS = {name.upper(): number for number, name in enumerate(month_abbr) if name}
# 到期码的日可以是一位数（6NOV26）也可以是两位数（25DEC26）
EXPIRY_PATTERN = re.compile(r"^(\d{1,2})([A-Z]{3})(\d{2})$")
PERCENT = 100.0
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class OptionQuote:
    """一条期权：`mark_price` 是币本位权利金，`mark_iv` 已归一为小数。"""

    instrument: str
    expiry: str
    kind: str
    strike: float
    mark_price: float
    mark_iv: float | None
    bid: float | None
    ask: float | None
    open_interest: float
    underlying: float | None


@dataclass(frozen=True)
class ForwardPoint:
    """目标期限的远期：由最近期货推出的 carry 外推，`via` 记录用了哪张合约。"""

    forward: float
    carry: float
    via: str
    years: float


@dataclass(frozen=True)
class DeribitChain:
    """某一时刻的整条链：指数、期货（含永续）与全部期权。"""

    fetched_at: pd.Timestamp
    currency: str
    index_price: float
    futures: dict[str, float]
    options: tuple[OptionQuote, ...]

    def expiries(self) -> list[str]:
        """全部到期月份，按到期时刻排序。"""
        return sorted({q.expiry for q in self.options}, key=expiry_instant)

    def by_expiry(self, expiry: str) -> list[OptionQuote]:
        """某一到期月份的期权，按行权价升序。"""
        selected = (q for q in self.options if q.expiry == expiry)
        return sorted(selected, key=lambda q: q.strike)


def is_expiry_code(expiry: str) -> bool:
    """到期码形如 6NOV26 / 25DEC26；永续（PERPETUAL）之类不是——当成到期合约会算出无穷 carry。"""
    return EXPIRY_PATTERN.match(expiry) is not None


def expiry_instant(expiry: str) -> pd.Timestamp:
    """到期码 → 到期时刻。Deribit 一律 08:00 UTC。

    不用 `strptime`：那返回 `Timestamp | NaTType`，而且一位数日（6NOV26）会解析失败；
    这里用正则取出日/月/年再拼 ISO，非法码当场 ValueError。
    """
    match = EXPIRY_PATTERN.match(expiry)
    if match is None:
        raise ValueError(f"不是 Deribit 到期码：{expiry!r}")
    day, month, year = match.groups()
    stamp = (
        f"{YEAR_BASE + int(year):04d}-{MONTHS[month]:02d}-{int(day):02d}"
        f"T{EXPIRY_HOUR_UTC:02d}:00:00+00:00"
    )
    return parse_instant(stamp, f"到期码 {expiry}")


def _positive(value: Any) -> float | None:
    """0 当作「没有报价」：Deribit 对无买盘的深尾档返回 0。"""
    number = float(value or 0.0)
    return number if number > 0.0 else None


def parse_option(payload: dict[str, Any]) -> OptionQuote | None:
    """读 book summary 的一条；instrument_name 不合形就返回 None（上游偶发脏行）。"""
    name = str(payload.get("instrument_name", ""))
    parts = name.split("-")
    if len(parts) != OPTION_PARTS:
        return None
    try:
        strike = float(parts[2])
    except ValueError:
        return None
    mark_iv = _positive(payload.get("mark_iv"))
    return OptionQuote(
        instrument=name,
        expiry=parts[1],
        kind=parts[3],
        strike=strike,
        mark_price=float(payload.get("mark_price") or 0.0),
        mark_iv=mark_iv / PERCENT if mark_iv is not None else None,
        bid=_positive(payload.get("bid_price")),
        ask=_positive(payload.get("ask_price")),
        open_interest=float(payload.get("open_interest") or 0.0),
        underlying=_positive(payload.get("underlying_price")),
    )


def _result(payload: Any, label: str) -> Any:
    """Deribit 一律回 {"jsonrpc":…, "result":…}；缺 result 说明拿到的不是我们要的东西。"""
    if not isinstance(payload, dict) or "result" not in payload:
        raise RuntimeError(f"{label} 响应缺 result：{str(payload)[:200]}")
    return payload["result"]


def fetch_chain(session: requests.Session, currency: str = "BTC") -> DeribitChain:
    """整条链：指数 + 全部期货 + 全部期权，三个请求，不按到期日拆。"""
    if currency not in INDEX_NAMES:
        raise ValueError(f"不支持的币种 {currency}，只有 {sorted(INDEX_NAMES)}")
    summary = f"{DERIBIT_API}/public/get_book_summary_by_currency"
    index = _result(
        get_json(
            session,
            f"{DERIBIT_API}/public/get_index_price",
            {"index_name": INDEX_NAMES[currency]},
        ),
        "指数",
    )
    futures = _result(
        get_json(session, summary, {"currency": currency, "kind": KIND_FUTURE}), "期货"
    )
    options = _result(
        get_json(session, summary, {"currency": currency, "kind": KIND_OPTION}), "期权"
    )
    quotes = tuple(
        quote
        for quote in (parse_option(item) for item in options if isinstance(item, dict))
        if quote is not None
    )
    if not quotes:
        raise RuntimeError(f"{currency} 期权链为空，拒绝用空链当参照")
    return DeribitChain(
        fetched_at=pd.Timestamp.now(tz="UTC").floor("s"),
        currency=currency,
        index_price=float(index["index_price"]),
        futures={
            str(item["instrument_name"]): float(item["mark_price"])
            for item in futures
            if isinstance(item, dict) and item.get("mark_price")
        },
        options=quotes,
    )


def forward_curve(chain: DeribitChain, target: pd.Timestamp) -> ForwardPoint:
    """用「距目标最近的那张期货」定年化 carry，再把目标期限外推成远期。

    carry 定义在指数（现货）与期货之间：`carry = ln(F/index)/T`——对币本位合约来说
    这就是把美元远期搬到当下的成本，与 `one_touch` 的漂移约定配套。
    """
    target_years = float((target - chain.fetched_at).days) / DAYS_PER_YEAR
    if target_years <= 0.0:
        raise ValueError(f"目标到期 {target} 不在未来")
    best: tuple[float, str, float, float] | None = None
    for name, mark in chain.futures.items():
        parts = name.split("-")
        if len(parts) != FUTURE_PARTS or parts[0] != chain.currency or mark <= 0.0:
            continue
        if not is_expiry_code(parts[1]):
            continue
        years = (
            float((expiry_instant(parts[1]) - chain.fetched_at).days) / DAYS_PER_YEAR
        )
        if years <= 0.0:
            continue
        distance = abs(years - target_years)
        if best is None or distance < best[0]:
            best = (distance, name, mark, years)
    if best is None:
        raise RuntimeError("链里没有可用的期货合约，无法定 carry")
    _, via, mark, years = best
    carry = math.log(mark / chain.index_price) / years
    return ForwardPoint(
        forward=chain.index_price * math.exp(carry * target_years),
        carry=carry,
        via=via,
        years=years,
    )


def nearest_expiry(chain: DeribitChain, target: pd.Timestamp) -> str:
    """离目标到期最近的期权月份——跨期限搬微笑的误差比久期错配小。"""
    if not chain.options:
        raise RuntimeError("链里没有期权")
    return min(chain.expiries(), key=lambda e: abs(expiry_instant(e) - target))


def save_chain(chain: DeribitChain, path: Path) -> None:
    """落链快照（研究产物，不进数据湖），供离线复算同一张对照表。"""
    payload = {
        "fetched_at": chain.fetched_at.isoformat(),
        "currency": chain.currency,
        "index_price": chain.index_price,
        "futures": chain.futures,
        "options": [asdict(quote) for quote in chain.options],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_chain(path: Path) -> DeribitChain:
    """读回 `save_chain` 的产物。"""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return DeribitChain(
        fetched_at=parse_instant(payload["fetched_at"], "fetched_at"),
        currency=str(payload["currency"]),
        index_price=float(payload["index_price"]),
        futures={str(k): float(v) for k, v in payload["futures"].items()},
        options=tuple(OptionQuote(**item) for item in payload["options"]),
    )
