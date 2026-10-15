"""阈值阶梯的装载与口径自检：从 Polymarket 快照行抽出「同一张阶梯」。

价格语义已核实（docs/polymarket/README.md §七）：判定源是 Binance 的 **1 分钟 High/Low**、
`immediately resolve` 即触碰即结算。本模块只做两件事——把档位抽出来、把文本里的
定价前提核一遍；定价在 `polymarket_digital.py`。

历史教训：
- slug 尾部带版本数字段（`-113-862-581`），必须用锚定前缀抽行权价，否则产出
  800/1000/2250 这类伪档并触发假的阶梯单调性违反；
- 窗口起点是各市场自己的创建时刻（`from-july-21`/`from-august-24` 变体与 `hit` 族
  起点各不相同），起点不同的档位不是同一张阶梯，不能混进同一次拟合。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.data_sources.base import parse_instant

THOUSAND = 1000.0

# 锚定前缀：`will-bitcoin-reach-90000-by-…` 与 `will-bitcoin-dip-to-60k-from-…` 都认
SLUG_PATTERN = re.compile(
    r"^will-(bitcoin|btc|ethereum|eth)-(reach|dip-to|hit)-(\d+)(k?)(?:-|$)"
)
ASSETS = {"bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "eth": "ethereum"}
UP_FAMILIES = frozenset({"reach", "hit"})
WINDOW_CREATION = "creation"
WINDOW_UNBOUNDED = "unbounded"

GRANULARITY_PATTERN = re.compile(r"(\d+)[- ]minute candle")
# 上游同一族的市场混用直引号与弯引号（“Low”），只用直引号会漏掉一半；
# 另有模板同时写 "High" or "Low"（方向由标题定），故价格类型可以是 both
PRICE_TYPE_PATTERN = re.compile(r'["“”](High|Low)["“”]')
PRICE_TYPE_BOTH = "both"
CREATION_PATTERN = re.compile(r"the creation of this market")
HARDCODED_WINDOW_PATTERN = re.compile(r"between\s+([A-Z][a-z]+ \d{1,2}, \d{4}),")
BINANCE_HINT = "Binance"
IMMEDIATE_HINT = "immediately resolve"
EXPECTED_GRANULARITY_MINUTES = 1


@dataclass(frozen=True)
class TouchRule:
    """从结算文本里抽出的定价前提；任何一项对不上就不该拿公式去套。"""

    price_type: str
    granularity_minutes: int
    immediate: bool
    window_from: str


@dataclass(frozen=True)
class LadderRow:
    """一档阈值市场。`direction` 为 up（碰上方）或 down（碰下方）。"""

    slug: str
    asset: str
    strike: float
    family: str
    direction: str
    price: float
    created_at: pd.Timestamp
    end_at: pd.Timestamp
    window_from: str
    volume: float
    spread: float | None
    bid: float | None
    ask: float | None
    rule: TouchRule | None

    @property
    def ladder_key(self) -> str:
        """同一张阶梯的分组键：资产 + 族别 + 窗口起点——任一不同就不在同一张阶梯上。"""
        return f"{self.asset}/{self.family}/{self.window_from}"


def parse_touch_rule(description: str) -> TouchRule | None:
    """抽出判定源/价格类型/监控粒度/窗口起点；认不出来返回 None，宁可显式失败也不猜。"""
    granularity = GRANULARITY_PATTERN.search(description)
    price_types = set(PRICE_TYPE_PATTERN.findall(description))
    if granularity is None or not price_types or BINANCE_HINT not in description:
        return None
    price_type = price_types.pop() if len(price_types) == 1 else PRICE_TYPE_BOTH
    if CREATION_PATTERN.search(description):
        window_from = WINDOW_CREATION
    else:
        hardcoded = HARDCODED_WINDOW_PATTERN.search(description)
        window_from = (
            WINDOW_UNBOUNDED
            if hardcoded is None
            else str(pd.Timestamp(hardcoded.group(1)).date())
        )
    return TouchRule(
        price_type=price_type,
        granularity_minutes=int(granularity.group(1)),
        immediate=IMMEDIATE_HINT in description,
        window_from=window_from,
    )


def _yes_price(raw: Any) -> float | None:
    """Gamma 的 outcomePrices 是 JSON 字符串（也可能是列表），取 YES 那一侧。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        return None
    try:
        return float(raw[0])
    except (TypeError, ValueError):
        return None


def _number(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def ladder_row(snapshot: dict[str, Any]) -> LadderRow | None:
    """从一条 markets 快照里读出阈值档位；非阈值族或缺价返回 None。"""
    match = SLUG_PATTERN.match(str(snapshot.get("slug", "")))
    price = _yes_price(snapshot.get("outcomePrices"))
    if match is None or price is None:
        return None
    family = match.group(2)
    rule = parse_touch_rule(str(snapshot.get("description") or ""))
    return LadderRow(
        slug=str(snapshot["slug"]),
        asset=ASSETS[match.group(1)],
        strike=float(match.group(3)) * (THOUSAND if match.group(4) else 1.0),
        family=family,
        direction="up" if family in UP_FAMILIES else "down",
        price=price,
        created_at=parse_instant(snapshot["createdAt"], "createdAt"),
        end_at=parse_instant(snapshot["endDate"], "endDate"),
        window_from=WINDOW_UNBOUNDED if rule is None else rule.window_from,
        volume=float(snapshot.get("volumeNum") or 0.0),
        spread=_number(snapshot.get("spread")),
        bid=_number(snapshot.get("bestBid")),
        ask=_number(snapshot.get("bestAsk")),
        rule=rule,
    )


def load_ladders(snapshots: Iterable[dict[str, Any]]) -> dict[str, list[LadderRow]]:
    """快照行 → 按 `族别/窗口起点` 分组的阶梯；组内按行权价升序、同价只留成交额最大那档。"""
    grouped: dict[str, list[LadderRow]] = {}
    for snapshot in snapshots:
        row = ladder_row(snapshot)
        if row is not None:
            grouped.setdefault(row.ladder_key, []).append(row)
    for key, rows in grouped.items():
        best: dict[float, LadderRow] = {}
        for row in rows:
            current = best.get(row.strike)
            if current is None or row.volume > current.volume:
                best[row.strike] = row
        grouped[key] = sorted(best.values(), key=lambda item: item.strike)
    return grouped


def rule_warnings(rows: Sequence[LadderRow]) -> list[str]:
    """口径自检：文本里的价格类型、粒度、结算方式必须与档位族一致，否则不能定价。"""
    warnings: list[str] = []
    for row in rows:
        if row.rule is None:
            warnings.append(f"{row.slug}: 结算文本认不出判定源/价格类型，跳过定价")
            continue
        expected = "High" if row.direction == "up" else "Low"
        if row.rule.price_type not in (expected, PRICE_TYPE_BOTH):
            warnings.append(f"{row.slug}: {row.direction} 档但文本写 {row.rule.price_type}")
        if row.rule.granularity_minutes != EXPECTED_GRANULARITY_MINUTES:
            warnings.append(
                f"{row.slug}: 监控粒度 {row.rule.granularity_minutes} 分钟，"
                f"非 {EXPECTED_GRANULARITY_MINUTES} 分钟"
            )
        if not row.rule.immediate:
            warnings.append(f"{row.slug}: 文本未见 immediately resolve，可能非触碰即结算")
    return warnings
