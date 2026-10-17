"""
src/utils/data_sources/okx_liquidation.py
OKX 强平流公开端点的 transport 层：取数、解析、合约面值换算。

契约是 2026-10-17 实测核对（无需鉴权；本机直连不通，统一走 `create_session()` 的代理）：

- `GET /api/v5/public/liquidation-orders` 必带 `instType=SWAP`，`uly` 与 `instFamily`
  二选一即可（只给 instType 报 50015）；`state` 只接受 filled/unfilled（partial-filled
  报 51000）；`limit` 上限 100（传 500 报 51000）。
- 每个族只滚动保留**最近 100 条**：BTC 约 11 条/小时，即 100 条约覆盖 9 小时；
  `after=now-24h` 与 `after=now-48h` 都返回 0 条——**没有历史回补入口**，只能在存续
  期内高频轮询，这就是采集器必须常驻、且要内置缺口检测的原因。
- 分页参数语义反直觉：`after=<ts>` 返回的是**早于该时刻**的记录，`before=now` 返回 0 条。
  故这里只用默认页（最新 100 条），跨轮次去重靠业务键。
- 响应按 `instId` 分组，组内 `details` 没有订单 id，去重键只能用
  `(instId, ts, side, posSide, bkPx, sz)`。
- `sz` 是**合约张数**：换算基础币要乘 `/public/instruments` 的 `ctVal`（线性合约
  `ctType=linear`、`ctValCcy` 为基础币）。名义 = sz × ctVal × bkPx，这个换算写错
  会把「0.09 张 BTC」读成 0.09 BTC（相差 100 倍），故解析时要求 ctVal 必须存在。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.utils.data_sources.base import get_json

BASE_URL = "https://www.okx.com/api/v5"
LIQUIDATION_PATH = "/public/liquidation-orders"
INSTRUMENTS_PATH = "/public/instruments"
INST_TYPE_SWAP = "SWAP"
STATE_FILLED = "filled"
PAGE_LIMIT = 100
SIDES = frozenset({"buy", "sell"})
POS_SIDES = frozenset({"long", "short"})


@dataclass(frozen=True)
class LiquidationEvent:
    """一条强平成交：上游 `details` 里的一行，已换算成基础币与名义金额。"""

    inst_id: str
    inst_family: str
    side: str
    pos_side: str
    price: float
    size_contracts: float
    ct_val: float
    ts_ms: int

    @property
    def size_ccy(self) -> float:
        """被强平的基础币数量（张数 × 合约面值）。"""
        return self.size_contracts * self.ct_val

    @property
    def notional(self) -> float:
        """被强平的名义金额（计价币，USDT 永续即 USDT）。"""
        return self.size_ccy * self.price

    @property
    def key(self) -> tuple[str, int, str, str, float, float]:
        """业务键：上游无订单 id，六个字段一起才够区分同一毫秒的多笔强平。"""
        return (
            self.inst_id,
            self.ts_ms,
            self.side,
            self.pos_side,
            self.price,
            self.size_contracts,
        )


@dataclass(frozen=True)
class ContractSpec:
    """合约面值口径：线性永续的 `ctVal` 是「每张合约的基础币数量」。"""

    inst_id: str
    inst_family: str
    ct_val: float
    ct_val_ccy: str
    ct_type: str


def _as_float(value: Any, field: str) -> float:
    """上游数字全是字符串且空值为 ""；解析失败一律抛错，不静默补 0。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"字段 {field} 不是数字：{value!r}") from error
    if number <= 0:
        raise ValueError(f"字段 {field} 非正：{value!r}")
    return number


def parse_contracts(payload: dict[str, Any]) -> dict[str, ContractSpec]:
    """解析 `/public/instruments` 响应，只留线性 USDT 永续的面值表。"""
    if payload.get("code") != "0":
        raise RuntimeError(f"合约列表返回异常：{payload.get('msg')!r}")
    specs: dict[str, ContractSpec] = {}
    for row in payload.get("data", []):
        if row.get("ctType") != "linear":
            continue
        specs[row["instId"]] = ContractSpec(
            inst_id=row["instId"],
            inst_family=row.get("instFamily") or row["instId"],
            ct_val=_as_float(row.get("ctVal"), "ctVal"),
            ct_val_ccy=row.get("ctValCcy") or "",
            ct_type=row.get("ctType") or "",
        )
    return specs


def fetch_contracts(session: requests.Session) -> dict[str, ContractSpec]:
    """一次取回全部 SWAP 合约面值（单次请求，按 instId 建索引）。"""
    payload = get_json(
        session, BASE_URL + INSTRUMENTS_PATH, {"instType": INST_TYPE_SWAP}
    )
    return parse_contracts(payload)


def parse_events(
    payload: dict[str, Any], contracts: dict[str, ContractSpec]
) -> list[LiquidationEvent]:
    """把一轮响应摊平成事件列表；面值缺失的合约直接跳过（不猜换算关系）。"""
    if payload.get("code") != "0":
        raise RuntimeError(f"强平端点返回异常：{payload.get('msg')!r}")
    events: list[LiquidationEvent] = []
    for group in payload.get("data", []):
        inst_id = group["instId"]
        spec = contracts.get(inst_id)
        if spec is None:
            continue
        family = group.get("instFamily") or spec.inst_family
        for detail in group.get("details", []):
            side = detail.get("side")
            pos_side = detail.get("posSide")
            if side not in SIDES or pos_side not in POS_SIDES:
                raise ValueError(f"未知方向：side={side!r} posSide={pos_side!r}")
            events.append(
                LiquidationEvent(
                    inst_id=inst_id,
                    inst_family=family,
                    side=side,
                    pos_side=pos_side,
                    price=_as_float(detail.get("bkPx"), "bkPx"),
                    size_contracts=_as_float(detail.get("sz"), "sz"),
                    ct_val=spec.ct_val,
                    ts_ms=int(detail.get("ts") or detail.get("time")),
                )
            )
    return events


def fetch_events(
    session: requests.Session,
    inst_family: str,
    contracts: dict[str, ContractSpec],
    *,
    limit: int = PAGE_LIMIT,
) -> list[LiquidationEvent]:
    """取一个族的最新强平记录（默认页 = 最近 100 条，无历史回补）。"""
    payload = get_json(
        session,
        BASE_URL + LIQUIDATION_PATH,
        {
            "instType": INST_TYPE_SWAP,
            "instFamily": inst_family,
            "state": STATE_FILLED,
            "limit": limit,
        },
    )
    return parse_events(payload, contracts)
