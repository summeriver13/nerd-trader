"""maker 挂单执行模型。

实测依据（1m OHLC 上 12.9 万根 BTCUSDT，见 data/results 的 maker 研究）：

- 挂在市价（offset=0）：触价率 99.6%，但这是**吃价成交**——成交价就是市价，
  费率必须按 taker 计，没有任何节省；此前按 maker 计费是白捡 7.9bp/边。
- 挂在市价下方 1bp：触价率降到 86.6%，被动成交按 maker 计费、买价还低 1bp，
  含「挂不上回退市价」的成本后净优势约 6.1bp/边。
- 急涨（前 15 根 > +50bp）：挂低价会被吃穿，改挂当前价 = 追单，按 taker 计。

故默认 ``offset=1bp``：只有真正被动成交的那部分才拿 maker 费率。

本模块把「目标仓位序列」转成「实际成交仓位序列 + 逐根成本」，供向量化
回测使用；实盘下单可复用同一套挂单规则（挂单价、耐心、未成交回退）。

时序约定：``target_pos[t]`` 是 bar t 的目标仓位，挂单以 ``close[t]`` 为
参考价建立。挂在市价（含急涨追单）的订单当根即成，按 taker 计费；挂在市价
下方的订单从 bar t+1 起用 low/high 判断是否被触及——同一根 K 线的 ``low``
发生在 ``close`` 之前，用它验证基于该根 ``close`` 的挂单会引入前视偏差。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.costs import PERP_FEES, FeeSchedule

FILL_COLUMNS = ("bar", "side", "fill", "price", "ref", "delta", "cost")


@dataclass(frozen=True)
class MakerPolicy:
    """maker 挂单执行参数（费率口径见 src/core/costs.py，默认永续）。"""

    enabled: bool = True
    fees: FeeSchedule = PERP_FEES
    offset: float = 0.0001  # 挂单价差：0 = 挂在市价（吃价成交），>0 = 等价格回落
    patience: int = 1  # 挂单耐心（根），超时回退市价
    spike_lookback: int = 15  # 急涨判定的回看根数
    spike_threshold: float = 0.005  # 急涨阈值（前 N 根收益 > 0.5%）

    @property
    def maker_net_cost(self) -> float:
        """maker 成交的单边净成本：手续费减去挂单价差带来的价格改善。"""
        return self.fees.maker_net(self.offset)

    @property
    def taker_net_cost(self) -> float:
        """市价成交的单边净成本：手续费加滑点。"""
        return self.fees.taker_net()


@dataclass(frozen=True)
class PendingOrder:
    """在途挂单。"""

    is_buy: bool
    limit: float
    ref: float
    bar: int
    deadline: int
    delta: float

    @property
    def at_market(self) -> bool:
        """挂在市价（无价差）：当根即成，但属于吃价成交，按 taker 计费。"""
        return self.limit == self.ref


@dataclass(frozen=True)
class _Fill:
    """一笔挂单的结局。"""

    kind: str  # maker 成交 / taker 回退
    price: float
    rate: float


@dataclass
class ExecutionResult:
    """maker 执行结果。"""

    pos: pd.Series  # 实际生效仓位
    cost: pd.Series  # 逐根成本（收益率口径，未乘杠杆）
    fills: pd.DataFrame  # 每笔交易明细


def _is_spike(close: np.ndarray, bar: int, policy: MakerPolicy) -> bool:
    """判断 bar 前 spike_lookback 根是否处于急涨。"""
    start = bar - policy.spike_lookback
    if start < 0:
        return False
    ref = close[start]
    return bool(ref > 0 and close[bar] / ref - 1 > policy.spike_threshold)


def _make_order(
    close: np.ndarray, bar: int, delta: float, policy: MakerPolicy
) -> PendingOrder:
    """生成挂单：常态挂 -offset，急涨的买单改挂当前价。"""
    is_buy = delta > 0
    off = 0.0 if (is_buy and _is_spike(close, bar, policy)) else policy.offset
    ref = close[bar]
    limit = ref * (1 - off) if is_buy else ref * (1 + off)
    return PendingOrder(
        is_buy=is_buy,
        limit=limit,
        ref=ref,
        bar=bar,
        deadline=bar + policy.patience,
        delta=delta,
    )


def _resolve_pending(
    order: PendingOrder,
    bar: int,
    low: float,
    high: float,
    close: float,
    policy: MakerPolicy,
) -> _Fill | None:
    """在途挂单在 bar 内的结局：成交、超时回退市价，或继续等待。"""
    if bar == order.bar:
        # 挂在市价 / 急涨追单：当根即成，但吃的是对手价，按 taker 计费
        if not order.at_market:
            return None
        return _Fill("taker", order.limit, policy.taker_net_cost)
    touched = low <= order.limit if order.is_buy else high >= order.limit
    if touched:
        # 被本根 low/high 触及：被动成交，按 maker 计费
        return _Fill("maker", order.limit, policy.maker_net_cost)
    if bar >= order.deadline:
        return _Fill("taker", close, policy.taker_net_cost)
    return None


def execute_maker(
    df: pd.DataFrame, target_pos: pd.Series, policy: MakerPolicy
) -> ExecutionResult:
    """按 maker 挂单规则把目标仓位转成实际成交仓位与成本。

    ``target_pos[t]`` 是 bar t 的目标仓位；挂单以 ``close[t]`` 为参考价建立，
    挂在市价（含急涨追单）的当根即成且按 taker 计费，挂在市价外的从 bar t+1
    起等 ``patience`` 根、被 low/high 触及才被动成交，超时回退市价。
    目标在等待期间变化时撤单重挂。
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    target = target_pos.to_numpy(dtype=float)
    n = len(df)

    pos = np.zeros(n, dtype=float)
    cost = np.zeros(n, dtype=float)
    records: list[dict[str, object]] = []
    cur = 0.0
    pending: PendingOrder | None = None

    for t in range(n):
        if pending is not None and cur + pending.delta != target[t]:
            pending = None  # 目标已变，撤单重挂
        if pending is None and target[t] != cur:
            pending = _make_order(close, t, target[t] - cur, policy)

        if pending is not None:
            fill = _resolve_pending(pending, t, low[t], high[t], close[t], policy)
            if fill is not None:
                cur += pending.delta
                cost[t] += abs(pending.delta) * fill.rate
                records.append(_record(t, pending, fill))
                pending = None

        pos[t] = cur

    fills = pd.DataFrame(records, columns=pd.Index(FILL_COLUMNS))
    return ExecutionResult(
        pos=pd.Series(pos, index=df.index),
        cost=pd.Series(cost, index=df.index),
        fills=fills,
    )


def _record(bar: int, order: PendingOrder, fill: _Fill) -> dict[str, object]:
    """生成一条成交明细。"""
    return {
        "bar": bar,
        "side": "buy" if order.is_buy else "sell",
        "fill": fill.kind,
        "price": fill.price,
        "ref": order.ref,
        "delta": order.delta,
        "cost": abs(order.delta) * fill.rate,
    }
