"""执行模型对账：拿交易所真实成交回单，验证 src/core/costs.py 的费率与滑点假设。

模型侧有三条假设，全部要对上真实回单才算通过：

1. 市价单按 taker 计费（现货 0.100% / 永续 0.050%），另吃 5bp 滑点；
2. 限价单挂在市场同侧（买单价 ≥ 卖一价）即吃价成交，按 taker 计费；
3. 限价单挂在市价之外、被触及才按 maker 计费（现货 0.080% / 永续 0.020%）。

用法：
    uv run python scripts/reconcile_execution.py --market spot
    uv run python scripts/reconcile_execution.py --market spot --place-orders

安全：``--place-orders`` 只在 ``OKX_SANDBOX=true`` 时允许——实盘一律拒绝，
免得拿真金白银去验证一个可以离线推断的假设。判据是评审给的「单边费率误差 < 1bp」。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, Final

from src.config.settings import GLOBAL_CONFIG, create_exchange
from src.core.costs import PERP_FEES, SPOT_FEES, FeeSchedule
from src.utils.console import pad

# 判据与探针参数
FEE_TOLERANCE_BP: Final[float] = 1.0
SLIPPAGE_TOLERANCE_BP: Final[float] = 10.0
MAKER_OFFSET_BP: Final[float] = 1.0
PROBE_NOTIONAL_USDT: Final[float] = 20.0
FILL_WAIT_SECONDS: Final[float] = 20.0
POLL_SECONDS: Final[float] = 2.0
MAX_PRINTED_ROWS: Final[int] = 20
BP: Final[float] = 1e4

MARKETS: Final[dict[str, tuple[str, FeeSchedule]]] = {
    "spot": ("BTC/USDT", SPOT_FEES),
    "swap": ("BTC/USDT:USDT", PERP_FEES),
}


def _trade_fee_bp(trade: dict[str, Any], base: str, quote: str) -> float | None:
    """回单里的实际单边费率（bp）。手续费可能以基础币或计价币计，两种都要能算。"""
    fee = trade.get("fee") or {}
    cost = fee.get("cost")
    amount = trade.get("amount")
    price = trade.get("price")
    if not cost or not amount or not price:
        return None
    if fee.get("currency") == base:
        rate = float(cost) / float(amount)
    elif fee.get("currency") == quote:
        rate = float(cost) / (float(price) * float(amount))
    else:
        return None
    return rate * BP


def _expected_fee_bp(trade: dict[str, Any], schedule: FeeSchedule) -> float:
    """模型该收多少：只比手续费本身，滑点单独在探针单里量。"""
    kind = trade.get("takerOrMaker")
    return (schedule.taker if kind == "taker" else schedule.maker) * BP


def reconcile_trades(
    trades: list[dict[str, Any]], schedule: FeeSchedule, base: str, quote: str
) -> bool:
    """逐笔比对实际费率与模型费率，返回是否全部落在容差内。"""
    rows: list[tuple[str, str, str, float, float, float]] = []
    for trade in trades:
        actual = _trade_fee_bp(trade, base, quote)
        if actual is None:
            continue
        kind = str(trade.get("takerOrMaker") or "?")
        expected = _expected_fee_bp(trade, schedule)
        stamp = str(trade.get("datetime") or "")[:19]
        rows.append(
            (stamp, str(trade.get("side")), kind, actual, expected, actual - expected)
        )

    if not rows:
        print("回单里没有可解析费率的成交（模拟盘新账户可能一笔都没成交过）")
        return False

    print(f"\n逐笔对账 {len(rows)} 笔（费率单位 bp = 万分之一）：")
    header = (
        f"  {pad('时间', 21)}{pad('方向', 6)}{pad('类型', 7)}"
        f"{pad('实际', 10, 'right')}{pad('模型', 10, 'right')}{pad('差', 10, 'right')}"
    )
    print(header)
    for stamp, side, kind, actual, expected, diff in rows[:MAX_PRINTED_ROWS]:
        print(
            f"  {pad(stamp, 21)}{pad(side, 6)}{pad(kind, 7)}"
            f"{pad(f'{actual:.3f}', 10, 'right')}{pad(f'{expected:.3f}', 10, 'right')}"
            f"{pad(f'{diff:+.3f}', 10, 'right')}"
        )
    if len(rows) > MAX_PRINTED_ROWS:
        print(f"  …（共 {len(rows)} 笔，只打印前 {MAX_PRINTED_ROWS} 笔）")

    diffs = [abs(row[5]) for row in rows]
    median = statistics.median(diffs)
    worst = max(diffs)
    ok = worst <= FEE_TOLERANCE_BP
    print(
        f"  费率偏差：中位 {median:.3f}bp、最大 {worst:.3f}bp"
        f"（容差 {FEE_TOLERANCE_BP:.1f}bp）→ {'通过' if ok else '未通过'}"
    )
    return ok


def _probe_plans(
    side: str, bid: float, ask: float
) -> list[tuple[str, str, float | None, float]]:
    """三笔探针单：市价、挂在市场同侧（吃价）、挂在市场外 1bp（被动）。"""
    word = "买" if side == "buy" else "卖"
    if side == "buy":
        market_ref, cross_ref = ask, ask
        passive = bid * (1 - MAKER_OFFSET_BP / BP)
        cross_name, passive_name = "卖一", "买一"
    else:
        market_ref, cross_ref = bid, bid
        passive = ask * (1 + MAKER_OFFSET_BP / BP)
        cross_name, passive_name = "买一", "卖一"
    return [
        (f"市价{word}", "market", None, market_ref),
        (f"限价{word}@{cross_name}（吃价）", "limit", cross_ref, cross_ref),
        (f"限价{word}@{passive_name}外1bp（被动）", "limit", passive, market_ref),
    ]


def place_probe_orders(
    exchange: Any, symbol: str, notional: float, side: str
) -> list[dict[str, Any]]:
    """下三笔探针单并返回每笔的下单前盘口，供后续比对滑点与费率。"""
    ticker = exchange.fetch_ticker(symbol)
    ask = float(ticker["ask"])
    bid = float(ticker["bid"])
    amount = float(exchange.amount_to_precision(symbol, notional / ask))
    probes: list[dict[str, Any]] = []
    for label, kind, price, reference in _probe_plans(side, bid, ask):
        order = exchange.create_order(
            symbol,
            kind,
            side,
            amount,
            None
            if price is None
            else float(exchange.price_to_precision(symbol, price)),
        )
        probes.append(
            {
                "label": label,
                "kind": kind,
                "side": side,
                "price": price,
                "reference": reference,
                "id": order["id"],
            }
        )
        time.sleep(POLL_SECONDS)
    return probes


def _slippage_bp(side: str, average: float, reference: float) -> float:
    """成交价相对下单前盘口的偏离（正数 = 对我不利，买吃亏是成交价更高、卖吃亏是更低）。"""
    sign = 1.0 if side == "buy" else -1.0
    return sign * (average - reference) / reference * BP


def _settle_probes(exchange: Any, symbol: str, probes: list[dict[str, Any]]) -> None:
    """等到全部终态，把仍未成交的撤掉（模拟盘挂单不该留在那里）。"""
    deadline = time.time() + FILL_WAIT_SECONDS
    while time.time() < deadline:
        if all(
            exchange.fetch_order(p["id"], symbol)["status"] == "closed" for p in probes
        ):
            break
        time.sleep(POLL_SECONDS)
    for probe in probes:
        order = exchange.fetch_order(probe["id"], symbol)
        probe["status"] = order["status"]
        probe["filled"] = float(order.get("filled") or 0.0)
        probe["average"] = order.get("average")
        if order["status"] == "open":
            exchange.cancel_order(probe["id"], symbol)
            probe["status"] = "canceled"


def report_probes(
    probes: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    schedule: FeeSchedule,
    base: str,
    quote: str,
) -> bool:
    """打印探针单结果：滑点、OKX 对「吃价/被动」的实际判定。"""
    by_order = {t.get("order"): t for t in trades}
    ok = True
    print(f"\n探针单结果（下单名义额 {PROBE_NOTIONAL_USDT:.0f} {quote}）：")
    for probe in probes:
        trade = by_order.get(probe["id"]) or {}
        if probe["status"] != "closed" or not probe["average"]:
            print(
                f"  {pad(probe['label'], 28)}{probe['status']}："
                f"未成交（等待 {FILL_WAIT_SECONDS:.0f}s 后撤单）"
            )
            if probe["kind"] == "limit" and "被动" in probe["label"]:
                print("     → 「挂单被触及即成交」对 1bp 外的挂单偏乐观，本样本不支持")
            continue
        kind = str(trade.get("takerOrMaker") or "?")
        actual = _trade_fee_bp(trade, base, quote)
        expected = _expected_fee_bp(trade, schedule)
        line = f"  {pad(probe['label'], 28)}成交 @ {probe['average']:.2f} | OKX 标记 {kind:<5}"
        if actual is not None:
            diff = actual - expected
            line += f" | 实际 {actual:.3f}bp vs 模型 {expected:.3f}bp（差 {diff:+.3f}）"
            ok = ok and abs(diff) <= FEE_TOLERANCE_BP
        if probe["kind"] == "market":
            slip = _slippage_bp(
                str(probe["side"]), float(probe["average"]), float(probe["reference"])
            )
            line += f" | 滑点 {slip:+.2f}bp（模型 {schedule.slippage * BP:.1f}bp）"
            ok = ok and abs(slip) <= SLIPPAGE_TOLERANCE_BP
        print(line)
    return ok


def _run(args: argparse.Namespace) -> int:
    symbol, schedule = MARKETS[args.market]
    if args.place_orders and not GLOBAL_CONFIG.sandbox_mode:
        print("❌ 拒绝下单：--place-orders 只允许在模拟盘（.env 里 OKX_SANDBOX=true）")
        return 2
    exchange: Any = create_exchange("okx")
    try:
        exchange.load_markets()
        market = exchange.market(symbol)
        base, quote = str(market["base"]), str(market["quote"])
        balance = exchange.fetch_balance()
        free = (balance.get(quote) or {}).get("free")
        print(
            f"对账 {symbol}（{args.market}）| 模拟盘 = {GLOBAL_CONFIG.sandbox_mode} "
            f"| 模型 maker {schedule.maker * BP:.3f}bp / taker {schedule.taker * BP:.3f}bp"
        )
        print(f"账户可用 {quote} = {free} | 探针方向 = {args.side}")
        ok = True
        if args.place_orders:
            probes = place_probe_orders(exchange, symbol, args.notional, args.side)
            _settle_probes(exchange, symbol, probes)
            trades = exchange.fetch_my_trades(symbol, limit=50)
            ok = report_probes(probes, trades, schedule, base, quote)
        trades = exchange.fetch_my_trades(symbol, limit=args.limit)
        ok = reconcile_trades(trades, schedule, base, quote) and ok
    except Exception as e:
        # 「未完成」（网络/权限）与「未通过」（数字不对）必须分开报
        print(f"\n未完成：{type(e).__name__}: {str(e)[:200]}")
        if "api key" in str(e).lower() or "50111" in str(e) or "50110" in str(e):
            print(
                f"   鉴权失败而当前模拟盘 = {GLOBAL_CONFIG.sandbox_mode}："
                "模拟盘要用「交易 → 模拟交易 → API 管理」里单独创建的 key，与实盘 key 不通用"
            )
        return 2
    finally:
        exchange.close()
    print(f"\n结论：费率与滑点对账 {'通过' if ok else '未通过'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="执行模型对账（费率与滑点 vs 真实回单）"
    )
    parser.add_argument("--market", choices=sorted(MARKETS), default="spot")
    parser.add_argument("--limit", type=int, default=50, help="拉取回单条数")
    parser.add_argument("--notional", type=float, default=PROBE_NOTIONAL_USDT)
    parser.add_argument(
        "--side",
        choices=("buy", "sell"),
        default="buy",
        help="探针方向：模拟盘若没有计价币余额，用 sell 可直接跑",
    )
    parser.add_argument(
        "--place-orders", action="store_true", help="在模拟盘下探针单（实盘拒绝）"
    )
    return _run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
