"""maker 挂单成交率实测：把「被触及即成交」这条假设放到真实盘口上量一量。

模型侧（src/research/execution.py）假设挂单挂在市价外 1bp、**被触及就成交**，据此算出
maker 相对市价约有 +6.1bp/边 的优势。这条假设有两半，得分开量：

1. **触及率** P(窗口内市场价到过挂单价)：1m 数据上实测 86.6%，可离线复算；
2. **成交率** P(实际成交)：模型隐含 100%（触及即成交），**排队位置的成本完全没建**。

本脚本在模拟盘上量第 2 半：每个 trial 在市场外 1bp 挂一张小额定单，窗口内盯订单状态，
窗口结束后用 1m K 线的 high/low 判「市场到没到过我们的价位」，得到三个数：

    触及率       = 触及 / 总数    ← 与模型的 86.6% 对照
    成交率       = 成交 / 总数    ← 与模型隐含的 100% 对照
    触及后成交率 = 成交 / 触及    ← 模型没建的那个数，就是排队位置的代价

用法：
    uv run python scripts/measure_maker_fill.py --trials 20              # 只报计划
    uv run python scripts/measure_maker_fill.py --trials 24 --side both --place-orders

安全：``--place-orders`` 只在 ``OKX_SANDBOX=true`` 时允许，实盘一律拒绝。
20 个 trial × 60 秒窗口 ≈ 21 分钟，适合后台跑（输出重定向到 data/results/logs/）。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Final

from src.config.settings import GLOBAL_CONFIG, create_data_exchange, create_exchange
from src.utils.console import pad

# 探针参数
OFFSET_BP: Final[float] = 1.0  # 挂在市场外 1bp（模型默认）
WINDOW_SECONDS: Final[float] = 60.0  # 1 分钟窗口 = 模型的 bar 单位
POLL_SECONDS: Final[float] = 2.0
PROBE_NOTIONAL_USDT: Final[float] = 20.0
DEFAULT_TRIALS: Final[int] = 20
MODEL_TOUCH_RATE: Final[float] = 0.866  # 1m BTCUSDT 12.9 万根上的触价率
BP: Final[float] = 1e4
SYMBOL: Final[str] = "BTC/USDT"
TRADE_LIMIT: Final[int] = 100
MAX_PRINTED_TRIALS: Final[int] = 24
SIDES: Final[tuple[str, ...]] = ("sell", "buy")


@dataclass(frozen=True)
class ProbePlan:
    """一次实测的参数（参数多时用对象传，别堆位置参数）。"""

    side: str
    trials: int
    notional: float = PROBE_NOTIONAL_USDT
    window: float = WINDOW_SECONDS


@dataclass
class Trial:
    """单个 trial 的观测。"""

    index: int
    side: str
    limit: float
    amount: float
    spread_bp: float
    order_id: str = ""
    stamp: float = 0.0
    filled: bool = False
    filled_amount: float = 0.0
    average: float | None = None
    touched: bool = False
    seconds: float = 0.0


def _limit_price(side: str, bid: float, ask: float) -> float:
    """挂在市场外 1bp：卖单挂卖一之上、买单挂买一之下，保证是被动单。"""
    reference = ask if side == "sell" else bid
    sign = 1.0 if side == "sell" else -1.0
    return reference * (1.0 + sign * OFFSET_BP / BP)


def _reaches(side: str, price: float, limit: float) -> bool:
    """市场价是否到过我们的挂单价：卖单要有人出到我们的价，买单相反。"""
    return price >= limit if side == "sell" else price <= limit


def _trial_side(plan: ProbePlan, index: int) -> str:
    """买卖交替：只挂一边会碰上行情单边走的时段，样本就带方向偏差。"""
    if plan.side != "both":
        return plan.side
    return SIDES[index % len(SIDES)]


def _place(exchange: Any, plan: ProbePlan, index: int) -> Trial:
    """取盘口 → 挂一张市场外 1bp 的限价单。"""
    side = _trial_side(plan, index)
    ticker = exchange.fetch_ticker(SYMBOL)
    bid, ask = float(ticker["bid"]), float(ticker["ask"])
    limit = float(exchange.price_to_precision(SYMBOL, _limit_price(side, bid, ask)))
    amount = float(exchange.amount_to_precision(SYMBOL, plan.notional / limit))
    order = exchange.create_order(SYMBOL, "limit", side, amount, limit)
    return Trial(
        index=index,
        side=side,
        limit=limit,
        amount=amount,
        spread_bp=(ask - bid) / ((ask + bid) / 2.0) * BP,
        order_id=str(order["id"]),
        stamp=time.time(),
    )


def _bar_reaches(exchange: Any, trial: Trial, since: float) -> bool:
    """窗口内市场有没有到过挂单价——用 1m K 线的最高/最低价判。

    模型侧的触价率就是这么算的（1m bar 的 high/low 是否触及），同一口径才好对照。
    公开成交流水不能用来判触及：BTC 上 100 笔只覆盖一两秒，2 秒轮询必然漏。
    """
    since_ms = int(since) // 60 * 60 * 1000  # 对齐到分钟
    for bar in exchange.fetch_ohlcv(SYMBOL, "1m", since=since_ms, limit=3):
        high, low = float(bar[2]), float(bar[3])
        if _reaches(trial.side, high, trial.limit) or _reaches(
            trial.side, low, trial.limit
        ):
            return True
    return False


def _watch(trade: Any, market: Any, plan: ProbePlan, trial: Trial) -> None:
    """盯订单：成交就提前结束，否则等满窗口后撤单，最后用 1m K 线补判触及。

    成交状态以模拟盘客户端为准（下单就在那里），触及判定走实盘行情客户端：
    沙盒 feed 是整体微扰的另一套数据，拿它判触及等于用扰动后的价格当市场。
    """
    started = time.time()
    deadline = started + plan.window
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        order = trade.fetch_order(trial.order_id, SYMBOL)
        trial.filled_amount = float(order.get("filled") or 0.0)
        if trial.filled_amount > 0:
            trial.filled = True
            trial.average = order.get("average")
            break
    trial.seconds = time.time() - started
    if not trial.filled:
        trade.cancel_order(trial.order_id, SYMBOL)
    # 成交必然意味着触及：K 线口径会漏掉刚好在窗口边界的那一根，所以取并集
    trial.touched = trial.filled or _bar_reaches(market, trial, started)


def _cancel_leftovers(exchange: Any) -> int:
    """清理上一轮因网络中断留下的挂单，免得干扰本次测量。"""
    leftovers = exchange.fetch_open_orders(SYMBOL)
    for order in leftovers:
        exchange.cancel_order(str(order["id"]), SYMBOL)
    return len(leftovers)


def _safe_cancel(exchange: Any, trial: Trial) -> None:
    """trial 中断时尽力撤单，撤不掉就说明白（不能静默吞掉）。"""
    try:
        exchange.cancel_order(trial.order_id, SYMBOL)
    except Exception as e:
        print(
            f"   撤单未成功（{type(e).__name__}），下一轮启动时会清理残留挂单",
            flush=True,
        )


def run_trials(trade: Any, market: Any, plan: ProbePlan) -> list[Trial]:
    """顺序跑完所有 trial（并发会互相抢同一档价位，成交率就不独立了）。

    单个 trial 撞上网络抖动只跳它自己——网络在这台机器上长期不稳，一次抖动不该让整轮
    白跑；中断的 trial 不入汇总（宁可样本少，也不要半个观测）。
    """
    results: list[Trial] = []
    for index in range(1, plan.trials + 1):
        trial = _place(trade, plan, index)
        print(
            f"  trial {index:2d}/{plan.trials} 挂 {trial.side} @ {trial.limit:.2f}"
            f"（价差 {trial.spread_bp:.2f}bp）...",
            end="",
            flush=True,
        )
        try:
            _watch(trade, market, plan, trial)
        except Exception as e:
            print(f" 中断（{type(e).__name__}: {str(e)[:80]}）", flush=True)
            _safe_cancel(trade, trial)
            continue
        verdict = (
            "成交" if trial.filled else ("触及未成交" if trial.touched else "未触及")
        )
        print(f" {verdict}（{trial.seconds:.0f}s）", flush=True)
        results.append(trial)
    return results


def _maker_counts(exchange: Any, since: float) -> tuple[int, int]:
    """本次运行期间成交的 maker / taker 笔数——被动挂单理应全是 maker。"""
    since_ms = int(since * 1000)
    maker = taker = 0
    for trade in exchange.fetch_my_trades(SYMBOL, limit=TRADE_LIMIT):
        stamp = trade.get("timestamp")
        if stamp is None or int(stamp) < since_ms:
            continue
        if trade.get("takerOrMaker") == "maker":
            maker += 1
        elif trade.get("takerOrMaker") == "taker":
            taker += 1
    return maker, taker


def _share(hit: int, total: int) -> str:
    """比例永远跟分母一起打印，别给一个孤立百分数。"""
    return "n/a" if total == 0 else f"{hit}/{total} = {hit / total:.1%}"


def _side_summary(side: str, group: list[Trial]) -> None:
    """一个方向的三个数：成交率、触及率、触及后成交率。"""
    filled = sum(1 for t in group if t.filled)
    touched = sum(1 for t in group if t.touched)
    times = [t.seconds for t in group if t.filled]
    print(f"  {pad(side, 6)}{_share(filled, len(group))} 成交")
    print(
        f"  {pad('', 6)}{_share(touched, len(group))} 触及（模型 1m 实测 {MODEL_TOUCH_RATE:.1%}）"
    )
    print(
        f"  {pad('', 6)}{_share(filled, touched)} 触及后成交 ← 排队位置的代价，模型没建"
    )
    if times:
        print(
            f"  {pad('', 6)}成交耗时中位 {statistics.median(times):.0f}s"
            f"（最短 {min(times):.0f}s / 最长 {max(times):.0f}s）"
        )


def report(exchange: Any, plan: ProbePlan, trials: list[Trial]) -> None:
    """打印逐 trial 明细与分方向的汇总。"""
    if not trials:
        print("\n没有任何 trial 完成，无法汇总")
        return
    print(f"\n逐 trial 明细（挂在市场外 {OFFSET_BP:.0f}bp，窗口 {plan.window:.0f}s）：")
    print(
        f"  {pad('trial', 7)}{pad('方向', 6)}{pad('挂单价', 12, 'right')}"
        f"{pad('价差', 10, 'right')}{pad('触及', 7)}{pad('成交', 7)}{pad('耗时', 8, 'right')}"
    )
    for trial in trials[:MAX_PRINTED_TRIALS]:
        print(
            f"  {pad(str(trial.index), 7)}{pad(trial.side, 6)}"
            f"{pad(f'{trial.limit:.2f}', 12, 'right')}"
            f"{pad(f'{trial.spread_bp:.2f}bp', 10, 'right')}"
            f"{pad('是' if trial.touched else '否', 7)}"
            f"{pad('是' if trial.filled else '否', 7)}"
            f"{pad(f'{trial.seconds:.0f}s', 8, 'right')}"
        )
    if len(trials) > MAX_PRINTED_TRIALS:
        print(f"  …（共 {len(trials)} 个，只打印前 {MAX_PRINTED_TRIALS} 个）")

    total = len(trials)
    filled = sum(1 for t in trials if t.filled)
    print(f"\n汇总（n={total}）：")
    print(
        f"  成交率        {_share(filled, total)}   ← 模型隐含 100%（触及即成交），"
        f"低 {1.0 - filled / total:.1%}"
    )
    for side in SIDES:
        group = [t for t in trials if t.side == side]
        if group:
            _side_summary(side, group)
    maker, taker = _maker_counts(exchange, min(t.stamp for t in trials))
    print(f"  本次成交类型  maker {maker} / taker {taker}（被动挂单理应全是 maker）")
    print(
        "  口径差异：模型用 bar 收盘价作参照；本脚本用实时卖一/买一价、下单起 60 秒窗口，"
        "触及率只能对照不能等同。"
    )


def _run(args: argparse.Namespace) -> int:
    if args.place_orders and not GLOBAL_CONFIG.sandbox_mode:
        print("❌ 拒绝下单：--place-orders 只允许在模拟盘（.env 里 OKX_SANDBOX=true）")
        return 2
    plan = ProbePlan(side=args.side, trials=args.trials, notional=args.notional)
    exchange: Any = create_exchange("okx")
    market: Any = create_data_exchange("okx")  # 触及判定用实盘行情，不用沙盒 feed
    try:
        exchange.load_markets()
        ticker = exchange.fetch_ticker(SYMBOL)
        bid, ask = float(ticker["bid"]), float(ticker["ask"])
        spread_bp = (ask - bid) / ((ask + bid) / 2.0) * BP
        print(
            f"maker 成交率实测 {SYMBOL} | 模拟盘 = {GLOBAL_CONFIG.sandbox_mode} "
            f"| 方向 {plan.side} | 挂市场外 {OFFSET_BP:.0f}bp | 窗口 {plan.window:.0f}s"
        )
        print(f"当前盘口 买一 {bid:.2f} / 卖一 {ask:.2f}（价差 {spread_bp:.2f}bp）")
        print(
            f"计划：{plan.trials} 个 trial，名义额 {plan.notional:.0f} USDT，"
            f"预计 {plan.trials * plan.window / 60:.0f} 分钟"
        )
        if not args.place_orders:
            print("（未加 --place-orders：只报计划，不下单）")
            return 0
        leftovers = _cancel_leftovers(exchange)
        if leftovers:
            print(f"启动前清理了 {leftovers} 张上一轮残留的挂单")
        report(exchange, plan, run_trials(exchange, market, plan))
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
        market.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="maker 挂单成交率实测（触及 vs 成交）")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trial 个数")
    parser.add_argument("--notional", type=float, default=PROBE_NOTIONAL_USDT)
    parser.add_argument(
        "--side",
        choices=("sell", "buy", "both"),
        default="sell",
        help="探针方向：both = 买卖交替（避免撞上单边行情带方向偏差）",
    )
    parser.add_argument(
        "--place-orders", action="store_true", help="在模拟盘真下单（实盘拒绝）"
    )
    return _run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
