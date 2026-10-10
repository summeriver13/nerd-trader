"""经典趋势追踪与自适应双均线的正面比较（脚本只读本地缓存，不联网）。

问题：292 行的自适应双均线（牛市/熊市分参数 + ADX 过滤 + ATR 止损）
是否真的比经典唐奇安突破更好？两者在同一数据、同一引擎、同一成本下对比。

用法：
    uv run python scripts/trend_duel.py --symbol BTC/USDT --timeframe 1d
    uv run python scripts/trend_duel.py --symbol ETH/USDT --timeframe 1d

输出两张表：
1. 候选策略对比（含买入持有基准）
2. 唐奇安参数敏感性（入场窗口从 10 到 200，出场窗口取入场的 1/3）
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.paths import crypto_bar_path
from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine
from src.research.strategies.base import BaseStrategy
from src.research.strategies.buy_and_hold import BuyAndHoldStrategy
from src.research.strategies.donchian_trend import DonchianTrendStrategy
from src.research.strategies.vector_dual_ma import DualMAStrategy

EXCHANGE = "okx"
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 1.0
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1d"
# 与 registry 里的出厂参数一致，避免对比时偷换基线
DUAL_MA_PARAMS: dict[str, int] = {
    "trend_period": 120,
    "bull_fast": 10,
    "bull_slow": 30,
}
SCAN_ENTRIES = (10, 20, 30, 40, 55, 75, 100, 150, 200)
EXIT_RATIO = 3
MIN_EXIT_PERIOD = 5
SEGMENTS = 4
TABLE_WIDTH = 104


@dataclass(frozen=True)
class Row:
    """一个策略的绩效读数。"""

    name: str
    total_ret: float
    annual_ret: float
    sharpe: float
    calmar: float
    max_dd: float
    trades: int
    win_rate: float
    exposure: float
    cost: float


def run_one(bars: pd.DataFrame, name: str, strategy: BaseStrategy) -> Row | None:
    """算指标、生成信号、跑引擎，取绩效读数。"""
    data = strategy.generate_signals(strategy.calculate_indicators(bars.copy()))
    engine = BacktestEngine(
        domain=CryptoDomain(), initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE
    )
    result = engine.run(data)
    if not result:
        return None
    return Row(
        name=name,
        total_ret=float(result["total_ret"]),
        annual_ret=float(result["annual_ret"]),
        sharpe=float(result["sharpe_ratio"]),
        calmar=float(result["calmar_ratio"]),
        max_dd=float(result["max_drawdown"]),
        trades=int(result["trade_count"]),
        win_rate=float(result["win_rate"]),
        exposure=float(result["exposure"]),
        cost=float(result["total_cost"]),
    )


def print_rows(title: str, rows: list[Row]) -> None:
    """打印对比表。"""
    print(f"\n{title}")
    header = (
        f"{'strategy':<24}{'total':>10}{'annual':>9}{'sharpe':>8}{'calmar':>8}"
        f"{'maxDD':>9}{'trades':>8}{'win':>7}{'expo':>7}{'cost':>9}"
    )
    print(header)
    print("-" * TABLE_WIDTH)
    for row in rows:
        print(
            f"{row.name:<24}{row.total_ret:>9.1%}{row.annual_ret:>9.1%}"
            f"{row.sharpe:>8.2f}{row.calmar:>8.2f}{row.max_dd:>9.1%}"
            f"{row.trades:>8}{row.win_rate:>7.1%}{row.exposure:>7.1%}"
            f"{row.cost:>9.0f}"
        )


def build_candidates() -> list[tuple[str, BaseStrategy]]:
    """候选策略：基准、经典唐奇安各变体、自适应双均线。"""
    return [
        ("Buy & Hold", BuyAndHoldStrategy()),
        ("Donchian 55/20 stop", DonchianTrendStrategy(55, 20, atr_mult=2.0)),
        ("Donchian 55/20 no-stop", DonchianTrendStrategy(55, 20, atr_mult=0.0)),
        ("Donchian 20/10 stop", DonchianTrendStrategy(20, 10, atr_mult=2.0)),
        ("Donchian 100/50 stop", DonchianTrendStrategy(100, 50, atr_mult=2.0)),
        ("DualMA adaptive", DualMAStrategy(**DUAL_MA_PARAMS)),
    ]


def build_scan() -> list[tuple[str, BaseStrategy]]:
    """唐奇安参数敏感性：出场窗口取入场的三分之一。"""
    return [
        (
            f"Donchian {n}/{max(MIN_EXIT_PERIOD, n // EXIT_RATIO)}",
            DonchianTrendStrategy(n, max(MIN_EXIT_PERIOD, n // EXIT_RATIO)),
        )
        for n in SCAN_ENTRIES
    ]


def print_segments(
    bars: pd.DataFrame, candidates: list[tuple[str, BaseStrategy]], segments: int
) -> None:
    """分段稳定性：等分全历史，看每个策略是否每段都能站住脚。"""
    edges = [int(round(x)) for x in np.linspace(0, len(bars), segments + 1)]
    labels = [
        f"{bars.index[edges[i]]:%Y-%m}~{bars.index[edges[i + 1] - 1]:%Y-%m}"
        for i in range(segments)
    ]
    print("\n=== 分段稳定性（各段总收益）===")
    header = f"{'strategy':<24}" + "".join(f"{label:>16}" for label in labels)
    print(header)
    print("-" * len(header))
    for name, strategy in candidates:
        # 指标与信号在全量上算好再切段，避免每段重新预热
        data = strategy.generate_signals(strategy.calculate_indicators(bars.copy()))
        cells = ""
        for i in range(segments):
            chunk = data.iloc[edges[i] : edges[i + 1]]
            engine = BacktestEngine(
                domain=CryptoDomain(),
                initial_capital=INITIAL_CAPITAL,
                leverage=LEVERAGE,
            )
            result = engine.run(chunk)
            cells += f"{result['total_ret']:>16.1%}" if result else f"{'n/a':>16}"
        print(f"{name:<24}{cells}")


def run_duel(symbol: str, timeframe: str) -> None:
    """跑一个标的的完整对比。"""
    bars = pd.read_parquet(crypto_bar_path(EXCHANGE, symbol, timeframe))
    print(
        f"标的 {symbol} | 周期 {timeframe} | {len(bars)} 根 | "
        f"{bars.index[0]} -> {bars.index[-1]}"
    )
    print(
        f"口径：初始资金 {INITIAL_CAPITAL:.0f} USDT、杠杆 {LEVERAGE}x、市价单、含手续费与滑点"
    )

    for title, candidates in (
        ("=== 经典 vs 自适应 ===", build_candidates()),
        ("=== 唐奇安参数敏感性（出场 = 入场/3）===", build_scan()),
    ):
        rows: list[Row] = []
        for name, strategy in candidates:
            row = run_one(bars, name, strategy)
            if row is not None:
                rows.append(row)
        print_rows(title, rows)

    print_segments(bars, build_candidates(), SEGMENTS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="经典趋势追踪 vs 自适应双均线（只读本地缓存，不联网）"
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="ccxt 符号，现货")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="K 线周期")
    args = parser.parse_args()
    run_duel(args.symbol, args.timeframe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
