"""单策略统一测试：全策略同口径评估、参数邻域与多重检验校正，并落盘收益流。

阶段 1（本脚本）：把一个标的上所有策略放在同一数据、同一引擎、同一成本下评估，
产出四张表——默认参数对比、参数邻域稳健性、去膨胀夏普（DSR）、回测过拟合概率（PBO），
并把逐根净收益序列落盘，作为阶段 2 相关性分析与阶段 3 组合构建的原料。

用法：
    uv run python scripts/strategy_suite.py
    uv run python scripts/strategy_suite.py --symbols BTC/USDT,ETH/USDT

产出：
- 控制台四张表
- data/results/strategy_suite/<SYMBOL>_net_ret.parquet：宽表，index 为时间、列为策略，
  只含各策略的净收益（已扣手续费、滑点）
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.analytics.metrics import sharpe_ratio
from src.config.paths import crypto_bar_path, symbol_stem
from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine
from src.research.overfitting import (
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_estimator_variance,
)
from src.research.registry import (
    STRATEGY_MAP,
    STRATEGY_SHORT_NAMES,
    resolve_symbols,
)
from src.research.strategies.base import BaseStrategy
from src.research.strategies.buy_and_hold import BuyAndHoldStrategy

EXCHANGE = "okx"
TIMEFRAME = "1d"
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 1.0
OUTPUT_DIR = Path("data/results/strategy_suite")
BASELINE_KEY = "buy-and-hold"
# 参数邻域：每个策略只动一个核心参数，其余保持出厂默认，用于判断结论是否依赖单点
NEIGHBORHOODS: dict[str, dict[str, tuple[float, ...]]] = {
    "1": {"bull_fast": (8, 10, 14)},
    "2": {"bull_fast": (8, 10, 14)},
    "3": {"window": (15, 20, 30)},
    "4": {"rsrs_n": (14, 18, 24)},
    "5": {"entry_period": (20, 55, 100)},
}
VARIANT_WIDTH = 20


@dataclass(frozen=True)
class Row:
    """一个策略在一个标的上的绩效读数。"""

    key: str
    total_ret: float
    annual_ret: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    trades: int
    win_rate: float
    exposure: float
    cost: float


def load_bars(symbol: str) -> pd.DataFrame:
    """只读本地 parquet 缓存，不联网。"""
    return pd.read_parquet(crypto_bar_path(EXCHANGE, symbol, TIMEFRAME))


def build_strategies() -> list[tuple[str, BaseStrategy]]:
    """出厂默认参数下的全部候选，含买入持有基准。"""
    items: list[tuple[str, BaseStrategy]] = [(BASELINE_KEY, BuyAndHoldStrategy())]
    for key, conf in STRATEGY_MAP.items():
        params: dict[str, Any] = conf["default_params"]
        items.append((key, conf["class"](**params)))
    return items


def run_one(
    bars: pd.DataFrame, strategy: BaseStrategy
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    """算指标、生成信号、跑引擎，返回引擎结果、逐根净收益与持仓序列。"""
    data = strategy.generate_signals(strategy.calculate_indicators(bars.copy()))
    engine = BacktestEngine(
        domain=CryptoDomain(), initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE
    )
    result = engine.run(data)
    if not result:
        raise RuntimeError("回测未返回结果")
    frame = cast(pd.DataFrame, result["df"])
    return (
        result,
        cast(pd.Series, frame["net_ret"]),
        cast(pd.Series, frame["pos"]),
    )


def to_row(key: str, result: dict[str, Any]) -> Row:
    """从引擎结果取绩效读数。"""
    return Row(
        key=key,
        total_ret=float(result["total_ret"]),
        annual_ret=float(result["annual_ret"]),
        sharpe=float(result["sharpe_ratio"]),
        sortino=float(result["sortino_ratio"]),
        calmar=float(result["calmar_ratio"]),
        max_dd=float(result["max_drawdown"]),
        trades=int(result["trade_count"]),
        win_rate=float(result["win_rate"]),
        exposure=float(result["exposure"]),
        cost=float(result["total_cost"]),
    )


def print_metrics(title: str, rows: list[Row]) -> None:
    """打印统一评估表（夏普为年化，展示口径）。"""
    print(f"\n{title}")
    header = (
        f"{'strategy':<12}{'total':>10}{'annual':>9}{'sharpe':>8}{'sortino':>9}"
        f"{'calmar':>8}{'maxDD':>9}{'trades':>8}{'win':>7}{'expo':>7}{'cost':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{STRATEGY_SHORT_NAMES.get(row.key, row.key):<12}{row.total_ret:>9.1%}"
            f"{row.annual_ret:>9.1%}{row.sharpe:>8.2f}{row.sortino:>9.2f}"
            f"{row.calmar:>8.2f}{row.max_dd:>9.1%}{row.trades:>8}{row.win_rate:>7.1%}"
            f"{row.exposure:>7.1%}{row.cost:>9.0f}"
        )


def neighborhood_rows(bars: pd.DataFrame, key: str) -> list[tuple[str, Row]]:
    """跑一个策略的参数邻域，返回 (变体标签, 绩效)。"""
    specs = NEIGHBORHOODS.get(key)
    if not specs:
        return []
    conf = STRATEGY_MAP[key]
    base_params: dict[str, Any] = conf["default_params"]
    out: list[tuple[str, Row]] = []
    for param, values in specs.items():
        for value in values:
            params = {**base_params, param: value}
            variant = f"{param}={value}"
            if base_params.get(param) == value:
                variant += " (base)"
            result, _, _ = run_one(bars, conf["class"](**params))
            out.append((variant, to_row(key, result)))
    return out


def print_neighborhood(title: str, rows: list[tuple[str, Row]]) -> None:
    """打印参数邻域稳健性表。"""
    print(f"\n{title}")
    header = (
        f"{'strategy':<12}{'variant':<24}{'total':>11}{'annual':>9}"
        f"{'sharpe':>8}{'maxDD':>9}{'trades':>8}"
    )
    print(header)
    print("-" * len(header))
    for variant, row in rows:
        print(
            f"{STRATEGY_SHORT_NAMES.get(row.key, row.key):<12}{variant:<24}"
            f"{row.total_ret:>10.1%}{row.annual_ret:>9.1%}{row.sharpe:>8.2f}"
            f"{row.max_dd:>9.1%}{row.trades:>8}"
        )


def print_overfitting(
    title: str, series_map: dict[str, pd.Series], n_bars: int
) -> None:
    """打印 DSR 与 PBO：这批策略里挑冠军是不是运气。"""
    keys = list(series_map)
    sharpes = [sharpe_ratio(series_map[k]) for k in keys]
    variance = sharpe_estimator_variance(sharpes, n_bars)
    expected_max = expected_max_sharpe(variance, trials=len(keys))

    print(f"\n{title}")
    print("口径：夏普为逐根（未年化），×sqrt(365) 即年化；DSR/PBO 必须用逐根口径")
    print(f"trials={len(keys)}  bars={n_bars}  expected_max={expected_max:.4f}")
    header = (
        f"{'strategy':<12}{'sharpe/bar':>11}{'exp_max':>9}{'DSR':>8}{'obs':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        test = deflated_sharpe(
            STRATEGY_SHORT_NAMES.get(key, key), series_map[key], expected_max
        )
        verdict = (
            "n/a (few obs)"
            if not test.sufficient
            else ("PASS" if test.passed else "FAIL")
        )
        print(
            f"{STRATEGY_SHORT_NAMES.get(key, key):<12}{test.sharpe:>11.4f}"
            f"{test.expected_max:>9.4f}{test.dsr:>8.3f}{test.observations:>7}  {verdict}"
        )

    pbo = probability_of_backtest_overfitting(series_map)
    if pbo is None:
        print("\nPBO: 样本不足以分块（或候选退化），未定义")
        return
    verdict = "PASS" if pbo.passed else "FAIL"
    print(
        f"\nPBO CSCV: blocks={pbo.blocks} splits={pbo.splits} pbo={pbo.pbo:.1%} "
        f"winner_oos_sharpe_median={pbo.winner_oos_sharpe_median:.2f} -> {verdict}"
    )
    print("解读：样本内冠军在样本外跌到中位数以下的比例，越高说明「挑冠军」越像运气")


def run_suite(symbol: str) -> None:
    """跑一个标的的完整单策略测试。"""
    bars = load_bars(symbol)
    print(
        f"\n{'=' * 78}\n标的 {symbol} | {TIMEFRAME} | {len(bars)} 根 | "
        f"{bars.index[0]:%Y-%m-%d} -> {bars.index[-1]:%Y-%m-%d}\n"
        f"口径：初始资金 {INITIAL_CAPITAL:.0f} USDT、{LEVERAGE}x、市价单、含手续费与滑点"
        f"\n{'=' * 78}"
    )

    rows: list[Row] = []
    series_map: dict[str, pd.Series] = {}
    pos_map: dict[str, pd.Series] = {}
    for key, strategy in build_strategies():
        result, net_ret, pos = run_one(bars, strategy)
        rows.append(to_row(key, result))
        series_map[key] = net_ret
        pos_map[key] = pos
    print_metrics("=== 默认参数全策略对比 ===", rows)

    neighbors: list[tuple[str, Row]] = []
    for key in NEIGHBORHOODS:
        neighbors.extend(neighborhood_rows(bars, key))
    print_neighborhood("=== 参数邻域稳健性 ===", neighbors)

    print_overfitting("=== 多重检验校正 ===", series_map, len(bars))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = symbol_stem(symbol)
    ret_path = OUTPUT_DIR / f"{stem}_net_ret.parquet"
    pos_path = OUTPUT_DIR / f"{stem}_pos.parquet"
    pd.DataFrame(series_map).to_parquet(ret_path, compression="snappy")
    pd.DataFrame(pos_map).to_parquet(pos_path, compression="snappy")
    print(f"\n收益流已落盘：{ret_path}\n持仓序列已落盘：{pos_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单策略统一测试（只读本地缓存，不联网）"
    )
    parser.add_argument(
        "--symbols", default=None, help="逗号分隔，默认 BTC/USDT 与 ETH/USDT"
    )
    args = parser.parse_args()
    for symbol in resolve_symbols(args.symbols):
        run_suite(symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
