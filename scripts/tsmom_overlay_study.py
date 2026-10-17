"""scripts/tsmom_overlay_study.py
机构拿到一个候选策略会先跑的两件事：执行敏感性、组合层边际贡献。

用途（`docs/trend/README.md` §组合层定位与执行敏感性 的读数出处）：

1. **执行敏感性**：收益序列固定为 close-to-close，只换成本档位与成交时点。
   如果读数在成本 ×3、开盘成交、信号再滞一根下都站得住，说明它不靠「抢跑成本优势」
   或「信号日当天的一段行情」赚钱，实施风险低；反之则要按真实成交建模。
2. **组合层**：把策略当 overlay 叠在 BTC 买入持有上扫描权重。单看「年化跑不过 BTC」
   并不否决一个策略——机构关心的是加进去之后组合的夏普与回撤是否变好，以及最优权重。

用法::

    uv run python scripts/tsmom_overlay_study.py            # 两张表都打
    uv run python scripts/tsmom_overlay_study.py --min-bars 365

口径：现货 taker + 滑点 15bp/边（`SPOT_FEES.taker_net()`），等权 1/N_t 与持仓滞后
一根都与 `src/research/trend/paper.py` 同源（复用其 `signal_panel`），只多一层权重混合。
"""

from __future__ import annotations

import argparse
from typing import Any, cast

import numpy as np
import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.core.costs import SPOT_FEES
from src.research.study.universe import load_bars_dir
from src.research.trend.paper import PaperSpec, signal_panel

BARS_PER_YEAR = 365
SPOT_DIR = crypto_dataset_dir("okx", "spot", "1d")
BTC_STEM = "BTC-USDT"
WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
COST_MULTIPLES: tuple[float, ...] = (1.0, 2.0, 3.0)


def stats(net: pd.Series, nav: pd.Series, exposure: pd.Series) -> dict[str, float]:
    """与 `paper.forward_stats` 同口径的读数（年化/波动/夏普/回撤/暴露）。"""
    total = float(nav.iloc[-1]) - 1.0
    years = len(net) / BARS_PER_YEAR
    std = float(net.std())
    return {
        "total": total,
        "annual": (1.0 + total) ** (1.0 / years) - 1.0,
        "vol": std * np.sqrt(BARS_PER_YEAR),
        "sharpe": float(net.mean()) / std * np.sqrt(BARS_PER_YEAR),
        "maxdd": float((nav / nav.cummax() - 1.0).min()),
        "expo": float(exposure.mean()),
    }


def run(
    returns: pd.DataFrame, held: pd.DataFrame, cost: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """向量化撮合：净收益 = 持仓收益 − 换手成本，返回 (净收益, 净值, 暴露)。"""
    gross = cast(pd.Series, (held * returns).sum(axis=1))
    turnover = cast(pd.Series, held.diff().abs().sum(axis=1))
    turnover.iloc[0] = 0.0
    net = cast(pd.Series, gross - turnover * cost)
    return net, (1.0 + net).cumprod(), cast(pd.Series, held.abs().sum(axis=1))


def build(min_bars: int) -> dict[str, Any]:
    """装载标的池并产出「收益面板 + 两种滞后的持仓面板」。"""
    bars_map = load_bars_dir(SPOT_DIR, min_bars)
    if BTC_STEM not in bars_map:
        raise SystemExit(f"标的池里没有 {BTC_STEM}，无法做 BTC 对照")
    close = pd.DataFrame({stem: bars["close"] for stem, bars in bars_map.items()})
    opens = pd.DataFrame({stem: bars["open"] for stem, bars in bars_map.items()})
    counts = close.pct_change().notna().sum(axis=1).astype(float)
    signals = signal_panel(bars_map, PaperSpec())
    # 分母必须取持仓当日（除法在 shift 之后），与台账口径一致
    held = cast(pd.DataFrame, signals.shift(1).div(counts, axis=0).fillna(0.0))
    delayed = cast(pd.DataFrame, signals.shift(2).div(counts, axis=0).fillna(0.0))
    return {
        "close": close,
        "opens": opens,
        "held": held,
        "delayed": delayed,
        "bars_map": bars_map,
    }


def print_execution(data: dict[str, Any], cost: float) -> pd.Series:
    """表一：执行敏感性，返回基准净收益（表二要用）。"""
    close = data["close"]
    close_returns = close.pct_change()
    rows: dict[str, dict[str, float]] = {}
    baseline: pd.Series | None = None
    for multiple in COST_MULTIPLES:
        net, nav, expo = run(close_returns, data["held"], cost * multiple)
        label = f"现金成本 {cost * multiple * 1e4:.0f}bp/边" + (
            "（基准）" if multiple == 1 else ""
        )
        rows[label] = stats(net, nav, expo)
        if multiple == 1.0:
            baseline = net
    opens = data["opens"]
    net, nav, expo = run(opens.shift(-1) / opens - 1.0, data["held"], cost)
    rows["开盘成交（open-to-open）"] = stats(net, nav, expo)
    net, nav, expo = run(close_returns, data["delayed"], cost)
    rows["信号再滞一根"] = stats(net, nav, expo)

    print("\n=== A. 执行敏感性（收益恒为 close-to-close，只换成本与成交时点）===")
    print(
        f"{'口径':<26}{'总收益':>10}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>10}{'暴露':>8}"
    )
    for label, row in rows.items():
        print(
            f"{label:<26}{row['total']:>9.1%}{row['annual']:>9.2%}{row['vol']:>9.1%}"
            f"{row['sharpe']:>8.2f}{row['maxdd']:>10.1%}{row['expo']:>8.2%}"
        )
    assert baseline is not None
    return baseline


def print_overlay(data: dict[str, Any], strategy: pd.Series) -> None:
    """表二：组合层扫描，最后一行按等风险预算（逆向波动）给权重作对照。"""
    btc_ret = cast(pd.Series, data["close"][BTC_STEM].pct_change().fillna(0.0))
    inverse_vol = (1.0 / float(strategy.std())) / (
        1.0 / float(strategy.std()) + 1.0 / float(btc_ret.std())
    )
    print("\n=== B. 组合层：把策略当 overlay 叠在 BTC 买入持有上 ===")
    print(
        f"（波动：策略 {float(strategy.std()) * np.sqrt(BARS_PER_YEAR):.1%}"
        f" vs BTC {float(btc_ret.std()) * np.sqrt(BARS_PER_YEAR):.1%}，"
        f"等风险预算权重 w_策略 = {inverse_vol:.0%} 属事后口径，只作对照）\n"
    )
    print(
        f"{'权重 策略/BTC':<18}{'总收益':>10}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>10}"
    )
    grid = sorted(set(WEIGHTS + (round(inverse_vol, 2),)))
    for weight in grid:
        blend = cast(pd.Series, weight * strategy + (1.0 - weight) * btc_ret)
        nav = (1.0 + blend).cumprod()
        row = stats(blend, nav, pd.Series(1.0, index=nav.index))
        print(
            f"{f'{weight:.0%} / {1 - weight:.0%}':<18}{row['total']:>9.1%}{row['annual']:>9.2%}"
            f"{row['vol']:>9.1%}{row['sharpe']:>8.2f}{row['maxdd']:>10.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="执行敏感性与组合层 overlay 扫描")
    parser.add_argument(
        "--min-bars", type=int, default=365, help="标的入池的最少 K 线根数"
    )
    args = parser.parse_args()

    data = build(args.min_bars)
    print(
        f"标的池 {len(data['bars_map'])} 只（≥{args.min_bars} 根）"
        f" ｜ 单边成本 {SPOT_FEES.taker_net() * 1e4:.1f}bp"
        f"（现货 taker {SPOT_FEES.taker * 1e4:.0f}bp + 滑点 {SPOT_FEES.slippage * 1e4:.0f}bp）"
    )
    strategy = print_execution(data, SPOT_FEES.taker_net())
    print_overlay(data, strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
