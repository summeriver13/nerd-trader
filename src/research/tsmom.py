"""
src/research/tsmom.py
多标的时序动量（TSMOM）：把单标的趋势信号拼成组合，口径与趋势线一致。

设计要点都是被前面的研究逼出来的：

- **多速度平均而不是挑速度**：同一标的跑多个入场窗口，仓位取平均。挑窗口就是又一次
  搜索，而本仓纪律是报平台不挑点（趋势线已实测：入场窗口 20/55/100 在 BTC/ETH 上恒正，
  所以这里直接取三档平均，不选）。
- **波动率缩放只缩不放**：仓位按近期实现波动缩放到目标，上限 1——不加杠杆，因此不引入
  强平与资金费建模（`docs/vol_target` 已证明缩放能可靠降波动、不能可靠提夏普）。
- **等权 1/N_t**：每个日期按当时有数据的标的数等分资金，新上市标的不会因为数据短而
  拿到额外权重，也不会因为池子变大而稀释老标的。
- 信号到仓位的滞后由引擎统一处理（`engine.py`：`signal.shift(1)`），本模块不重复滞后。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.core.costs import FeeSchedule
from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine
from src.research.strategies.donchian_trend import DonchianTrendStrategy

# 多速度档位：短/中/长，取自趋势线已实测恒正的邻域（20/55/100）
DEFAULT_ENTRY_PERIODS: tuple[int, ...] = (20, 55, 100)
# 出场窗口 = 入场窗口 / 该比例，且不小于下界（与 trend_duel 的扫描口径一致）
EXIT_RATIO = 3
MIN_EXIT_PERIOD = 5
# 波动率缩放：目标年化波动与回看根数
DEFAULT_VOL_TARGET = 0.40
DEFAULT_VOL_LOOKBACK = 30
BARS_PER_YEAR = 365
INITIAL_CAPITAL = 10_000.0
LEVERAGE = 1.0


@dataclass(frozen=True)
class TsmomOptions:
    """多速度趋势信号的口径。"""

    entry_periods: tuple[int, ...] = DEFAULT_ENTRY_PERIODS
    atr_mult: float = 0.0  # 0 = 无止损（趋势线实测：单倍杠杆下止损削自己）
    vol_target: float | None = DEFAULT_VOL_TARGET
    vol_lookback: int = DEFAULT_VOL_LOOKBACK


def donchian_signal(
    bars: pd.DataFrame, entry: int, *, atr_mult: float = 0.0
) -> pd.Series:
    """单速度唐奇安信号（0/1）：复用策略类，保证与趋势线同一实现。"""
    exit_period = max(entry // EXIT_RATIO, MIN_EXIT_PERIOD)
    strategy = DonchianTrendStrategy(
        entry_period=entry, exit_period=exit_period, atr_mult=atr_mult
    )
    data = strategy.generate_signals(strategy.calculate_indicators(bars.copy()))
    return cast(pd.Series, data["signal"])


def multi_speed_signal(bars: pd.DataFrame, options: TsmomOptions) -> pd.Series:
    """多速度信号：各速度的 0/1 信号取平均（取值 0、1/n、…、1）。"""
    signals = [
        donchian_signal(bars, entry, atr_mult=options.atr_mult)
        for entry in options.entry_periods
    ]
    return cast(pd.Series, pd.concat(signals, axis=1).mean(axis=1))


def volatility_scale(bars: pd.DataFrame, options: TsmomOptions) -> pd.Series:
    """波动率缩放系数 min(1, 目标/实现波动)：只缩不放，暖机期取 0。

    实现波动只用截至当根的收益（引擎再把仓位滞后一根），故不引入前视。
    """
    if options.vol_target is None:
        return pd.Series(1.0, index=bars.index)
    returns = cast(pd.Series, bars["close"].pct_change())
    realized = returns.rolling(options.vol_lookback).std() * np.sqrt(BARS_PER_YEAR)
    scale = options.vol_target / realized
    return cast(pd.Series, scale.clip(upper=1.0).fillna(0.0))


def tsmom_signal(bars: pd.DataFrame, options: TsmomOptions) -> pd.Series:
    """给引擎用的 signal 列：多速度信号 × 波动率缩放。"""
    signal = multi_speed_signal(bars, options)
    if options.vol_target is None:
        return signal
    return cast(pd.Series, signal * volatility_scale(bars, options))


@dataclass(frozen=True)
class SymbolRun:
    """一个标的、一个信号下的引擎输出（逐根，单位是初始本金的收益率）。"""

    net_ret: pd.Series
    pos: pd.Series
    cost: pd.Series


def run_symbol(
    bars: pd.DataFrame, signal: pd.Series, *, cost: FeeSchedule | None = None
) -> SymbolRun:
    """跑引擎取逐根净收益、仓位与成本；成本默认与趋势线同口径（永续 taker + 滑点）。"""
    data = bars.copy()
    data["signal"] = signal
    domain = (
        CryptoDomain()
        if cost is None
        else CryptoDomain(fee_rate=cost.taker, slippage=cost.slippage)
    )
    engine = BacktestEngine(
        domain=domain, initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE
    )
    result = engine.run(data)
    frame = cast(pd.DataFrame, result["df"])
    return SymbolRun(
        net_ret=cast(pd.Series, frame["net_ret"]),
        pos=cast(pd.Series, frame["pos"]),
        cost=cast(pd.Series, frame["cost"]),
    )


def cross_section_mean(frame: pd.DataFrame) -> pd.Series:
    """按行取截面均值（跳过缺失）：等权 1/N_t 的通用形式。

    自写而不是直接写 `frame.mean(axis=1)`：类型桩会把后者推成 float，
    而本仓禁用 `# pyright: ignore` 之类的强制手段。
    """
    return cast(pd.Series, frame.mean(axis=1))


def portfolio_returns(streams: dict[str, pd.Series]) -> pd.Series:
    """等权组合的逐根净收益：按当时有数据的标的数等分（缺失标的退出分母）。"""
    return cross_section_mean(pd.DataFrame(streams))


def active_counts(streams: dict[str, pd.Series]) -> pd.Series:
    """每个日期参与组合的标的数（即 1/N_t 里的 N）。"""
    return cast(pd.Series, pd.DataFrame(streams).notna().sum(axis=1))
