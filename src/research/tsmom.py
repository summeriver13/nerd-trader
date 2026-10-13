"""
src/research/tsmom.py
多标的时序动量（TSMOM）：把单标的趋势信号拼成组合，口径与趋势线一致。

设计要点都是被前面的研究逼出来的：

- **多速度平均而不是挑速度**：同一标的跑多个入场窗口，仓位取平均。挑窗口就是又一次
  搜索，而本仓纪律是报平台不挑点（趋势线已实测：入场窗口 20/55/100 在 BTC/ETH 上恒正，
  所以这里直接取三档平均，不选）。
- **波动率缩放只缩不放**：仓位按近期实现波动缩放到目标，上限 1——不加杠杆，因此不引入
  强平建模（`docs/vol_target` 已证明缩放能可靠降波动、不能可靠提夏普）。现货臂不涉及
  资金费；永续多空臂（`allow_short=True`）必须把日频资金费传给引擎，否则空头白拿补贴。
- **空头是镜像状态机**：策略类只多，做空另写（下破 entry 窗口最低价开空、上破 exit
  窗口最高价平空），且只做单倍杠杆不做逐仓，故不建模强平。
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
    allow_short: bool = False  # 现货口径只多；永续口径开空头镜像


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


def donchian_short_signal(bars: pd.DataFrame, entry: int) -> pd.Series:
    """单速度做空信号（0/−1）：下破 entry 窗口最低价开空，上破 exit 窗口最高价平空。

    多头复用策略类，做空没有对应实现，这里按同一状态机镜像写：上下轨都用 shift(1)
    只看已完成的历史；ATR 止损是给多头臂的（趋势线实测单倍杠杆下止损削自己），
    做空臂不带走损。
    """
    exit_period = max(entry // EXIT_RATIO, MIN_EXIT_PERIOD)
    close = bars["close"].to_numpy(dtype=float)
    lower = bars["low"].rolling(entry).min().shift(1).to_numpy(dtype=float)
    upper = bars["high"].rolling(exit_period).max().shift(1).to_numpy(dtype=float)
    signals = np.zeros(len(close))
    holding = False
    for i in range(len(close)):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if not holding:
            if close[i] < lower[i]:
                holding = True
                signals[i] = -1.0
            continue
        if close[i] > upper[i]:
            holding = False
        else:
            signals[i] = -1.0
    return cast(pd.Series, pd.Series(signals, index=bars.index))


def multi_speed_signal(bars: pd.DataFrame, options: TsmomOptions) -> pd.Series:
    """多速度信号：各速度取平均（只多 0…1；多空 −1…1）。

    多空时先按速度把多头与空头合成有符号信号（两者互斥，同时成立则归零），再跨速度
    取平均——先把三档都算成 ±1 再平均，跟把多头与空头信号各算一遍再除以两倍速度数
    得到的仓位尺度不同。
    """
    signals = []
    for entry in options.entry_periods:
        leg = donchian_signal(bars, entry, atr_mult=options.atr_mult)
        if options.allow_short:
            leg = leg + donchian_short_signal(bars, entry)
        signals.append(leg)
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
    bars: pd.DataFrame,
    signal: pd.Series,
    *,
    cost: FeeSchedule | None = None,
    funding: pd.Series | None = None,
) -> SymbolRun:
    """跑引擎取逐根净收益、仓位与成本。

    成本默认与趋势线同口径（永续 taker + 滑点）；``funding`` 是按日资金费率
    （正 = 多头付钱，空头收钱），用 ``daily_rates`` 把 8h 结算压成日频后再传，
    缺日引擎按 0 处理。只多现货臂可省，永续多空臂必须传。
    """
    data = bars.copy()
    data["signal"] = signal
    domain = (
        CryptoDomain()
        if cost is None
        else CryptoDomain(fee_rate=cost.taker, slippage=cost.slippage)
    )
    engine = BacktestEngine(
        domain=domain,
        initial_capital=INITIAL_CAPITAL,
        leverage=LEVERAGE,
        funding=funding,
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


# === 正交性与账本级缩放（研究侧工具）===


def trailing_volatility(
    bars: pd.DataFrame, lookback: int = DEFAULT_VOL_LOOKBACK
) -> pd.Series:
    """标的自身的滚动实现波动（年化），只用截至当根的收益。"""
    returns = cast(pd.Series, bars["close"].pct_change())
    return cast(pd.Series, returns.rolling(lookback).std() * np.sqrt(BARS_PER_YEAR))


def book_scale(
    book_returns: pd.Series,
    *,
    target: float,
    cap: float,
    lookback: int = DEFAULT_VOL_LOOKBACK,
) -> pd.Series:
    """账本级缩放系数：min(cap, 目标 / 账本滚动实现波动)，只用截至当根的收益。

    与逐标的缩放的差别在分母：这里量的是**组合**的波动，所以能把账本整体推到目标；
    逐标的 40% 目标在信号多数时间为零时只换来约 12% 的账本波动。
    """
    realized = book_returns.rolling(lookback).std() * np.sqrt(BARS_PER_YEAR)
    scale = target / realized
    return cast(pd.Series, scale.clip(upper=cap).fillna(0.0))


def split_portfolio(
    streams: dict[str, pd.Series],
    vols: dict[str, pd.Series],
    *,
    low: bool,
    share: float = 0.5,
    min_members: int = 1,
) -> pd.Series:
    """按每日滚动波动分位把标的切两半，取其中一半的等权组合收益。

    ``vols`` 必须已滞后一根（用 t−1 已知的波动决定 t 的成员），否则就是前视；
    没有波动读数的标的当天自动退出分母。``min_members`` 是宽度门槛：半池成员少于
    它时返回 NaN——池子早期只有两只标的，半池单标的爆掉会把回撤读成 −100%。
    """
    frame = pd.DataFrame(streams)
    rank = pd.DataFrame(vols).reindex_like(frame).rank(axis=1, pct=True)
    mask = rank <= share if low else rank > share
    masked = frame.where(mask)
    members = cast(pd.Series, masked.notna().sum(axis=1))
    return cast(pd.Series, cross_section_mean(masked).where(members >= min_members))
