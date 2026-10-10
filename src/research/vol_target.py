"""
src/research/vol_target.py
波动率目标仓位的研究口径：参数网格、暴露路径、等平均暴露对照。

核心问题：按近期实现波动率反比调仓位，能否改善 BTC 的风险调整收益？

三条必须守住的口径：

1. **无前视**：暴露由第 t 根及以前的收盘价算出（`VolatilityTarget.exposure` 的
   rolling std 含当前根，这是对的），引擎再 `shift(1)` 落到第 t+1 根持仓。
2. **必须跟「等平均暴露的固定仓位」比**（`matched_control`）。波动率目标天然会
   降低平均暴露，而回撤随暴露近似线性缩小——不跟同平均暴露的固定仓位比，任何
   降暴露的做法都会在回撤指标上「赢」，而那只是仓位变小，不是波动率目标带来的。
   这是 Moreira & Muir (2017) 的标准对照，缺了它整个研究无效。
3. **成本与其它研究同口径**：走 `BacktestEngine` + `CryptoDomain`，换手成本按
   `pos_change` 计，指标用 `factors/portfolio.leg_stats`（复利年化 + 夏普）。

`max_exposure=1.0` 是「只降杠杆」版本，`>1` 才允许在低波动期加仓——两者的经济
含义完全不同，报结论时必须分开，不能混在一个「最优参数」里挑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

import pandas as pd

from src.analytics.metrics import ANNUAL_DAYS
from src.core.costs import MARKET_SLIPPAGE, PERP_TAKER_FEE
from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine
from src.research.factors.portfolio import LegStats, leg_stats
from src.research.position_sizing import VolatilityTarget
from src.research.strategies.buy_and_hold import BuyAndHoldStrategy

INITIAL_CAPITAL: Final[float] = 10_000.0
BASELINE_LABEL: Final[str] = "buy_and_hold"

DEFAULT_TARGET_VOLS: Final[tuple[float, ...]] = (0.20, 0.30, 0.40, 0.50, 0.60)
DEFAULT_LOOKBACKS: Final[tuple[int, ...]] = (10, 20, 30, 60, 90)
DEFAULT_REBALANCES: Final[tuple[int, ...]] = (1, 7, 30)
DEFAULT_MAX_EXPOSURES: Final[tuple[float, ...]] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class VolTargetSpec:
    """一个波动率目标参数组合。"""

    target_vol: float
    lookback: int
    rebalance_every: int
    max_exposure: float

    def label(self) -> str:
        """短标签，用于表格与落盘（宽度可控）。"""
        return (
            f"tv{self.target_vol:.0%}"
            f"_lb{self.lookback}"
            f"_rb{self.rebalance_every}"
            f"_mx{self.max_exposure:g}"
        )

    def sizing(self) -> VolatilityTarget:
        """转成仓位缩放层认识的参数对象。"""
        return VolatilityTarget(
            target_vol=self.target_vol,
            lookback=self.lookback,
            rebalance_every=self.rebalance_every,
            max_exposure=self.max_exposure,
        )


@dataclass(frozen=True)
class VolTargetRun:
    """一个参数组合（或对照）的实测读数。"""

    label: str
    spec: VolTargetSpec | None
    stats: LegStats
    avg_exposure: float
    turnover: float
    cost: float
    net_ret: pd.Series


def build_grid(
    target_vols: tuple[float, ...] = DEFAULT_TARGET_VOLS,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    rebalances: tuple[int, ...] = DEFAULT_REBALANCES,
    max_exposures: tuple[float, ...] = DEFAULT_MAX_EXPOSURES,
) -> list[VolTargetSpec]:
    """全网格。每个维度都是真实的设计选择，组合数就是多重检验的试验次数。"""
    return [
        VolTargetSpec(target_vol, lookback, rebalance, max_exposure)
        for target_vol in target_vols
        for lookback in lookbacks
        for rebalance in rebalances
        for max_exposure in max_exposures
    ]


def _run_signal(
    bars: pd.DataFrame,
    signal: pd.Series,
    label: str,
    *,
    spec: VolTargetSpec | None,
    fee_rate: float,
    slippage: float,
) -> VolTargetRun:
    """把一个目标仓位序列喂进引擎，返回净收益与读数。"""
    data = bars.copy()
    data["signal"] = signal
    engine = BacktestEngine(
        domain=CryptoDomain(fee_rate=fee_rate, slippage=slippage),
        initial_capital=INITIAL_CAPITAL,
        leverage=1.0,
    )
    result: dict[str, Any] = engine.run(data)
    if not result:
        raise RuntimeError(f"{label} 的回测未返回结果")
    frame = cast(pd.DataFrame, result["df"])
    net_ret = cast(pd.Series, frame["net_ret"])
    pos = cast(pd.Series, frame["pos"])
    return VolTargetRun(
        label=label,
        spec=spec,
        stats=leg_stats(net_ret, ANNUAL_DAYS),
        avg_exposure=float(pos.abs().mean()),
        turnover=float(pos.diff().abs().fillna(0.0).mean()),
        cost=float(result["total_cost"]),
        net_ret=net_ret,
    )


def run_spec(
    bars: pd.DataFrame,
    spec: VolTargetSpec,
    *,
    fee_rate: float = PERP_TAKER_FEE,
    slippage: float = MARKET_SLIPPAGE,
) -> VolTargetRun:
    """按参数组合生成暴露路径并回测。"""
    close = cast(pd.Series, bars["close"])
    return _run_signal(
        bars,
        spec.sizing().exposure(close),
        spec.label(),
        spec=spec,
        fee_rate=fee_rate,
        slippage=slippage,
    )


def run_baseline(
    bars: pd.DataFrame,
    *,
    fee_rate: float = PERP_TAKER_FEE,
    slippage: float = MARKET_SLIPPAGE,
) -> VolTargetRun:
    """满仓买入持有（暴露恒为 1）。任何改进都必须先打败它。"""
    signal = BuyAndHoldStrategy().generate_signals(bars.copy())["signal"]
    return _run_signal(
        bars,
        cast(pd.Series, signal),
        BASELINE_LABEL,
        spec=None,
        fee_rate=fee_rate,
        slippage=slippage,
    )


def run_constant(
    bars: pd.DataFrame,
    level: float,
    label: str,
    *,
    fee_rate: float = PERP_TAKER_FEE,
    slippage: float = MARKET_SLIPPAGE,
) -> VolTargetRun:
    """恒定暴露对照：一次性建仓后不再动，用于回答「只是仓位变小了吗」。"""
    signal = pd.Series(level, index=bars.index, dtype=float)
    return _run_signal(
        bars, signal, label, spec=None, fee_rate=fee_rate, slippage=slippage
    )


def match_level(avg_exposure: float, n_bars: int) -> float:
    """把「平均 |pos|」反解成恒定 signal 值。

    引擎把 signal 右移一根并给首根补 0，所以恒定 signal=c 的平均 |pos| 是
    c×(n−1)/n；反解即 c = E×n/(n−1)。不反解的话对照的平均暴露会低 n 分之一，
    在 n=3195 时无所谓，但口径要写死免得以后换短样本时悄悄失配。
    """
    if n_bars < 2:
        return avg_exposure
    return avg_exposure * n_bars / (n_bars - 1)


def matched_control(
    bars: pd.DataFrame,
    run: VolTargetRun,
    *,
    fee_rate: float = PERP_TAKER_FEE,
    slippage: float = MARKET_SLIPPAGE,
) -> VolTargetRun:
    """等平均暴露对照：把 run 的平均暴露压成固定仓位再跑一次。

    这是本研究的核心证伪。波动率目标降低回撤有两个来源——「波动高时减仓」和
    「平均仓位本来就小」。前者是策略，后者只是少下注。这个对照把后者单独拿出来，
    两者之差才是波动率目标的真实贡献。
    """
    level = match_level(run.avg_exposure, bars.index.size)
    return run_constant(
        bars,
        level,
        f"const_{run.avg_exposure:.3f}",
        fee_rate=fee_rate,
        slippage=slippage,
    )


def run_grid(
    bars: pd.DataFrame,
    specs: list[VolTargetSpec],
    *,
    fee_rate: float = PERP_TAKER_FEE,
    slippage: float = MARKET_SLIPPAGE,
) -> list[VolTargetRun]:
    """按网格顺序跑完全部参数组合。"""
    return [
        run_spec(bars, spec, fee_rate=fee_rate, slippage=slippage) for spec in specs
    ]
