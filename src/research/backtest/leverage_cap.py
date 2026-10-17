"""
src/research/backtest/leverage_cap.py
容量口径的杠杆额度：逐标的、逐根的倍率上限。

`src/research/trend/tsmom.py:book_scale` 把账本整体推到目标波动（上限 3 倍），但它假设每个
标的都能同倍放大。真实约束不是「几倍杠杆」，而是「这笔钱占该标的成交额多少」——细价币
上 3 倍等于自己造出一个市场。把约束写成可算的形式：

    单标的仓位（美元）= (1 / N_t) × 倍率_i,t × M  ≤  参与率 × ADV_i,t

反解出当天允许的倍率：

    允许倍率_i,t = 参与率 × ADV_i,t × N_t / M

最终倍率_i,t = min(账本倍率_t, 允许倍率_i,t, 上限)。ADV 取滞后 ``lag`` 根的中位日成交
额（只用 t−lag 及更早的读数，不引入前视），所以 M 越小越接近无约束的 3 倍、M 越大越被
池尾压住——扫一遍 M 得到的就是容量上限。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.core.costs import FeeSchedule
from src.research.trend.tsmom import (
    SymbolRun,
    TsmomOptions,
    multi_speed_signal,
    run_symbol,
)

# 参与率：单标的仓位不超过其滞后中位日成交额的比例（建仓冲击的粗略上限）
PARTICIPATION = 0.05
ADV_WINDOW = 30
ADV_LAG = 1
MAX_MULTIPLIER = 3.0
# 账本名义规模的默认扫描档：从「细价币也装得下」到「只有头部十几只装得下」
DEFAULT_NOTIONALS: tuple[float, ...] = (1e5, 1e6, 1e7, 1e8)


@dataclass(frozen=True)
class CapacitySpec:
    """一次容量口径的参数。"""

    notional: float
    participation: float = PARTICIPATION
    cap: float = MAX_MULTIPLIER
    window: int = ADV_WINDOW
    lag: int = ADV_LAG


@dataclass(frozen=True)
class Capacity:
    """容量约束下的倍率面板与两条诊断读数。"""

    mult: pd.DataFrame
    # 容量本身允许的倍率（未与账本倍率取小）：用来数「哪些标的吃不下多少仓位」
    allowed: pd.DataFrame
    # 有 ADV 读数的格子里被容量压住的比重（受限于池尾，不是受限于账本倍率）
    constrained_share: float
    # 有 ADV 读数的格子的平均倍率（含信号为 0 的日子；nan 由调用方决定怎么显示）
    mean_multiplier: float

    def quota_counts(self, *, cap: float = MAX_MULTIPLIER) -> tuple[int, int]:
        """按标的中位允许倍率数两个数：能拿满 ``cap`` 倍的只数、连 1 倍都吃不下的只数。"""
        median = cast(pd.Series, self.allowed.median()).dropna()
        return int((median >= cap).sum()), int((median < 1.0).sum())


def dollar_volume(bars: pd.DataFrame) -> pd.Series:
    """日名义成交额（收盘价 × 成交量）。"""
    return cast(pd.Series, bars["close"] * bars["volume"])


def average_dollar_volume(
    bars: pd.DataFrame, *, window: int = ADV_WINDOW, lag: int = ADV_LAG
) -> pd.Series:
    """滞后中位日成交额：窗口只含 t−lag 及更早的读数，故无前视。

    用中位而非均值：单日天量成交会把均值抬到不真实的水平，而容量约束关心的是常态。
    """
    return cast(pd.Series, dollar_volume(bars).rolling(window).median().shift(lag))


def capacity_frame(
    bars_map: dict[str, pd.DataFrame],
    counts: pd.Series,
    book: pd.Series,
    spec: CapacitySpec,
) -> Capacity:
    """算逐标的倍率 = min(账本倍率, 允许倍率, 上限)，附两条容量诊断。"""
    allowed: dict[str, pd.Series] = {}
    for stem, bars in bars_map.items():
        adv = average_dollar_volume(bars, window=spec.window, lag=spec.lag)
        members = counts.reindex(adv.index).astype(float)
        allowed[stem] = spec.participation * adv * members / spec.notional
    frame = pd.DataFrame(allowed)
    capped = frame.clip(upper=spec.cap)
    mult = cast(pd.DataFrame, capped.clip(upper=book, axis=0).fillna(0.0))
    known = frame.notna()
    mask = known.to_numpy()
    return Capacity(
        mult=mult,
        allowed=frame,
        constrained_share=_pressed_share(capped, book, known),
        mean_multiplier=float(mult.to_numpy()[mask].mean()) if mask.any() else 0.0,
    )


def _pressed_share(
    allowed: pd.DataFrame, book: pd.Series, known: pd.DataFrame
) -> float:
    """被容量压住的格子比重：分母只数有 ADV 读数的格子。"""
    mask = known.to_numpy()
    row = book.reindex(allowed.index).to_numpy(dtype=float).reshape(-1, 1)
    pressed = (allowed.to_numpy(dtype=float) < row) & mask
    total = int(mask.sum())
    return float(pressed.sum() / total) if total else 0.0


def run_capacity(
    bars_map: dict[str, pd.DataFrame],
    options: TsmomOptions,
    capacity: Capacity,
    *,
    cost: FeeSchedule | None = None,
) -> dict[str, SymbolRun]:
    """按逐标的倍率跑引擎：信号 × min(账本倍率, 允许倍率)，成本按真实仓位计。"""
    out: dict[str, SymbolRun] = {}
    for stem, bars in bars_map.items():
        scale = capacity.mult[stem].reindex(bars.index).fillna(0.0)
        out[stem] = run_symbol(
            bars, multi_speed_signal(bars, options) * scale, cost=cost
        )
    return out
