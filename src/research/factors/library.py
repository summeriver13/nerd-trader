"""
src/research/factors/library.py
量价因子库：每个因子的定义（窗口、方向、预热）在这里唯一出一次。

三条约定：
1. **方向**：`direction=+1` 表示因子值越大越看多。报告与组合统一消费
   `FactorSpec.values()`（已按方向调整），因此 IC 为正就代表因子有效，
   不会出现「有的因子负号是好事」的读表歧义。
2. **预热**：`warmup` 是第一个有效值的下标（前 warmup 根必为 NaN），
   调用方据此对齐评估窗起点；每个 spec 的 warmup 都写在测试里钉死。
3. **只吃历史**：因子在第 t 根只能用第 t 根及以前的数据。这条由
   tests/research/test_factor_panel.py 的不变量测试钉住（篡改未来不改变过去）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.research.factors.panel import Panel

# 因子族（报告按族分组，便于看「同一族里的因子是否互相重复」）
FAMILY_PRICE_VOLUME: str = "量价"

# === 因子窗口（根）===
MOMENTUM_SHORT: int = 7
MOMENTUM_LONG: int = 30
REVERSAL_WINDOW: int = 3
VOLATILITY_WINDOW: int = 30
LIQUIDITY_WINDOW: int = 30
HIGH_WINDOW: int = 60

DIRECTION_LONG: int = 1  # 值越大越看多
DIRECTION_SHORT: int = -1  # 值越小越看多


@dataclass(frozen=True)
class FactorSpec:
    """一个因子的完整定义。"""

    key: str
    label: str
    family: str
    direction: int
    # 第一个有效值的下标：前 warmup 根必为 NaN
    warmup: int
    compute: Callable[[Panel], pd.DataFrame]

    def values(self, panel: Panel) -> pd.DataFrame:
        """方向调整后的因子值：数值越大 = 越看多。"""
        return self.compute(panel) * self.direction

    def describe(self) -> str:
        """报告里的一行说明。"""
        arrow = "越大越好" if self.direction == DIRECTION_LONG else "越小越好"
        return f"{self.label}（{arrow}，预热 {self.warmup} 根）"


# === 因子计算（只吃历史：全部用 shift/rolling 的向后看窗口）===


def _momentum(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根的累计收益。"""
    return panel.close / panel.close.shift(window) - 1


def _volatility(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根收益的标准差（已实现波动）。"""
    # stub 按 axis 重载把逐元素结果标成 Series，运行时是 DataFrame；用 cast 收窄
    rolling_std = panel.returns().rolling(window, min_periods=window).std()
    return cast(pd.DataFrame, rolling_std)


def _log_notional(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根中位名义额的对数：规模/流动性代理。"""
    median = panel.notional.rolling(window, min_periods=window).median()
    return cast(pd.DataFrame, np.log(median.where(median > 0)))


def _distance_to_high(panel: Panel, window: int) -> pd.DataFrame:
    """距过去 window 根最高收盘价的距离（≤0，越接近 0 越靠近新高）。"""
    highest = panel.close.rolling(window, min_periods=window).max()
    return panel.close / highest - 1


# === 因子工厂 ===


def _momentum_spec(key: str, window: int, direction: int, label: str) -> FactorSpec:
    return FactorSpec(
        key=key,
        label=label,
        family=FAMILY_PRICE_VOLUME,
        direction=direction,
        warmup=window,
        compute=lambda panel: _momentum(panel, window),
    )


def _volatility_spec(window: int) -> FactorSpec:
    # 收益序列首根为 NaN，满窗口要等到第 window 根（＝warmup 取 window）
    return FactorSpec(
        key=f"vol_{window}",
        label=f"{window} 根已实现波动",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_SHORT,  # 低波动异象：波动越大越看空
        warmup=window,
        compute=lambda panel: _volatility(panel, window),
    )


def _liquidity_spec(window: int) -> FactorSpec:
    return FactorSpec(
        key=f"liq_{window}",
        label=f"{window} 日中位名义额（对数）",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_SHORT,  # 流动性/规模溢价：越小越看多
        # 名义额无前置 NaN，满 min_periods 落在下标 window-1
        warmup=window - 1,
        compute=lambda panel: _log_notional(panel, window),
    )


def _high_spec(window: int) -> FactorSpec:
    return FactorSpec(
        key=f"dth_{window}",
        label=f"距 {window} 根最高点",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_LONG,  # 高点邻近效应：越接近新高越看多
        warmup=window - 1,
        compute=lambda panel: _distance_to_high(panel, window),
    )


FACTOR_LIBRARY: dict[str, FactorSpec] = {
    spec.key: spec
    for spec in (
        _momentum_spec(
            f"mom_{MOMENTUM_SHORT}",
            MOMENTUM_SHORT,
            DIRECTION_LONG,
            f"{MOMENTUM_SHORT} 根动量",
        ),
        _momentum_spec(
            f"mom_{MOMENTUM_LONG}",
            MOMENTUM_LONG,
            DIRECTION_LONG,
            f"{MOMENTUM_LONG} 根动量",
        ),
        _momentum_spec(
            f"rev_{REVERSAL_WINDOW}",
            REVERSAL_WINDOW,
            DIRECTION_SHORT,
            f"{REVERSAL_WINDOW} 根反转",
        ),
        _volatility_spec(VOLATILITY_WINDOW),
        _liquidity_spec(LIQUIDITY_WINDOW),
        _high_spec(HIGH_WINDOW),
    )
}


def resolve_factors(keys: str | None = None) -> list[FactorSpec]:
    """把逗号分隔的因子键解析成 spec 列表；省略则取全库。未知键当场报错。"""
    if not keys:
        return list(FACTOR_LIBRARY.values())
    unknown = [
        key
        for key in (item.strip() for item in keys.split(","))
        if key not in FACTOR_LIBRARY
    ]
    if unknown:
        raise KeyError(
            f"未知因子 {', '.join(unknown)}；可用：{', '.join(FACTOR_LIBRARY)}"
        )
    return [FACTOR_LIBRARY[key.strip()] for key in keys.split(",")]
