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

from src.research import ffd
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
# 分数阶差分（de Prado 第 5 章）：阶数与固定窗阈值。阈值 1e-3 对应宽度 72 / 54 根，
# 本仓日线面板中位历史 958 根，196 只里 184~186 只喂得满
FFD_ORDERS: tuple[float, ...] = (0.2, 0.4)
FFD_THRESHOLD: float = 1e-3

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


def momentum(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根的累计收益。"""
    return panel.close / panel.close.shift(window) - 1


def volatility(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根收益的标准差（已实现波动）。"""
    # stub 按 axis 重载把逐元素结果标成 Series，运行时是 DataFrame；用 cast 收窄
    rolling_std = panel.returns().rolling(window, min_periods=window).std()
    return cast(pd.DataFrame, rolling_std)


def log_notional(panel: Panel, window: int) -> pd.DataFrame:
    """过去 window 根中位名义额的对数：规模/流动性代理。"""
    median = panel.notional.rolling(window, min_periods=window).median()
    return cast(pd.DataFrame, np.log(median.where(median > 0)))


def distance_to_high(panel: Panel, window: int) -> pd.DataFrame:
    """距过去 window 根最高收盘价的距离（≤0，越接近 0 越靠近新高）。"""
    highest = panel.close.rolling(window, min_periods=window).max()
    return panel.close / highest - 1


def fractional_difference(panel: Panel, order: float) -> pd.DataFrame:
    """对数收盘价的分数阶差分，减掉水平泄漏项（de Prado 第 5 章）。

    实测（swap 日线）：**原始**变换与对数价格水平的逐期截面相关是 +1.000（d=0.2）/
    +0.997（d=0.4），截面标准差比信号部分大 13~35 倍——直接用就等于拿价格水平当因子。
    这里减去「权重和 × 对数水平」把泄漏项精确抵消（等价于把权重重新标准化成零和），
    得到的因子对「整体乘一个常数」完全不变，才是「偏离自身慢趋势」那个信号。
    """
    weights = ffd.fractional_weights(order, FFD_THRESHOLD)
    # numpy 的桩把 DataFrame 的逐元素结果标成 ndarray，运行时是 DataFrame
    log_close = cast(pd.DataFrame, np.log(panel.close))
    raw = log_close.apply(
        lambda column: ffd.fractional_difference(column, order, FFD_THRESHOLD)
    )
    return cast(pd.DataFrame, raw) - float(weights.sum()) * log_close


# === 因子工厂 ===


def _momentum_spec(key: str, window: int, direction: int, label: str) -> FactorSpec:
    return FactorSpec(
        key=key,
        label=label,
        family=FAMILY_PRICE_VOLUME,
        direction=direction,
        warmup=window,
        compute=lambda panel: momentum(panel, window),
    )


def _volatility_spec(window: int) -> FactorSpec:
    # 收益序列首根为 NaN，满窗口要等到第 window 根（＝warmup 取 window）
    return FactorSpec(
        key=f"vol_{window}",
        label=f"{window} 根已实现波动",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_SHORT,  # 低波动异象：波动越大越看空
        warmup=window,
        compute=lambda panel: volatility(panel, window),
    )


def _liquidity_spec(window: int) -> FactorSpec:
    return FactorSpec(
        key=f"liq_{window}",
        label=f"{window} 日中位名义额（对数）",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_SHORT,  # 流动性/规模溢价：越小越看多
        # 名义额无前置 NaN，满 min_periods 落在下标 window-1
        warmup=window - 1,
        compute=lambda panel: log_notional(panel, window),
    )


def _ffd_spec(order: float) -> FactorSpec:
    """分数阶差分因子。方向按实测 IC：去泄漏后「偏离慢趋势越多越看空」。"""
    return FactorSpec(
        key=f"ffd_d{order:g}".replace(".", "p"),
        label=f"{order:g} 阶分数差分（去水平泄漏）",
        family=FAMILY_PRICE_VOLUME,
        # 实测原始因子 RankIC −0.060（h=1、t −13.1），与 mom_7 的截面秩相关 +0.73，
        # 与反转族同向，故方向取 SHORT
        direction=DIRECTION_SHORT,
        warmup=ffd.window_width(order, FFD_THRESHOLD),
        compute=lambda panel: fractional_difference(panel, order),
    )


def _high_spec(window: int) -> FactorSpec:
    return FactorSpec(
        key=f"dth_{window}",
        label=f"距 {window} 根最高点",
        family=FAMILY_PRICE_VOLUME,
        direction=DIRECTION_LONG,  # 高点邻近效应：越接近新高越看多
        warmup=window - 1,
        compute=lambda panel: distance_to_high(panel, window),
    )


# 主池：第七到第二十五节所有对照脚本用的同一批因子（顺序即报告里的顺序）。
# 两个研究脚本与合成床各用一份，定义只在这里出一次。
MAIN_POOL_KEYS: tuple[str, ...] = (
    "rev_3",
    "mom_7",
    "mom_30",
    "vol_30",
    "liq_30",
    "dth_60",
    "ffd_d0p4",
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
        *(_ffd_spec(order) for order in FFD_ORDERS),
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
