"""测试共用的合成面板构造器：数值断言必须建立在可手算的数据上。

真实缓存会随采集推进而变化（新标的、新根数），拿它做数值断言等于把测试绑在数据上；
需要断言具体数字时用这里的合成面板，需要验证口径规则时才用临时 parquet 缓存。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel

type PanelFactory = Callable[[dict[str, list[float]]], Panel]
type SpecFactory = Callable[[bool], FactorSpec]
type PanelCodes = tuple[np.ndarray, np.ndarray]

DATE_PANEL_DAYS: int = 40
DATE_PANEL_PER_DAY: int = 3

DAILY_NOTIONAL: float = 1e6
# 手算合成数据的口径：第 i 个标的每根收益 = STEP × i，分位与 IC 都能一眼算出
NAMES: int = 20
PERIODS: int = 4
STEP: float = 0.01


@pytest.fixture
def panel_factory() -> PanelFactory:
    """把 {标的: 收盘价序列} 变成面板；名义额固定（默认 100 万美元/根）。"""

    def _make(closes: dict[str, list[float]]) -> Panel:
        length = len(next(iter(closes.values())))
        index = pd.date_range("2020-01-01", periods=length, freq="1D", tz="UTC")
        close = pd.DataFrame(closes, index=index)
        volume = pd.DataFrame(DAILY_NOTIONAL, index=index, columns=close.columns)
        return Panel(
            timeframe="1d",
            close=close,
            volume=volume,
            notional=close * volume,
        )

    return _make


@pytest.fixture
def coded_panel() -> PanelCodes:
    """(日期序号, 特征)：同一天内特征相同、天与天之间递增——第 9 章的折切分用它。"""
    codes = np.repeat(np.arange(DATE_PANEL_DAYS), DATE_PANEL_PER_DAY)
    features = np.column_stack([codes.astype(float), np.ones(codes.size)])
    return codes, features


@pytest.fixture
def ranked_panel(panel_factory: PanelFactory) -> tuple[Panel, list[float]]:
    """确定性行情：第 i 个标的每根收益 = STEP × i，返回面板与收益率列表。"""
    rates = [STEP * index for index in range(NAMES)]
    closes = {
        f"N{index:02d}-USDT": [100.0 * (1 + rate) ** bar for bar in range(PERIODS)]
        for index, rate in enumerate(rates)
    }
    return panel_factory(closes), rates


@pytest.fixture
def rank_spec() -> SpecFactory:
    """因子值 = 标的下标（越大越看多）的工厂；reverse=True 验证符号方向。"""

    def _make(reverse: bool = False) -> FactorSpec:
        sign = -1.0 if reverse else 1.0

        def compute(panel: Panel) -> pd.DataFrame:
            return pd.DataFrame(
                {name: float(index) * sign for index, name in enumerate(panel.symbols)},
                index=panel.index,
            )

        return FactorSpec(
            key="rank",
            label="手算排名因子",
            family="测试",
            direction=1,
            warmup=0,
            compute=compute,
        )

    return _make
