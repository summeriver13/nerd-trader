"""测试共用的合成面板构造器：数值断言必须建立在可手算的数据上。

真实缓存会随采集推进而变化（新标的、新根数），拿它做数值断言等于把测试绑在数据上；
需要断言具体数字时用这里的合成面板，需要验证口径规则时才用临时 parquet 缓存。
"""

from collections.abc import Callable

import pandas as pd
import pytest

from src.research.factors.panel import Panel

type PanelFactory = Callable[[dict[str, list[float]]], Panel]

DAILY_NOTIONAL: float = 1e6


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
