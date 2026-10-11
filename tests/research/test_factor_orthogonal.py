"""
tests/research/test_factor_orthogonal.py
正交化检验的口径测试：只测会「静默算错」的地方——残差是否真的把控制变量剔干净、
分档是否等宽有序、完美代理是否真的剩不下东西。

数值断言全部用手算得出来的合成面板（秩空间里的构造），真实缓存只在集成用例里
验口径规则（行数、列名、衰减定义）。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.research.factors.evaluation import (
    MIN_NAMES_PER_PERIOD,
    residual_ic_series,
    spearman,
)
from src.research.factors.library import DIRECTION_LONG, FactorSpec
from src.research.factors.orthogonal import (
    ORTHOGONAL_FACTORS,
    ORTHOGONAL_PAIRS,
    _bucket_rows,
    double_sort_frame,
    orthogonal_frame,
    orthogonal_ic,
)
from src.research.factors.panel import Panel
from src.research.factors.portfolio import MIN_NAMES_PER_QUANTILE

START = "2022-01-01"
NAMES = 20


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    """按列构造宽表（每个标的等长）。"""
    index = pd.date_range(
        START, periods=len(next(iter(values.values()))), freq="1D", tz="UTC"
    )
    return pd.DataFrame(values, index=index)


def _panel(bars: int = 60, symbols: int = 30, seed: int = 4) -> Panel:
    """合成面板：波动率与名义额按标的拉开档次，上市起点逐个错开。

    起点必须错开：都从第一根开始时上市天数在截面上是个常量，`age_days` 会退化成
    零方差因子，测不到它参与正交化的那条路径。
    """
    rng = np.random.default_rng(seed)
    close: dict[str, list[float]] = {}
    notional: dict[str, list[float]] = {}
    for i in range(symbols):
        sigma = 0.02 + 0.003 * i
        steps = rng.normal(0.0, sigma, bars - i)
        path = np.concatenate([np.full(i, np.nan), np.exp(np.cumsum(steps))])
        close[f"S{i:02d}-USDT"] = path.tolist()
        # 高波动 ↔ 低名义额：与真实横截面同向，控制变量的分档才有意义
        notional[f"S{i:02d}-USDT"] = [
            float("nan") if np.isnan(value) else float(np.exp(14.0 - 30.0 * sigma))
            for value in path
        ]
    return Panel(
        timeframe="1d",
        close=_frame(close),
        volume=_frame(close) * 0.0 + 1.0,
        notional=_frame(notional),
    )


def _mask(panel: Panel) -> pd.DataFrame:
    """全可交易掩码：合成面板不跑池子规则，正交化只看因子本身。"""
    return pd.DataFrame(True, index=panel.index, columns=panel.symbols)


def _rank_frame(values: np.ndarray, bars: int = NAMES) -> pd.DataFrame:
    """把一维数值铺成 bars 行同样的截面（单期测试用）。"""
    index = pd.date_range(START, periods=bars, freq="1D", tz="UTC")
    return pd.DataFrame(
        {f"S{i:02d}": [float(values[i])] for i in range(values.size)}, index=index
    )


def _flat_spec(key: str, value: float) -> FactorSpec:
    """常量因子：用来验证「控制变量毫无信息时不该误伤目标」。"""

    def compute(panel: Panel) -> pd.DataFrame:
        return cast(pd.DataFrame, panel.close * 0.0 + value)

    return FactorSpec(
        key=key,
        label=key,
        family="测试",
        direction=DIRECTION_LONG,
        warmup=0,
        compute=compute,
    )


# === 秩相关与残差：手算得出来的构造 ===


def test_spearman_matches_pandas_reference() -> None:
    """自算的秩相关必须与 pandas 的 spearman 一致（不为一次相关引入 scipy 的前提）。"""
    left = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    right = np.array([2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 3.0])
    expected = float(pd.Series(left).corr(pd.Series(right), method="spearman"))
    assert spearman(left, right) == pytest.approx(expected)


def test_residual_ic_recovers_the_orthogonal_component() -> None:
    """残差必须等于「目标里与控制正交的那一块」。

    构造：控制秩 x 取对称的 1..20，目标秩 y = x + z，z 取「对称」的 ±1（与 x 正交：
    x 反号对称而 z 正号对称，乘积逐对相消）。此时回归斜率恰为 1、残差恰为 z，
    前瞻收益也取 z，所以残差 IC 恰好是 1。
    """
    x = np.arange(NAMES, dtype=float) - (NAMES - 1) / 2.0
    z = np.concatenate([np.ones(5), -np.ones(10), np.ones(5)])
    index = pd.date_range(START, periods=2, freq="1D", tz="UTC")
    columns = pd.Index([f"S{i:02d}" for i in range(NAMES)])
    target_rank = pd.DataFrame([x + z, x + z], index=index, columns=columns)
    control_rank = pd.DataFrame([x, x], index=index, columns=columns)
    forward = pd.DataFrame([z, z], index=index, columns=columns)
    series = residual_ic_series(target_rank, control_rank, forward)
    assert list(series) == pytest.approx([1.0, 1.0])


def test_residual_ic_skips_narrow_periods() -> None:
    """截面太窄的期次直接跳过：少于此数算出来的秩相关没有意义。"""
    narrow = MIN_NAMES_PER_PERIOD - 1
    values = np.arange(narrow, dtype=float)
    frame = _rank_frame(values, bars=2)
    series = residual_ic_series(frame, frame * 1.0, frame * 2.0)
    assert series.empty


def test_residual_ic_treats_constant_control_as_no_information() -> None:
    """控制变量在截面上无差异 = 不带信息：斜率取 0，残差就是去均值的目标。

    去均值是单调变换、秩不变，所以前瞻收益取目标本身时残差 IC 恰好是 1——这条钉住
    「零方差控制不丢期」的语义（丢期会让退化控制变量静默地改变样本）。
    """
    x = np.arange(NAMES, dtype=float)
    index = pd.date_range(START, periods=1, freq="1D", tz="UTC")
    columns = pd.Index([f"S{i:02d}" for i in range(NAMES)])
    target = pd.DataFrame([x], index=index, columns=columns)
    control = pd.DataFrame([[5.0] * NAMES], index=index, columns=columns)
    series = residual_ic_series(target, control, target)
    assert list(series) == pytest.approx([1.0])


# === 分档：等宽、有序、太窄就丢 ===


def test_bucket_rows_split_evenly_and_ordered() -> None:
    """20 只分 5 档 = 每档 4 只；第 1 档是最小值那一档，中位可手算。"""
    control = np.arange(1.0, NAMES + 1.0)
    target = control * 1.0
    ahead = control * 1.0
    row = _bucket_rows(target, control, ahead, 5)
    assert row is not None
    ic_row, median_row, count_row = row
    assert count_row == {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0}
    assert median_row[1] == pytest.approx(2.5)
    assert median_row[5] == pytest.approx(18.5)
    assert all(value == pytest.approx(1.0) for value in ic_row.values())


def test_bucket_rows_reject_period_with_narrow_bucket() -> None:
    """装不满一档的期次整期丢弃：只报装得下的档会让各档口径不一致。"""
    size = 5 * MIN_NAMES_PER_QUANTILE - 1
    values = np.arange(size, dtype=float)
    assert _bucket_rows(values, values, values, 5) is None


# === 正交化：完美代理必须剩不下东西 ===


def test_perfect_proxy_leaves_nothing() -> None:
    """目标因子是控制因子的单调变换时，残差恒为 0 → 残差 IC 为空、保留比例为 0。"""
    panel = _panel()
    mask = _mask(panel)
    control = FactorSpec(
        key="close",
        label="收盘价",
        family="测试",
        direction=DIRECTION_LONG,
        warmup=0,
        compute=lambda panel: panel.close,
    )
    target = FactorSpec(
        key="close_x2",
        label="收盘价 ×2",
        family="测试",
        direction=DIRECTION_LONG,
        warmup=0,
        compute=lambda panel: panel.close * 2.0,
    )
    result = orthogonal_ic(target, control, panel, mask, 1)
    assert result.raw_ic == pytest.approx(result.control_ic)
    assert result.periods == 0
    assert result.residual_ic == pytest.approx(0.0)
    assert result.decay == pytest.approx(0.0)


def test_constant_control_does_not_change_target() -> None:
    """控制变量毫无信息（截面恒定）时不该误伤目标：残差 IC 恰好等于原始 IC。

    零方差控制取斜率 0，残差就是去均值的目标秩，而秩不变——所以两者是同一个数，
    不是「量级接近」。
    """
    panel = _panel()
    mask = _mask(panel)
    flat = _flat_spec("flat", 1.0)
    target = ORTHOGONAL_FACTORS["vol_30"]
    result = orthogonal_ic(target, flat, panel, mask, 1)
    assert result.periods > 0
    assert result.residual_ic == pytest.approx(result.raw_ic, rel=1e-9)


# === 表格：行数、列名、衰减定义 ===


def test_orthogonal_frame_shape_and_decay() -> None:
    """正交化表：每个 (目标, 控制, horizon) 一行，衰减 = 残差 IC / 原始 IC。"""
    panel = _panel()
    frame = orthogonal_frame(panel, _mask(panel), (1, 7))
    assert len(frame) == len(ORTHOGONAL_PAIRS) * 2
    assert set(frame["target"]) == {pair[0] for pair in ORTHOGONAL_PAIRS}
    assert {"raw_ic", "residual_ic", "decay", "residual_t"} <= set(frame.columns)
    for _, row in frame.iterrows():
        if row["raw_ic"] == 0.0:
            # 原始 IC 为 0 时比值无定义（退化因子的口径），不能拿 0 当分母
            assert np.isnan(row["decay"])
            continue
        assert row["decay"] == pytest.approx(row["residual_ic"] / row["raw_ic"])


def test_double_sort_frame_covers_every_bucket() -> None:
    """双重排序表：每个 (目标, 控制, horizon) 恰好 5 行，档号连续。"""
    panel = _panel()
    frame = double_sort_frame(panel, _mask(panel), (1,))
    assert len(frame) == len(ORTHOGONAL_PAIRS) * 5
    for key, group in frame.groupby(["target", "control"]):
        target, control = cast(tuple[str, str], key)
        assert sorted(group["bucket"].tolist()) == [1, 2, 3, 4, 5], (target, control)
        assert (group["periods"] > 0).all()
        assert (group["names"] >= MIN_NAMES_PER_QUANTILE).all()
