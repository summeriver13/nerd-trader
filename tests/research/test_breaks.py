"""SADF 的测试：正确性锚在与 `statsmodels.adfuller` 的对拍上。"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.research.breaks import (
    MIN_ROWS,
    AdfSweep,
    SadfOptions,
    adf_tstat,
    explosive_mask,
    monte_carlo_critical,
    sadf_series,
)

SEED = 20261012


def random_walk(size: int) -> np.ndarray:
    """固定 seed 的随机游走。"""
    rng = np.random.default_rng(SEED)
    return np.cumsum(rng.standard_normal(size))


def explosive(size: int, drift: float = 1.02) -> np.ndarray:
    """指数发散的序列：SADF 应当很高。"""
    rng = np.random.default_rng(SEED)
    return drift ** np.arange(size) * (1.0 + 0.01 * rng.standard_normal(size))


def reference_tstat(levels: np.ndarray, lags: int) -> float:
    """statsmodels 的 ADF t 值（对拍基准）。

    `autolag` 在 stub 里被标成 `str`，但运行时传 `None` 才是「不做滞后选择、就用
    maxlag」——这正是要与自算实现对齐的口径。用 cast 过桥，不用 ignore 硬压。

    `result_object=False` 是 statsmodels 自己给的出路：不传会发 FutureWarning，
    而本仓库的 `filterwarnings=["error"]` 会把它直接变成测试失败。
    """
    result = adfuller(
        levels,
        maxlag=lags,
        regression="c",
        autolag=cast(str, None),
        result_object=False,
    )
    return float(result[0])


@pytest.mark.parametrize("lags", [0, 1, 3])
@pytest.mark.parametrize("kind", ["walk", "explosive"])
def test_adf_tstat_matches_statsmodels(kind: str, lags: int) -> None:
    """自算的 ADF t 值必须与 statsmodels 同口径——这是整个模块的正确性锚。"""
    levels = random_walk(400) if kind == "walk" else explosive(400)
    assert adf_tstat(levels, lags) == pytest.approx(
        reference_tstat(levels, lags), abs=1e-8
    )


def test_adf_tstat_is_negative_on_stationary_series() -> None:
    """平稳序列应当给出强负值（左尾拒绝单位根）。"""
    rng = np.random.default_rng(SEED)
    values = np.zeros(400)
    for index in range(1, values.size):
        values[index] = 0.5 * values[index - 1] + rng.standard_normal()
    assert adf_tstat(values, 1) < -5.0


def test_single_window_and_batched_paths_agree() -> None:
    """`tstat`（单窗口）与 `sup_tstat`（批量）是两条代码路径，必须给出同一答案。

    批量路径是 SADF 的热循环（快 79 倍），一旦与单窗口分叉，测试的对拍就失去了
    覆盖它的一半意义。
    """
    levels = explosive(300)
    sweep = AdfSweep(levels, 1)
    for start in (0, 5, 20):
        assert sweep.tstat(start, levels.size - 1) == pytest.approx(
            sweep.sup_tstat(levels.size - 1, np.array([start])), abs=1e-10
        )


def test_prefix_sum_equals_direct_regression() -> None:
    """前缀和实现的整段 t 值，与直接建矩阵跑 OLS 的结果一致。"""
    levels = random_walk(250)
    sweep = AdfSweep(levels, 1)
    direct = adf_tstat(levels, 1)
    assert sweep.tstat(0, levels.size - 1) == pytest.approx(direct, abs=1e-10)


def test_sadf_separates_explosive_from_random_walk() -> None:
    """方向性：爆炸序列的 SADF 必须远高于随机游走。"""
    options = SadfOptions(min_window=60, lags=1, step=5)
    walk_peak = float(np.nanmax(sadf_series(pd.Series(random_walk(400)), options)))
    boom_peak = float(np.nanmax(sadf_series(pd.Series(explosive(400)), options)))
    assert boom_peak > walk_peak + 10.0


def test_sadf_is_undefined_before_the_window() -> None:
    """窗口之前没有历史，必须是 NaN——这是定义域，不是缺陷。"""
    options = SadfOptions(min_window=60, lags=1, step=5)
    series = sadf_series(pd.Series(random_walk(200)), options)
    assert series.iloc[:60].isna().all()
    assert series.iloc[60:].notna().all()


def test_short_window_returns_nan_instead_of_raising() -> None:
    """样本不足时返回 NaN：SADF 在每个端点上都会遇到短窗口，不能抛异常。"""
    assert np.isnan(adf_tstat(random_walk(MIN_ROWS - 1), 1))


def test_singular_window_returns_nan() -> None:
    """局部完全无波动的窗口会让差分列退化，必须报 NaN 而不是崩掉或给假值。"""
    flat = np.concatenate([np.zeros(120), random_walk(60)])
    sweep = AdfSweep(flat, 1)
    assert np.isnan(sweep.tstat(0, 119))


def test_monte_carlo_critical_exceeds_typical_walk_peak() -> None:
    """临界值来自随机游走的分位，必须高于同长度随机游走的典型峰值。"""
    options = SadfOptions(min_window=60, lags=1, step=5)
    critical = monte_carlo_critical(200, options, replications=40)
    typical = float(np.nanmax(sadf_series(pd.Series(random_walk(200)), options)))
    assert critical > typical


def test_explosive_mask_is_strict() -> None:
    """掩码是严格大于：恰好等于临界值不算爆炸。"""
    sadf = pd.Series([1.0, 2.0, 3.0])
    assert explosive_mask(sadf, 2.0).tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("field", "value"),
    [("min_window", MIN_ROWS - 1), ("lags", -1), ("step", 0)],
)
def test_invalid_options_are_rejected(field: str, value: int) -> None:
    """参数自检：窗口太短、负滞后、零步长都必须报错而不是静默出结果。"""
    with pytest.raises(ValueError):
        SadfOptions(**{field: value}).validate()
