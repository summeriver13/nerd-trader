"""趋势扫描标签（de Prado Ch 3.6）的用例。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import labels


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def _ramp(count: int, rate: float = 0.01, start: float = 100.0) -> pd.Series:
    """单调等比序列：每个窗口的斜率同号，但 |t| 随窗口长度变化。"""
    return _series([start * (1.0 + rate) ** k for k in range(count)])


def _strong_then_noise(count: int = 60) -> list[float]:
    """先一段强趋势、再一段原地噪声：噪声段的 |t| 必然远低于历史分位。"""
    strong = [100.0 * 1.02**k for k in range(count)]
    noise = [
        strong[-1] * (1.0 + (0.0005 if k % 2 == 0 else -0.0005)) for k in range(count)
    ]
    return strong + noise


def _options(**overrides: float | int) -> labels.TrendScanOptions:
    base: dict[str, float | int] = {"span": 10}
    base.update(overrides)
    return labels.TrendScanOptions(**base)  # type: ignore[arg-type]


def test_validate_rejects_degenerate_options() -> None:
    for bad in (
        {"min_window": 2},
        {"span": 2},
        {"threshold": 1.0},
        {"threshold": -0.1},
    ):
        with pytest.raises(ValueError):
            _options(**bad).validate()


def test_short_series_keeps_the_columns() -> None:
    frame = labels.trend_scan(_series([100.0] * 5), _options(span=10))
    assert list(frame.columns) == list(labels.TREND_COLUMNS)
    assert frame["label"].isna().to_numpy().all()


def test_monotone_series_keeps_its_sign_and_finite_t() -> None:
    """单调序列的每个窗口斜率同号：标签恒 ±1，t 值大而有限（不发散）。"""
    up = labels.trend_scan(_ramp(40), _options())
    valid = up["label"].notna()
    assert (up.loc[valid, "label"] == 1).all()
    assert (up.loc[valid, "span"] >= 3).all()
    assert np.isfinite(up.loc[valid, "t_value"]).all()

    down = labels.trend_scan(_ramp(40, rate=-0.01), _options())
    assert (down.loc[down["label"].notna(), "label"] == -1).all()


def test_smooth_series_picks_the_shortest_window() -> None:
    """等比序列上最短窗口胜出——这是 |t| 的性质，不是 bug：短窗口几乎完美拟合。

    钉住它是为了记住趋势扫描的 `span` 分布会偏向窗口下限：想让它看长趋势，
    要么拉长 `min_window`，要么用 `threshold` 把弱趋势判掉。
    """
    frame = labels.trend_scan(_ramp(40), _options())
    assert set(frame["span"].dropna().unique()) == {3.0}


def test_slope_t_matches_ols_residuals() -> None:
    """斜率 t 值必须等于「斜率 ÷ 其标准误」——用残差独立算一遍，不引用被测量的实现。"""
    windows = np.array(
        [
            [100.0, 101.0, 100.5, 103.0, 102.0, 105.0],
            [5.0, 4.0, 4.5, 3.0, 3.2, 2.0],
        ]
    )
    ramp = np.arange(windows.shape[1], dtype=float)
    centered = ramp - ramp.mean()
    sxx = float(centered @ centered)
    t_values = labels.slope_t_values(windows)
    for row, window in enumerate(windows):
        slope = float(centered @ (window - window.mean())) / sxx
        residual = window - (window.mean() + slope * centered)
        standard_error = np.sqrt(float(residual @ residual) / (window.size - 2) / sxx)
        assert t_values[row] == pytest.approx(slope / standard_error)


def test_constant_series_has_no_label() -> None:
    """价格没动就没有趋势：方差为 0 的窗口不许当「|t| = 0 的平趋势」参与取最大。"""
    frame = labels.trend_scan(_series([100.0] * 40), _options())
    assert frame["label"].isna().to_numpy().all()


def test_tail_rows_are_nan_not_zero() -> None:
    """尾部凑不满最小回归窗口：返回 NaN，不能返回 0（0 在下游是「有观点不下注」）。"""
    frame = labels.trend_scan(_ramp(40), _options(min_window=3))
    assert frame["label"].iloc[-2:].isna().to_numpy().all()
    assert frame["label"].iloc[:-2].notna().to_numpy().all()


def test_t1_span_and_ret_agree_with_the_price_series() -> None:
    close = _ramp(40, rate=0.013)
    frame = labels.trend_scan(close, _options())
    positions = close.index.get_indexer(frame.index)
    valid = frame["label"].notna().to_numpy()
    endpoints = frame["t1"].to_numpy(dtype=float)[valid]
    spans = frame["span"].to_numpy(dtype=float)[valid]
    assert np.allclose(endpoints, positions[valid] + spans - 1.0)
    values = close.to_numpy(dtype=float)
    expected = values[endpoints.astype(int)] / values[positions[valid]] - 1.0
    assert np.allclose(frame["ret"].to_numpy(dtype=float)[valid], expected)


def test_threshold_zeroes_the_weak_trends() -> None:
    close = _series(_strong_then_noise())
    plain = labels.trend_scan(close, _options(span=10))
    cut = labels.trend_scan(close, _options(span=10, threshold=0.5))
    assert (cut["label"] == 0).sum() > (plain["label"] == 0).sum()
    noise_rows = cut["label"].iloc[70:]
    assert (noise_rows == 0).mean() > 0.5


def test_threshold_never_looks_forward() -> None:
    """阈值只看过去：把未来接上去，前面每一根的标签都不许变。"""
    close = _series(_strong_then_noise())
    prefix = labels.trend_scan(close.iloc[:80], _options(span=10, threshold=0.5))
    full = labels.trend_scan(close, _options(span=10, threshold=0.5))
    assert np.allclose(
        prefix["label"].iloc[:70].to_numpy(dtype=float),
        full["label"].iloc[:70].to_numpy(dtype=float),
        equal_nan=True,
    )
