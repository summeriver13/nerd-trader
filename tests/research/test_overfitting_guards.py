"""多重检验校正的退化情形：候选完全相同、样本过短、门槛下界。

历史教训（修复前）：5 个完全相同的候选会同时通过 DSR 与 PBO（离散度≈0 把门槛清零、
平局秩把冠军判成第一名），而 3 个观测的序列把 nan 打印成「未通过」。
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import bars_per_year
from src.research.overfitting import (
    MIN_DSR_OBSERVATIONS,
    build_overfitting_report,
    expected_max_sharpe,
    sharpe_estimator_variance,
)

_INDEX = pd.date_range("2020-01-01", periods=1500, freq="1D", tz="UTC")


def _series(seed: int = 7, periods: int = 1500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.011, periods), index=_INDEX[:periods])


def test_identical_candidates_do_not_pass() -> None:
    """候选完全相同时门槛必须非零、PBO 报未定义，不能判「通过」。"""
    same = {f"候选{i}": _series() for i in range(5)}
    report = build_overfitting_report(same, trials=6)

    assert report.pbo is None
    assert report.sharpe_tests[0].expected_max > 0
    assert not any(test.passed for test in report.sharpe_tests)


def test_estimator_variance_floors_the_threshold() -> None:
    """门槛取「候选间离散度」与「估计量方差 (1+0.5·SR²)/T」的较大者。"""
    identical = sharpe_estimator_variance([0.05, 0.05], 1500)
    assert identical == pytest.approx((1 + 0.5 * 0.05**2) / 1500)

    spread = sharpe_estimator_variance([0.01, 0.20], 1500)
    assert spread > identical  # 离散度更大时用它
    assert expected_max_sharpe(identical, 6) > 0


def test_short_series_reports_insufficient_not_failed() -> None:
    """观测数不足时报「样本不足」（DSR = nan），不等于「未通过」。"""
    short = {f"短{i}": _series(seed=i, periods=3) for i in range(3)}
    report = build_overfitting_report(short, trials=6)

    assert report.observations == 3
    assert all(not test.sufficient for test in report.sharpe_tests)
    assert all(np.isnan(test.dsr) for test in report.sharpe_tests)
    assert MIN_DSR_OBSERVATIONS == 30


def test_bars_per_year_uses_seconds_not_truncated_days() -> None:
    """跨度按秒算：1h 下 500 根用整数天会偏高约 4%，进而让窗口根数偏大。"""
    index = pd.date_range("2026-01-01", periods=500, freq="1h", tz="UTC")
    true_span_days = (index[-1] - index[0]).total_seconds() / 86_400
    truncated_days = (index[-1] - index[0]).days

    assert truncated_days < true_span_days
    assert bars_per_year(index) == pytest.approx(len(index) / (true_span_days / 365))
    assert bars_per_year(index) < len(index) / (truncated_days / 365)
