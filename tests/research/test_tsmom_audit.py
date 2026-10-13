"""收尾体检模块的不变量：bootstrap 区间、数据时效与逐年宽度。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.tsmom_audit import (
    ALIVE_WINDOW_DAYS,
    alive_at_end,
    bootstrap_sharpe,
    bootstrap_sharpe_diff,
    breadth_by_year,
    coverage_table,
)

REPLICATES = 200
SEED = 20261014


def make_bars(periods: int, *, end: str, freq: str = "D") -> pd.DataFrame:
    """构造日线：只需要 close 列与索引。"""
    index = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, periods)}, index=index)


def test_bootstrap_sharpe_interval_covers_the_estimate() -> None:
    """点估计落在区间内，且重采样次数与块长如实回填。"""
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    returns = pd.Series(rng.normal(0.001, 0.02, 800), index=index)
    row = bootstrap_sharpe(
        returns, label="x", block=10, replicates=REPLICATES, seed=SEED
    )
    assert row.low <= row.sharpe <= row.high
    assert row.block == 10
    assert row.replicates == REPLICATES


def test_bootstrap_sharpe_of_zero_mean_series_straddles_zero() -> None:
    """均值恰为 0 的序列：区间必须含 0，且非正比例接近一半。"""
    pattern = np.tile([0.02, -0.02, 0.01, -0.01], 250)  # 均值恰为 0
    index = pd.date_range("2024-01-01", periods=pattern.size, freq="D", tz="UTC")
    returns = pd.Series(pattern, index=index)
    row = bootstrap_sharpe(returns, label="noise", replicates=REPLICATES, seed=SEED)
    assert row.low < 0.0 < row.high
    assert 0.2 < row.share_nonpositive < 0.8


def test_bootstrap_sharpe_diff_is_zero_for_identical_series() -> None:
    """配对差：同一条序列与自己的差恒为 0（区间退化到 0）。"""
    rng = np.random.default_rng(11)
    index = pd.date_range("2024-01-01", periods=600, freq="D", tz="UTC")
    returns = pd.Series(rng.normal(0.001, 0.02, 600), index=index)
    row = bootstrap_sharpe_diff(
        returns, returns, label="self", replicates=REPLICATES, seed=SEED
    )
    assert row.sharpe == pytest.approx(0.0)
    assert row.low == pytest.approx(0.0)
    assert row.high == pytest.approx(0.0)


def test_bootstrap_sharpe_diff_aligns_on_common_dates() -> None:
    """配对差只用两条序列共同有值的日期（缺值不参与重采样）。"""
    rng = np.random.default_rng(3)
    index = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    left = pd.Series(rng.normal(0.002, 0.02, 400), index=index)
    right = pd.Series(rng.normal(0.000, 0.02, 400), index=index)
    right.iloc[:200] = np.nan
    row = bootstrap_sharpe_diff(left, right, label="pair", replicates=REPLICATES)
    assert row.replicates == REPLICATES
    assert np.isfinite(row.sharpe)


def test_alive_at_end_splits_by_last_bar() -> None:
    """末端判定：最后行情距面板末端超过窗口的算退市。"""
    bars_map = {
        "alive": make_bars(300, end="2026-10-10"),
        "dead": make_bars(300, end="2025-01-01"),
    }
    alive = alive_at_end(bars_map)
    assert alive == ["alive"]
    assert ALIVE_WINDOW_DAYS == 30


def test_coverage_table_reports_first_and_last() -> None:
    """覆盖表：逐标的的根数与首末日期。"""
    bars_map = {
        "a": make_bars(120, end="2026-01-01"),
        "b": make_bars(60, end="2026-02-01"),
    }
    table = coverage_table(bars_map)
    assert set(table["stem"]) == {"a", "b"}
    assert int(table.loc[table["stem"] == "a", "bars"].iloc[0]) == 120
    assert table["last"].max() == pd.Timestamp("2026-02-01", tz="UTC")


def test_breadth_by_year_uses_median_members() -> None:
    """逐年宽度：按年取参与组合标的数的中位。"""
    index = pd.date_range("2024-12-25", periods=10, freq="D", tz="UTC")
    streams = {
        "a": pd.Series(0.0, index=index),
        "b": pd.Series(0.0, index=index[:5]),
    }
    breadth = breadth_by_year(streams)
    assert float(breadth.loc[2024]) == 2.0
    assert float(breadth.loc[2025]) == 1.0
