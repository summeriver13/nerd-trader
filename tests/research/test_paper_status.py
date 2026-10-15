"""退役判据的边界测试：合成净值面板，只验判决与阈值，不碰真实账本。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.paper_status import (
    BARS_PER_YEAR,
    DD_KILL,
    DD_WATCH,
    EXPOSURE_FLOOR,
    MIN_BARS,
    SHARPE_WATCH,
    VERDICT_ACCUMULATING,
    VERDICT_HEALTHY,
    VERDICT_KILL,
    VERDICT_WATCH,
    retire_status,
    rolling_sharpe,
    status_report,
)


def make_panel(net: np.ndarray, exposure: float = 0.10) -> pd.DataFrame:
    """由逐根净收益造账本面板（首行净值 1.0，与 ledger_panel 同形）。"""
    nav = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    index = pd.date_range("2026-01-01", periods=len(nav), freq="D")
    return pd.DataFrame(
        {
            "nav": nav,
            "btc_nav": np.ones(len(nav)),
            "exposure": np.full(len(nav), exposure),
            "turnover": np.zeros(len(nav)),
        },
        index=index,
    )


def test_short_record_only_accumulates() -> None:
    """未满一年不判定——哪怕这一年天天亏。"""
    panel = make_panel(np.full(MIN_BARS - 10, -0.001))

    status = retire_status(panel)

    assert status.verdict == VERDICT_ACCUMULATING
    assert status.exit_code == 0
    assert "样本不足" in status.reasons[0]
    assert "还差 10 根" in status_report(status)


def alternating(mean: float, amplitude: float, bars: int) -> np.ndarray:
    """确定性序列：均值精确可控、波动由 amplitude 决定（比随机数好断言）。"""
    net = np.full(bars, amplitude)
    net[1::2] = 2 * mean - amplitude
    return net


def test_healthy_record_passes_all_criteria() -> None:
    net = alternating(mean=0.0008, amplitude=0.005, bars=MIN_BARS + 10)

    status = retire_status(make_panel(net))

    assert status.verdict == VERDICT_HEALTHY
    assert status.exit_code == 0
    assert status.reasons == ("全部判据在阈值内",)
    assert status.rolling_sharpe > SHARPE_WATCH


def test_negative_rolling_sharpe_with_positions_kills() -> None:
    """有仓位还亏钱 = 符号被推翻，直接退役。"""
    panel = make_panel(np.full(MIN_BARS, -0.0002), exposure=EXPOSURE_FLOOR * 2)

    status = retire_status(panel)

    assert status.verdict == VERDICT_KILL
    assert status.exit_code == 2
    assert any("有仓位还在亏" in reason for reason in status.reasons)


def test_same_losses_without_positions_cannot_kill() -> None:
    """暴露低于门槛时低夏普可能只是没被触发，不许据此退役。"""
    panel = make_panel(np.full(MIN_BARS, -0.0002), exposure=EXPOSURE_FLOOR / 2)

    status = retire_status(panel)

    assert status.verdict != VERDICT_KILL
    assert any("没被触发" in reason for reason in status.reasons)


def test_half_decay_sharpe_watches() -> None:
    """夏普落在研究期一半以下但为正：只观察，不退役。"""
    net = alternating(mean=0.00005, amplitude=0.005, bars=MIN_BARS + 10)

    status = retire_status(make_panel(net))

    assert 0.0 < status.rolling_sharpe < SHARPE_WATCH
    assert status.verdict == VERDICT_WATCH
    assert status.exit_code == 1


def test_drawdown_beyond_in_sample_watches_then_kills() -> None:
    """回撤超研究期读数 → 观察；超 1.5 倍 → 退役。"""
    # 暴露置于门槛之下，把夏普判据摘掉，单独验回撤分支
    crash = lambda depth: np.concatenate(  # noqa: E731 - 测试内联更直观
        [np.full(100, 0.01), np.full(45, depth), np.zeros(MIN_BARS - 145)]
    )

    watch = retire_status(make_panel(crash(-0.005), exposure=0.0))
    assert DD_WATCH > watch.max_dd > DD_KILL
    assert watch.verdict == VERDICT_WATCH

    kill = retire_status(make_panel(crash(-0.007), exposure=0.0))
    assert kill.max_dd <= DD_KILL
    assert kill.verdict == VERDICT_KILL
    assert any("风险模型错" in reason for reason in kill.reasons)


def test_rolling_sharpe_uses_only_the_tail_window() -> None:
    """一年前的大涨不该撑住今天的滚动夏普。"""
    old = np.full(MIN_BARS, 0.01)
    recent = np.full(BARS_PER_YEAR, -0.001)

    assert rolling_sharpe(pd.Series(np.concatenate([[1.0], np.cumprod(1 + np.concatenate([old, recent]))]))) < 0


def test_status_report_contains_thresholds_context() -> None:
    panel = make_panel(np.full(MIN_BARS + 5, -0.0002), exposure=0.1)
    text = status_report(retire_status(panel))

    assert "kill" in text
    assert "滚动一年夏普" in text


@pytest.mark.parametrize("verdict,expected", [(VERDICT_HEALTHY, 0), (VERDICT_WATCH, 1), (VERDICT_KILL, 2)])
def test_exit_codes(verdict: str, expected: int) -> None:
    from src.research.paper_status import RetireStatus

    assert RetireStatus(400, verdict, 1.0, -0.01, 0.10, ()).exit_code == expected
