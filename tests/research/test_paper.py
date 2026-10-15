"""前向纸面账本的纯合成数据用例：口径、哈希链与数据闸。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.core.costs import SPOT_FEES
from src.research.paper import (
    COST_RATE,
    GENESIS,
    PaperSpec,
    append_ledger,
    coverage_ok,
    forward_stats,
    ledger_panel,
    ledger_rows,
    nav_panel,
    positions,
    read_ledger,
    scale_signals,
    signal_panel,
    verify_ledger,
)

BARS = 160
TREND = 1.02
# 波动率目标/回看取默认（0.40 / 30），入场窗口缩短以尽早出信号
SPEC = PaperSpec(entry_periods=(5, 10, 20), start=None)


def make_bars(stem: str, *, trend: float = TREND, bars: int = BARS) -> pd.DataFrame:
    """单调上行的合成日线：high = close×1.01，故收盘价可突破前高触发信号。"""
    index = pd.date_range("2024-01-01", periods=bars, freq="D", tz="UTC")
    close = 100.0 * trend ** np.arange(bars, dtype=float)
    return pd.DataFrame(
        {
            "open": close / trend,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def make_pool(stems: tuple[str, ...] = ("AAA-USDT", "BBB-USDT", "CCC-USDT")) -> dict[str, pd.DataFrame]:
    return {stem: make_bars(stem) for stem in stems}


def returns_of(bars_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({stem: bars["close"].pct_change() for stem, bars in bars_map.items()})


def test_positions_lag_divide_and_missing_cells() -> None:
    """持仓 = 信号滞后一根 ÷ 持仓当日 N_t；无当根数据的格子留 NaN 而不是 0。"""
    bars_map = make_pool()
    signals = signal_panel(bars_map, SPEC)
    returns = returns_of(bars_map)
    counts = returns.notna().sum(axis=1).astype(float)
    held = positions(signals, counts)

    assert held.index[0] == signals.index[0]
    assert held.iloc[0].isna().all()  # 滞后一根后首行没有前值
    assert float(held.sum(axis=1).iloc[-1]) == pytest.approx(
        float(signals.sum(axis=1).shift(1).iloc[-1]) / 3.0
    )

    # 砍掉最后两根里的一个标的：该格必须变 NaN，否则会被当成清仓计入换手
    trimmed = make_pool()
    trimmed["CCC-USDT"] = trimmed["CCC-USDT"].iloc[:-2]
    held_trimmed = positions(signal_panel(trimmed, SPEC), counts)
    assert held_trimmed["CCC-USDT"].iloc[-1:].isna().all()


def test_nav_panel_costs_and_compounding() -> None:
    """净值 = Π(1 + Σ持仓×收益 − 换手×费率)，首行按零仓位起算。"""
    bars_map = make_pool()
    returns = returns_of(bars_map)
    counts = returns.notna().sum(axis=1).astype(float)
    held = positions(signal_panel(bars_map, SPEC), counts)
    panel = nav_panel(held, returns)

    assert panel["turnover"].iloc[0] == pytest.approx(0.0)
    assert float(panel["exposure"].max()) > 0.0  # 趋势行情里必须真的持仓
    manual = (1.0 + panel["gross_ret"] - panel["turnover"] * COST_RATE).cumprod()
    assert float(panel["nav"].iloc[-1]) == pytest.approx(float(manual.iloc[-1]))
    assert float(panel["cost_ret"].sum()) == pytest.approx(
        float(panel["turnover"].sum()) * COST_RATE
    )


def test_nav_panel_forward_start_rebases_and_charges_entry() -> None:
    """前向起点：净值重新起算为 1，且把建仓成本记在首行。"""
    bars_map = make_pool()
    returns = returns_of(bars_map)
    counts = returns.notna().sum(axis=1).astype(float)
    held = positions(signal_panel(bars_map, SPEC), counts)
    start = cast(pd.Timestamp, held.index[-5])

    panel = nav_panel(held, returns, start=start)
    assert len(panel) == 5
    assert float(panel["gross_ret"].iloc[0]) == 0.0  # 起点前的涨跌不回填
    assert float(panel["nav"].iloc[0]) == pytest.approx(
        1.0 - float(panel["turnover"].iloc[0]) * COST_RATE
    )
    assert float(panel["turnover"].iloc[0]) == pytest.approx(
        float(held.loc[start].abs().sum())
    )
    assert float(panel["cost_ret"].iloc[0]) > 0.0  # 建仓成本落在首行
    assert float(panel["net_ret"].iloc[0]) != pytest.approx(
        float(panel["gross_ret"].iloc[0])
    )


def test_scale_signals_is_live_target() -> None:
    """账本里的 target 是信号当日的口径（实盘收盘后要调的仓位）。"""
    bars_map = make_pool()
    returns = returns_of(bars_map)
    counts = returns.notna().sum(axis=1).astype(float)
    signals = signal_panel(bars_map, SPEC)
    target = scale_signals(signals, counts)
    assert float(target.sum(axis=1).iloc[-1]) == pytest.approx(
        float(signals.sum(axis=1).iloc[-1]) / 3.0
    )


def test_ledger_roundtrip_and_hash_chain(tmp_path: Path) -> None:
    """账本只增不改：重复追加不产生新行，改动历史行会被链校验抓到。"""
    bars_map = make_pool()
    returns = returns_of(bars_map)
    counts = returns.notna().sum(axis=1).astype(float)
    signals = signal_panel(bars_map, SPEC)
    held = positions(signals, counts)
    panel = nav_panel(held, returns)
    rows = ledger_rows(panel, held, scale_signals(signals, counts))

    path = tmp_path / "ledger.jsonl"
    added, conflicts = append_ledger(path, rows[:10])
    assert (added, conflicts) == (10, [])
    again, conflicts_again = append_ledger(path, rows[:10])
    assert (again, conflicts_again) == (0, [])

    stored = read_ledger(path)
    assert len(stored) == 10
    assert stored[0]["prev_hash"] == GENESIS
    assert verify_ledger(stored) is None
    assert stored[-1]["n_target"] == len(stored[-1]["target"])

    # 追加后续行应接着链尾，而不是重开一条链
    tail_added, _ = append_ledger(path, rows)
    stored = read_ledger(path)
    assert tail_added == len(rows) - 10
    assert verify_ledger(stored) is None

    # 篡改历史行：链断裂点必须指到该行
    stored[3]["nav"] = float(stored[3]["nav"]) + 0.5
    assert verify_ledger(stored) == str(rows[3]["date"])

    # 同日重算结果不同 → 报冲突且不覆盖
    mutated = [{**row, "nav": float(row["nav"]) + 1.0} for row in rows[:10]]
    _, conflicts = append_ledger(path, mutated)
    assert conflicts == [str(row["date"]) for row in rows[:10]]
    assert float(read_ledger(path)[0]["nav"]) == float(rows[0]["nav"])


def test_forward_stats_and_readout(tmp_path: Path) -> None:
    """前向读数只统计账本里真正走出来的那段。"""
    rows = [
        {
            "date": "2026-01-01",
            "nav": 1.0,
            "btc_nav": 1.0,
            "exposure": 0.0,
            "turnover": 0.0,
        },
        {
            "date": "2026-01-02",
            "nav": 1.1,
            "btc_nav": 1.05,
            "exposure": 0.2,
            "turnover": 0.1,
        },
    ]
    stats = forward_stats(ledger_panel(rows))
    assert stats["bars"] == 1.0
    assert stats["total_ret"] == pytest.approx(0.1)
    assert stats["btc_total_ret"] == pytest.approx(0.05)
    assert stats["exposure"] == pytest.approx(0.1)
    assert forward_stats(pd.DataFrame())["bars"] == 0.0


def test_coverage_gate_rejects_stale_tail() -> None:
    """末根只有零星几只标的出数时必须拒绝记账（分母就是权重本身）。"""
    bars_map = make_pool()
    returns = returns_of(bars_map)
    assert coverage_ok(returns)

    stale = make_pool()
    stale["BBB-USDT"] = stale["BBB-USDT"].iloc[:-1]
    stale["CCC-USDT"] = stale["CCC-USDT"].iloc[:-1]
    assert not coverage_ok(returns_of(stale))


def test_cost_rate_matches_spot_schedule() -> None:
    """账本成本口径 = 现货 taker + 市价滑点，与研究侧的现货费率臂同源。"""
    assert pytest.approx(SPOT_FEES.taker + SPOT_FEES.slippage) == COST_RATE
