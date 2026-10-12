"""
扫描器的两道判据：平台连接规则（`plateau.py`）与有效试验数校正（`sweep.py`）。

IC 谁都会算，判据才是防过拟合的地方——这个文件钉住的就是判据本身：
段怎么连、尖峰怎么丢、门槛怎么算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.factors.evaluation import ic_stats
from src.research.factors.library import DIRECTION_LONG
from src.research.factors.panel import Panel
from src.research.factors.plateau import (
    KIND_PLATEAU,
    KIND_SPIKE,
    find_runs,
    plateau_table,
    verdict_table,
)
from src.research.factors.sweep import (
    SWEEP_FAMILIES,
    SweepFamily,
    critical_t,
    effective_tests,
    resolve_families,
    sweep,
    sweep_ic_table,
)

THRESHOLD = 2.5


def _series(values: dict[int, float]) -> pd.Series:
    """按窗口号建的序列（索引就是窗口号，不是 0..n-1）。"""
    return pd.Series(values, dtype=float).sort_index()


def _frame(data: dict[str, list[float]]) -> pd.DataFrame:
    bars = len(next(iter(data.values())))
    index = pd.date_range("2020-01-01", periods=bars, freq="1D", tz="UTC")
    return pd.DataFrame(data, index=index)


def _mask(panel: Panel) -> pd.DataFrame:
    """全可交易掩码：合成面板不跑池子规则，只看因子本身。"""
    return pd.DataFrame(True, index=panel.index, columns=panel.symbols)


def _mean_reverting_panel(bars: int = 240, symbols: int = 40, seed: int = 11) -> Panel:
    """收益为 AR(1) 且系数为负：过去涨得多的接下来跌，反转效应是真的。"""
    rng = np.random.default_rng(seed)
    close: dict[str, list[float]] = {}
    for i in range(symbols):
        noise = rng.normal(0.0, 0.02, bars)
        steps = np.zeros(bars)
        for t in range(1, bars):
            steps[t] = -0.6 * steps[t - 1] + noise[t]
        close[f"S{i:02d}-USDT"] = np.exp(np.cumsum(steps)).tolist()
    return Panel(
        timeframe="1d",
        close=_frame(close),
        volume=_frame(close) * 0.0 + 1.0,
        notional=_frame(close) * 0.0 + 1e7,
    )


# === 平台连接规则 ===


def test_runs_join_contiguous_same_sign() -> None:
    """连续同号且过门槛的窗口连成一段，段内取 IC 均值与最低 |t|。"""
    runs = find_runs(
        "mom",
        1,
        _series({1: -0.03, 2: -0.04, 3: -0.05}),
        _series({1: 3.0, 2: 4.0, 3: 5.0}),
        THRESHOLD,
    )
    assert len(runs) == 1
    assert (runs[0].start, runs[0].end, runs[0].windows) == (1, 3, 3)
    assert runs[0].mean_ic == pytest.approx(-0.04)
    assert runs[0].min_abs_t == pytest.approx(3.0)


def test_runs_split_on_sign_change() -> None:
    """符号翻转就断开：+0.03 与 −0.03 不是同一个效应。"""
    runs = find_runs(
        "mom",
        1,
        _series({1: 0.03, 2: 0.03, 3: -0.03, 4: -0.03}),
        _series(dict.fromkeys((1, 2, 3, 4), 4.0)),
        THRESHOLD,
    )
    assert [(run.start, run.end) for run in runs] == [(1, 2), (3, 4)]


def test_runs_split_on_gap() -> None:
    """窗口号不连续不算同一段：1 与 3 显著而 2 不显著，说明效应不稳。"""
    runs = find_runs(
        "mom",
        1,
        _series({1: -0.03, 2: -0.001, 3: -0.03}),
        _series({1: 4.0, 2: 0.5, 3: 4.0}),
        THRESHOLD,
    )
    assert [(run.start, run.windows) for run in runs] == [(1, 1), (3, 1)]


def test_runs_treat_zero_and_nan_as_failure() -> None:
    """0 与 NaN 都算「没通过」：0 是「没有效应」，不是「显著为零」。"""
    runs = find_runs(
        "mom",
        1,
        _series({1: -0.03, 2: 0.0, 3: float("nan"), 4: -0.03}),
        _series(dict.fromkeys((1, 2, 3, 4), 4.0)),
        THRESHOLD,
    )
    assert [run.start for run in runs] == [1, 4]


def test_plateau_table_labels_by_length() -> None:
    """长度够 min_plateau 的是平台，孤立的标尖峰。"""
    table = pd.DataFrame(
        {
            "key": [f"mom_{w}@1" for w in range(1, 6)],
            "family": ["mom"] * 5,
            "window": [1, 2, 3, 4, 5],
            "horizon": [1] * 5,
            "ic": [-0.04, -0.04, -0.04, -0.001, -0.05],
            "t_stat": [4.0, 4.0, 4.0, 0.5, 4.0],
        }
    )
    thresholds = pd.DataFrame({"horizon": [1], "critical_t": [THRESHOLD]})
    frame = plateau_table(table, thresholds, min_plateau=3)
    kinds = dict(zip(frame["start"], frame["kind"], strict=True))
    assert kinds[1] == KIND_PLATEAU
    assert kinds[5] == KIND_SPIKE


def test_verdict_table_drops_spikes() -> None:
    """裁决只统计平台：尖峰不参与，否则「挑最大值」又回来了。"""
    frame = pd.DataFrame(
        {
            "family": ["mom", "mom"],
            "horizon": [1, 1],
            "start": [1, 9],
            "end": [5, 9],
            "windows": [5, 1],
            "mean_ic": [-0.04, -0.09],
            "min_abs_t": [3.0, 9.0],
            "kind": [KIND_PLATEAU, KIND_SPIKE],
        }
    )
    verdict = verdict_table(frame)
    assert len(verdict) == 1
    assert float(verdict["longest"].iloc[0]) == 5.0
    assert float(verdict["mean_ic"].iloc[0]) == pytest.approx(-0.04)


# === 有效试验数校正 ===


def test_effective_tests_collapses_perfectly_correlated() -> None:
    """完全相关的候选只算一个试验：12 条同一条线时 N_eff → 1。"""
    base = np.linspace(-1.0, 1.0, 200)
    frame = pd.DataFrame({f"c{i}": base for i in range(12)})
    assert effective_tests(frame) == pytest.approx(1.0, abs=1e-6)


def test_effective_tests_counts_independent() -> None:
    """相互独立时 N_eff 接近名义数：校正不该凭空放宽门槛。"""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(4000, 20)))
    assert effective_tests(frame) > 14.0


def test_critical_t_shrinks_with_fewer_tests() -> None:
    """试验越少门槛越低——这正是校正的目的（别把真信号一起杀掉）。"""
    assert critical_t(1.0) < critical_t(100.0) < critical_t(10000.0)


# === 扫描器 ===


def test_sweep_family_spec_has_no_direction_assumption() -> None:
    """扫描不预设方向：IC 的符号就是发现，spec 的方向必须恒为 +1。"""
    spec = SWEEP_FAMILIES["mom"].spec(3)
    assert spec.direction == DIRECTION_LONG
    assert spec.key == "mom_3"
    assert spec.warmup == 3


def test_resolve_families_rejects_unknown() -> None:
    """未知族名当场报错，不静默返回空扫描。"""
    with pytest.raises(KeyError):
        resolve_families("mom,nope")


def test_sweep_ic_matches_ic_stats() -> None:
    """扫描表的 IC 与 t 必须与既有评估台逐位一致：口径不能各算各的。"""
    panel = _mean_reverting_panel(bars=120, symbols=20)
    mask = _mask(panel)
    family = SweepFamily(
        "mom", "动量", (3, 5), lambda w: w, SWEEP_FAMILIES["mom"].compute
    )
    table, series = sweep_ic_table(panel, mask, horizons=(1,), families=[family])
    assert len(table) == 2
    assert series.shape[1] == 2
    for row in table.to_dict(orient="records"):
        spec = family.spec(int(row["window"]))
        expected = ic_stats(spec, panel, mask, 1)
        assert float(row["ic"]) == pytest.approx(expected.mean, rel=1e-12)
        assert float(row["t_stat"]) == pytest.approx(expected.t_stat, rel=1e-12)


def test_sweep_finds_planted_platform() -> None:
    """植入对窗口规格稳健的反转效应：扫描必须报出跨多窗口的平台。"""
    panel = _mean_reverting_panel()
    family = SweepFamily(
        "mom", "动量", tuple(range(1, 8)), lambda w: w, SWEEP_FAMILIES["mom"].compute
    )
    result = sweep(panel, _mask(panel), horizons=(1,), families=[family], min_plateau=3)
    assert not result.verdicts.empty
    assert float(result.verdicts["longest"].iloc[0]) >= 3
    # 负号 = 过去涨得多的接下来跌，即反转
    assert float(result.verdicts["mean_ic"].iloc[0]) < 0.0


def test_every_family_scans_a_consecutive_window_axis() -> None:
    """平台判据只连接窗口号相邻的段：步长≠1 的轴永远只出尖峰（FFD 接进来时踩过）。"""
    for name, family in SWEEP_FAMILIES.items():
        windows = family.windows
        assert windows == tuple(range(windows[0], windows[-1] + 1)), name


def test_ffd_family_warmup_is_the_real_nan_run() -> None:
    """FFD 族的预热随阶数变，必须等于真实 NaN 段长度，否则 IC 会把暖机值当信号。"""
    panel = _mean_reverting_panel()
    family = SWEEP_FAMILIES["ffd"]
    for window in (20, 40):
        values = family.compute(panel, window)
        present = values.notna().to_numpy().any(axis=1)
        first_valid = int(np.flatnonzero(present)[0])
        assert first_valid == family.warmup_of(window) > 0
