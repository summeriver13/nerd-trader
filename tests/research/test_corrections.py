"""`src/research/corrections.py` 的不变量：重采样结构、检验的尺寸与功效、BH 步进规则。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.corrections import (
    DEFAULT_ALPHA,
    MIN_OBSERVATIONS,
    benjamini_hochberg,
    default_block,
    hansen_spa,
    naive_p_values,
    reality_check,
    stationary_bootstrap_indices,
)

# 用例规模：小到跑得动，大到尺寸/功效读得出信号。
OBSERVATIONS = 300
CANDIDATES = 15
BOOTSTRAPS = 80
EXPERIMENTS = 60
PLANTED_MEAN = 0.40


def _null_frame(
    seed: int, *, observations: int = OBSERVATIONS, candidates: int = CANDIDATES
) -> pd.DataFrame:
    """纯噪声候选：没有任何一个真的能预测。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(observations, candidates)))


def _planted_frame(seed: int, *, strength: float = PLANTED_MEAN) -> pd.DataFrame:
    """在一个噪声候选里种进真信号（其余照旧为噪声）。"""
    frame = _null_frame(seed)
    frame[0] = frame[0] + strength
    return frame


def test_stationary_bootstrap_block_one_restarts_almost_every_step() -> None:
    """块长 1 即 iid 重采样：连续取到「上一格 +1」的概率应约为 1/T。"""
    rng = np.random.default_rng(7)
    indices = stationary_bootstrap_indices(OBSERVATIONS, 1, rng)
    continuations = np.mean(indices[1:] == (indices[:-1] + 1) % OBSERVATIONS)
    assert indices.shape == (OBSERVATIONS,)
    assert indices.min() >= 0 and indices.max() < OBSERVATIONS
    assert continuations < 5.0 / OBSERVATIONS


def test_stationary_bootstrap_long_block_stays_contiguous() -> None:
    """块长等于样本长度时，几乎整条序列是一段连续片段（只该被极少数重启打断）。"""
    rng = np.random.default_rng(7)
    indices = stationary_bootstrap_indices(OBSERVATIONS, OBSERVATIONS, rng)
    breaks = np.sum(indices[1:] != (indices[:-1] + 1) % OBSERVATIONS)
    assert breaks < OBSERVATIONS * 0.1


def test_bootstrap_is_reproducible_and_seed_sensitive() -> None:
    """同种子逐位相同、换种子必须变——否则「固定种子」这件事本身没意义。"""
    first = stationary_bootstrap_indices(OBSERVATIONS, 7, np.random.default_rng(11))
    again = stationary_bootstrap_indices(OBSERVATIONS, 7, np.random.default_rng(11))
    other = stationary_bootstrap_indices(OBSERVATIONS, 7, np.random.default_rng(12))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_default_block_is_cube_root_of_observations() -> None:
    """默认块长 T^(1/3)（取整），且永远不小于 1。"""
    assert default_block(1000) == 10
    assert default_block(1) == 1


def test_spa_rejects_rarely_under_the_null_but_naive_max_does_not() -> None:
    """尺寸：整族都是噪声时 SPA 的拒绝率应贴近 α，而「挑最好的单独检验」会大规模误报。"""
    spa_rejections = 0
    naive_rejections = 0
    for seed in range(EXPERIMENTS):
        frame = _null_frame(seed)
        if (
            hansen_spa(frame, replications=BOOTSTRAPS, seed=seed).p_consistent
            <= DEFAULT_ALPHA
        ):
            spa_rejections += 1
        if float(naive_p_values(frame).min()) <= DEFAULT_ALPHA:
            naive_rejections += 1
    spa_rate = spa_rejections / EXPERIMENTS
    naive_rate = naive_rejections / EXPERIMENTS
    assert spa_rate <= 4.0 * DEFAULT_ALPHA, spa_rate
    assert naive_rate >= 0.35, naive_rate
    assert naive_rate > spa_rate


def test_spa_finds_a_planted_signal() -> None:
    """功效：种进 0.4σ 的真信号后，SPA 在绝大多数重抽样里都能拒掉。"""
    rejections = sum(
        hansen_spa(
            _planted_frame(seed), replications=BOOTSTRAPS, seed=seed
        ).p_consistent
        <= DEFAULT_ALPHA
        for seed in range(EXPERIMENTS)
    )
    assert rejections >= EXPERIMENTS * 0.9, rejections


def test_reality_check_and_spa_agree_on_a_single_candidate() -> None:
    """K = 1 时两种检验都退化成「最大 p 值 = 1/(R+1)」（有限次重采样的下界）。"""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"only": rng.normal(size=OBSERVATIONS) + PLANTED_MEAN})
    sparse = 5 * BOOTSTRAPS
    check = reality_check(frame, replications=sparse, seed=3)
    spa = hansen_spa(frame, replications=sparse, seed=3)
    assert check.p_value == pytest.approx(1.0 / (sparse + 1.0))
    assert spa.p_consistent == pytest.approx(1.0 / (sparse + 1.0))


def test_spa_upper_bound_is_never_more_optimistic_than_consistent() -> None:
    """保守上界必须 ≥ 一致版（把负表现候选剪到 0 只会放大零分布）。"""
    frame = _null_frame(0)
    frame[1] = frame[1] - 1.0
    frame[2] = frame[2] - 0.5
    spa = hansen_spa(frame, replications=200, seed=5)
    assert spa.p_upper >= spa.p_consistent
    assert 0.0 < spa.p_consistent <= 1.0
    assert int(spa.best) in set(frame.columns)


def test_constant_candidate_never_wins() -> None:
    """零波动候选的标准误取 inf：t 归零，不能靠「没有波动」冒充显著。"""
    frame = _null_frame(0)
    frame[3] = 5.0
    spa = hansen_spa(frame, replications=50, seed=9)
    assert float(spa.statistics.loc[3]) == 0.0
    assert spa.best != "3"


def test_missing_observations_are_dropped_and_counted() -> None:
    """有缺值的候选：整行丢掉（bootstrap 必须在同一个 T 上），候选数不变。"""
    frame = _null_frame(1)
    frame.iloc[:10, 2] = np.nan
    spa = hansen_spa(frame, replications=30, seed=13)
    assert spa.candidates == CANDIDATES
    assert spa.observations == OBSERVATIONS - 10


def test_short_sample_returns_nan_without_raising() -> None:
    """样本短于下限时给 nan 而不是抛异常，免得整条研究线被一个短候选打断。"""
    frame = _null_frame(1, observations=MIN_OBSERVATIONS - 1)
    spa = hansen_spa(frame, replications=20, seed=2)
    check = reality_check(frame, replications=20, seed=2)
    assert np.isnan(spa.p_consistent) and np.isnan(spa.p_upper)
    assert np.isnan(check.p_value)
    assert spa.statistics.empty and check.statistics.empty


def test_benjamini_hochberg_is_step_up_not_per_hypothesis() -> None:
    """BH 的要点是步进：单独看第一条不过门槛，跳到第二条却能把两条一起拒掉。"""
    result = benjamini_hochberg(pd.Series([0.03, 0.04], index=["a", "b"]), alpha=0.05)
    assert result.labels == ("a", "b")
    assert result.cutoff == pytest.approx(0.05)

    plain = benjamini_hochberg(pd.Series([0.03, 0.06], index=["a", "b"]), alpha=0.05)
    assert plain.labels == ()


def test_benjamini_hochberg_ignores_missing_p_values() -> None:
    """缺 p 值的候选直接不参与，且不影响门槛里的 m（否则门槛会被算低）。"""
    result = benjamini_hochberg(
        pd.Series([0.01, np.nan, 0.02], index=["a", "b", "c"]), alpha=0.05
    )
    assert result.labels == ("a", "c")
    assert result.cutoff == pytest.approx(0.05)
