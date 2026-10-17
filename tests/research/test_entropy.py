"""熵特征的不变量：估计量必须能分辨「规律 / 随机」，且不得用未来信息。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.sampling.entropy import (
    FEATURE_COLUMNS,
    MIN_SAMPLES,
    EntropyOptions,
    _lz76_count,
    binarize,
    entropy_panel,
    entropy_panel_name,
    kontoyiannis_bits,
    lempel_ziv_bits,
    ordinal_bits,
    shannon_bits,
    window_bits,
)

RANDOM_SIZE = 2000


def _random(size: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(size)


def test_panel_name_keys_every_configuration() -> None:
    """面板名必须区分窗口与基线次数（少一个就会静默读到别的配置）。"""
    base = EntropyOptions(window=250, surrogates=2)
    assert entropy_panel_name("swap", "1d", base) != entropy_panel_name(
        "swap", "1d", EntropyOptions(window=500, surrogates=2)
    )
    assert entropy_panel_name("swap", "1d", base) != entropy_panel_name(
        "swap", "1d", EntropyOptions(window=250, surrogates=8)
    )
    assert entropy_panel_name("swap", "1d", base) != entropy_panel_name(
        "spot", "1d", base
    )
    assert entropy_panel_name("swap", "1d", base, prefix="entropy_study").startswith(
        "entropy_study_swap_1d_w250_s2"
    )


def test_lz76_count_matches_hand_computed_values() -> None:
    """钉住解析本身：常数串与交替串各 4 段（手算），随机串段数必须远少于 n/4。

    n/4 这条界是防「每根一个短语」的实现错：曾把随机串数成 ≈n/2 段、熵率报成 5.46
    （二进制字母表的熵率上界是 1，超过 1 只可能是实现错）。
    """
    assert _lz76_count(np.zeros(8, dtype=np.int8)) == 4
    assert _lz76_count(np.array([0, 1] * 4, dtype=np.int8)) == 4
    assert _lz76_count(binarize(_random(RANDOM_SIZE))) < RANDOM_SIZE / 4


def test_constant_series_has_zero_dispersion_entropy() -> None:
    """常数序列：香农熵与顺序熵都是 0（不是缺失），LZ 也远低于随机。"""
    flat = np.ones(200)
    assert shannon_bits(flat) == 0.0
    assert ordinal_bits(flat) == 0.0
    assert lempel_ziv_bits(flat) < lempel_ziv_bits(_random(200))


def test_shannon_entropy_tracks_dispersion_and_not_equal_frequency() -> None:
    """固定边界分箱：宽波动窗口的熵必须高于窄波动窗口（等频分箱给不出这个差别）。

    等频分箱会把熵钉死在 log2(箱数)（随机与常数都给 3.3219），故本模块一律固定边界。
    """
    narrow = np.random.default_rng(5).normal(0.0, 0.002, 300)
    wide = np.random.default_rng(5).normal(0.0, 0.02, 300)
    assert shannon_bits(wide) > shannon_bits(narrow) + 0.5
    half = np.concatenate([np.full(100, -0.03), np.full(100, 0.03)])
    assert shannon_bits(half, np.array([-np.inf, 0.0, np.inf])) == pytest.approx(1.0)


def test_random_sequence_entropy_rate_is_one_with_the_measured_bias() -> None:
    """iid 序列的熵率读数在 1 bit/根附近（实测定值）：LZ 偏高约 5%，Kontoyiannis 偏高约 11%。

    这个上偏（n=2000 时 1.1135、n=20000 时 1.0840）就是实证里一律扣同窗置换基线的原因。
    """
    random = _random(RANDOM_SIZE)
    assert lempel_ziv_bits(random) == pytest.approx(1.0, abs=0.15)
    small = kontoyiannis_bits(random)
    large = kontoyiannis_bits(_random(RANDOM_SIZE * 10, seed=4))
    assert 1.0 <= large < small <= 1.6


def test_periodic_sequence_is_far_below_random() -> None:
    """周期序列（0/1 交替）几乎完全可预测，两个序列类估计量都该塌下去。"""
    random = _random(RANDOM_SIZE, seed=1)
    periodic = np.resize(np.array([-1.0, 1.0]), RANDOM_SIZE)
    assert lempel_ziv_bits(periodic) < 0.5 * lempel_ziv_bits(random)
    assert kontoyiannis_bits(periodic) < 0.5 * kontoyiannis_bits(random)
    assert binarize(periodic).sum() == RANDOM_SIZE / 2


def test_kontoyiannis_stays_above_zero_but_below_random_on_a_trend() -> None:
    """单调上行序列只有一种二值模式：熵率该显著低于随机，但不为 0（小样本偏置）。"""
    trend = np.arange(RANDOM_SIZE, dtype=float)
    assert (
        0.0 < kontoyiannis_bits(trend) < 0.5 * kontoyiannis_bits(_random(RANDOM_SIZE))
    )


def test_short_series_returns_missing_not_a_guess() -> None:
    """样本不足一律缺失：小样本熵估计偏低会被误读成「可预测」。"""
    tiny = _random(MIN_SAMPLES - 1)
    assert np.isnan(shannon_bits(tiny))
    assert np.isnan(ordinal_bits(tiny))
    assert np.isnan(lempel_ziv_bits(tiny))
    assert np.isnan(kontoyiannis_bits(tiny))


def test_window_bits_returns_every_feature_and_excess_is_relative() -> None:
    """窗口读数键齐全；置换对照让「excess」只留序列特有的结构。"""
    options = EntropyOptions(window=60, surrogates=2)
    random = _random(60, seed=2)
    point = window_bits(random, options)
    assert set(point) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(value) for value in point.values())
    periodic = np.resize(np.array([-1.0, 1.0]), 60)
    periodic_point = window_bits(periodic, options)
    assert periodic_point["lz_excess"] < point["lz_excess"]
    assert periodic_point["konto_excess"] < point["konto_excess"]


def test_entropy_panel_needs_a_full_window_and_never_looks_ahead() -> None:
    """面板：窗口未满不出行；改最后一天不得改动此前的读数（无未来信息）。"""
    values = _random(40 * 2, seed=3).reshape(40, 2)
    frame = pd.DataFrame({"a": values[:, 0], "b": values[:, 1]})
    options = EntropyOptions(window=30, surrogates=1)
    panel = entropy_panel(frame, options)
    assert panel["date"].min() == frame.index[29]
    assert panel.groupby("symbol").size().to_dict() == {"a": 11, "b": 11}

    changed = frame.copy()
    changed.iloc[-1] = changed.iloc[-1] + 100.0
    moved = entropy_panel(changed, options)
    key = ["date", "symbol"]
    joined = panel.merge(moved, on=key, suffixes=("_before", "_after"))
    last_date = frame.index[-1]
    earlier = joined[joined["date"] < last_date]
    for column in FEATURE_COLUMNS:
        assert np.allclose(earlier[f"{column}_before"], earlier[f"{column}_after"])
    tail = joined[joined["date"] == last_date]
    assert not np.allclose(tail["konto_bits_before"], tail["konto_bits_after"])
