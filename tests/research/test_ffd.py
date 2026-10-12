"""src/research/ffd.py 的用例。

标定口径：随机游走的 ADF 读数随种子浮动（实测 min d 在 0.25~0.55 之间），
所以断言写区间不写定值；这些数字是先跑探针量出来、再写成断言的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import ffd

WALK_LENGTH = 2400
WALK_LEVEL = 100.0


def _walk(seed: int, length: int = WALK_LENGTH) -> pd.Series:
    """带固定种子的随机游走（对数价格的替身）。"""
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(size=length)) + WALK_LEVEL)


class TestFractionalWeights:
    def test_integer_orders_are_binomials(self) -> None:
        assert ffd.fractional_weights(0.0) == pytest.approx([1.0])
        assert ffd.fractional_weights(1.0) == pytest.approx([1.0, -1.0])
        assert ffd.fractional_weights(2.0) == pytest.approx([1.0, -2.0, 1.0])

    def test_weights_follow_the_recursion(self) -> None:
        weights = ffd.fractional_weights(0.4, 1e-4)
        expected = [1.0]
        while abs(expected[-1] / len(expected) * (0.4 - len(expected) + 1)) >= 1e-4:
            k = len(expected)
            expected.append(-expected[-1] / k * (0.4 - k + 1))
        assert weights == pytest.approx(expected)

    def test_lag_order_puts_the_current_value_first(self) -> None:
        """w[0] 作用于当期（(1-B)^d 的定义），否则与 convolve 的时间方向会反。"""
        assert ffd.fractional_weights(1.0)[0] == 1.0

    @pytest.mark.parametrize(
        ("threshold", "width"),
        [(1e-1, 2), (1e-2, 10), (1e-3, 54), (1e-4, 281), (1e-5, 1457)],
    )
    def test_width_shrinks_as_the_threshold_grows(
        self, threshold: float, width: int
    ) -> None:
        assert ffd.window_width(0.4, threshold) == width

    def test_integer_orders_annihilate_the_level(self) -> None:
        assert ffd.fractional_weights(1.0).sum() == pytest.approx(0.0)

    def test_truncation_leaks_the_level(self) -> None:
        """截断丢掉尾巴，权重之和不再为 0，水平因此泄漏进特征（原文未提）。"""
        leaks = [
            float(ffd.fractional_weights(0.4, th).sum()) for th in (1e-1, 1e-2, 1e-3)
        ]
        assert leaks[0] == pytest.approx(0.48, abs=0.01)
        assert leaks[1] == pytest.approx(0.264, abs=0.002)
        assert leaks[0] > leaks[1] > leaks[2] > 0.0

    def test_constant_series_transforms_to_the_leak(self) -> None:
        const = pd.Series(np.full(400, 4.0))
        values = ffd.fractional_difference(const, 0.4, 1e-2).dropna()
        leak = float(ffd.fractional_weights(0.4, 1e-2).sum()) * 4.0
        assert values.to_numpy() == pytest.approx(leak)

    def test_rejects_bad_orders_and_thresholds(self) -> None:
        with pytest.raises(ValueError, match="阶数"):
            ffd.fractional_weights(-0.1)
        with pytest.raises(ValueError, match="阈值"):
            ffd.fractional_weights(0.4, 0.0)

    def test_caps_absurd_windows(self) -> None:
        with pytest.raises(ValueError, match="窗口超过"):
            ffd.fractional_weights(0.2, 1e-9)


class TestFractionalDifference:
    def test_d_zero_is_the_identity(self) -> None:
        series = _walk(3)
        assert ffd.fractional_difference(series, 0.0).equals(series)

    def test_d_one_is_the_first_difference(self) -> None:
        """这条抓到过真 bug：照抄原文再反转一次，符号与时间方向会同时反掉。"""
        series = _walk(4)
        values = ffd.fractional_difference(series, 1.0).dropna()
        assert values.to_numpy() == pytest.approx(
            series.diff().reindex(values.index).to_numpy()
        )

    def test_d_two_is_the_second_difference(self) -> None:
        """d=2 的核对称（[1,-2,1]），反转看不出来——必须与 d=1 那条并列才有判别力。"""
        series = _walk(5)
        values = ffd.fractional_difference(series, 2.0).dropna()
        expected = series.diff().diff().reindex(values.index).to_numpy()
        assert values.to_numpy() == pytest.approx(expected)

    def test_warmup_is_exactly_the_window_width(self) -> None:
        series = _walk(6)
        values = ffd.fractional_difference(series, 0.4, 1e-2)
        assert bool(values.iloc[:10].isna().all())
        assert bool(values.iloc[10:].notna().all())

    def test_no_lookahead(self) -> None:
        series = _walk(7)
        mutated = series.copy()
        mutated.iloc[-1] *= 1.5
        before = ffd.fractional_difference(series, 0.4, 1e-2).to_numpy()[:-1]
        after = ffd.fractional_difference(mutated, 0.4, 1e-2).to_numpy()[:-1]
        assert before == pytest.approx(after, nan_ok=True)

    def test_missing_rows_are_not_invented(self) -> None:
        series = _walk(8)
        series.iloc[1000] = np.nan
        values = ffd.fractional_difference(series, 0.4, 1e-2)
        assert bool(np.isnan(values.iloc[1000]))
        assert bool(np.isfinite(values.iloc[1001]))

    def test_short_series_is_all_warmup(self) -> None:
        values = ffd.fractional_difference(_walk(9, length=8), 0.4, 1e-2)
        assert bool(values.isna().all())


class TestOrders:
    def test_white_noise_is_already_stationary(self) -> None:
        rng = np.random.default_rng(11)
        noise = pd.Series(rng.normal(size=WALK_LENGTH))
        assert ffd.minimum_d(noise) == 0.0

    def test_random_walk_needs_a_fractional_order(self) -> None:
        """第 5 章的核心主张：水平不过 ADF，整数阶（d=1）平稳但记忆归零。"""
        for seed in (0, 1, 2):
            walk = _walk(seed)
            table = ffd.scan_orders(walk)
            assert not bool(table.loc[0.0, "passed"])
            order = ffd.minimum_d(walk)
            assert 0.0 < order < 0.7
            assert ffd.memory_retention(walk, order) > 0.95
            assert abs(ffd.memory_retention(walk, 1.0)) < 0.1

    def test_scan_passes_exactly_where_the_statistic_beats_the_critical_value(
        self,
    ) -> None:
        table = ffd.scan_orders(_walk(12), threshold=1e-2)
        assert list(table.columns) == [
            "adf_stat",
            "p_value",
            "critical_5pct",
            "memory",
            "passed",
        ]
        passed = table["passed"].to_numpy(dtype=bool)
        assert passed.any() and not passed[0]
        assert bool(
            (table["adf_stat"].to_numpy() < table["critical_5pct"].to_numpy())[
                passed
            ].all()
        )

    def test_short_series_yields_no_adf_reading(self) -> None:
        table = ffd.scan_orders(_walk(13, length=8))
        blanks = table[["adf_stat", "p_value", "critical_5pct"]].isna().to_numpy()
        assert bool(blanks.all())
        assert bool(table["passed"].eq(False).all())
