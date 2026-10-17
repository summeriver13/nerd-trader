"""src/research/backtest/sizing.py 的用例（de Prado 第 10 章下注规模）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.backtest import sizing


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def _options(horizon: int, cost_one_way: float = 0.0) -> sizing.PositionOptions:
    return sizing.PositionOptions(horizon=horizon, cost_one_way=cost_one_way)


class TestBetSizeFromProbability:
    def test_zero_at_chance(self) -> None:
        assert float(sizing.bet_size_from_probability(_series([0.5])).iloc[0]) == 0.0

    def test_matches_hand_computed_value(self) -> None:
        # p = 0.75: z = 0.25 / sqrt(0.75 * 0.25) = 0.57735027, Z[z] = 0.71814857
        size = float(sizing.bet_size_from_probability(_series([0.75])).iloc[0])
        assert size == pytest.approx(0.43629714, abs=1e-7)

    def test_saturates_at_the_extremes_without_warning(self) -> None:
        # filterwarnings = error：这里的除法若触发 RuntimeWarning，用例会直接失败
        sizes = sizing.bet_size_from_probability(_series([0.0, 1.0]))
        assert float(sizes.iloc[0]) == pytest.approx(-1.0)
        assert float(sizes.iloc[1]) == pytest.approx(1.0)

    def test_is_monotone_in_probability(self) -> None:
        probabilities = [float(p) for p in np.linspace(0.0, 1.0, 21)]
        sizes = sizing.bet_size_from_probability(_series(probabilities))
        assert sizes.is_monotonic_increasing

    def test_complement_negates(self) -> None:
        probabilities = [float(p) for p in np.linspace(0.01, 0.49, 7)]
        low = sizing.bet_size_from_probability(_series(probabilities))
        high = sizing.bet_size_from_probability(
            _series([1.0 - p for p in probabilities])
        )
        np.testing.assert_allclose(low.to_numpy() + high.to_numpy(), 0.0, atol=1e-12)

    def test_nan_propagates(self) -> None:
        sizes = sizing.bet_size_from_probability(_series([float("nan"), 0.7]))
        assert bool(np.isnan(sizes.iloc[0]))
        assert not bool(np.isnan(sizes.iloc[1]))


class TestSizedBet:
    def test_below_chance_means_no_bet_not_a_reverse_bet(self) -> None:
        """p < 0.5 时作者原式给负值；元标签里那语义是「别下」，不是「反着下」。"""
        bet = sizing.sized_bet(_series([0.2, 0.5]), _series([1.0, 1.0]))
        assert float(bet.iloc[0]) == 0.0
        assert float(bet.iloc[1]) == 0.0

    def test_carries_the_side_sign(self) -> None:
        bet = sizing.sized_bet(_series([0.8, 0.8]), _series([1.0, -1.0]))
        assert float(bet.iloc[0]) > 0.0
        assert float(bet.iloc[1]) < 0.0
        assert float(bet.iloc[0]) == pytest.approx(-float(bet.iloc[1]))


class TestAverageActiveBets:
    def test_constant_exposure_does_not_double(self) -> None:
        """连续每天各开一注 1 倍仓，暴露仍是 1 倍——取平均不是求和。"""
        weights = pd.DataFrame({"A": [1.0] * 5})
        active = sizing.average_active_bets(weights, horizon=2)
        np.testing.assert_allclose(active["A"].to_numpy()[1:], 1.0)

    def test_is_a_moving_average_of_the_desired_position(self) -> None:
        weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]})
        active = sizing.average_active_bets(weights, horizon=2)
        np.testing.assert_allclose(active["A"].to_numpy()[1:], [1.0, 0.5, 0.0])

    def test_warmup_is_missing_not_a_small_position(self) -> None:
        active = sizing.average_active_bets(pd.DataFrame({"A": [1.0] * 4}), horizon=3)
        assert list(active["A"].isna().to_numpy()) == [True, True, False, False]

    def test_goes_to_zero_when_the_signal_goes_quiet(self) -> None:
        weights = pd.DataFrame({"A": [1.0] + [0.0] * 4})
        active = sizing.average_active_bets(weights, horizon=2)
        assert float(active["A"].iloc[-1]) == 0.0

    def test_rejects_bad_horizon(self) -> None:
        with pytest.raises(ValueError, match="持有期"):
            _ = sizing.average_active_bets(pd.DataFrame({"A": [1.0]}), horizon=0)


class TestQuantizeBetSize:
    def test_snaps_to_the_grid(self) -> None:
        weights = pd.DataFrame({"A": [0.11, 0.29, -0.72]})
        quantized = sizing.quantize_bet_size(weights, step=0.25)
        np.testing.assert_allclose(quantized["A"].to_numpy(), [0.0, 0.25, -0.75])

    def test_zero_step_is_identity(self) -> None:
        weights = pd.DataFrame({"A": [0.111, 0.222]})
        assert sizing.quantize_bet_size(weights, step=0.0) is weights

    def test_reduces_turnover_on_a_jittery_path(self) -> None:
        """连续微调在未量化时全是换手；量化后大部分落回同一格。"""
        rng = np.random.default_rng(0)
        weights = pd.DataFrame({"A": 0.5 + 0.01 * rng.standard_normal(60)})
        raw = sizing.simulate_positions(weights, weights * 0.0, _options(horizon=1))
        snapped = sizing.simulate_positions(
            sizing.quantize_bet_size(weights, step=0.1),
            weights * 0.0,
            _options(horizon=1),
        )
        assert snapped.turnover.sum() < raw.turnover.sum()


class TestSimulatePositions:
    def test_pnl_uses_the_current_bar_position(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        weights = pd.DataFrame({"A": [1.0, -2.0, 0.0]}, index=index)
        returns = pd.DataFrame({"A": [0.10, -0.20, 0.30]}, index=index)
        stats = sizing.simulate_positions(weights, returns, _options(horizon=1))
        np.testing.assert_allclose(stats.gross_returns.to_numpy(), [0.10, 0.40, 0.0])

    def test_changing_a_later_return_does_not_change_an_earlier_pnl(self) -> None:
        """防未来函数：r[t] 只能影响 pnl[t]，不能影响 pnl[t-1]。"""
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        weights = pd.DataFrame({"A": [1.0] * 4}, index=index)
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=index)
        base = sizing.simulate_positions(weights, returns, _options(horizon=1))

        changed = returns.copy()
        changed.loc[index[2], "A"] = 99.0
        after = sizing.simulate_positions(weights, changed, _options(horizon=1))

        assert float(after.gross_returns.iloc[1]) == pytest.approx(
            float(base.gross_returns.iloc[1])
        )
        assert float(after.gross_returns.iloc[2]) == pytest.approx(99.0)

    def test_charges_cost_on_position_changes_only(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        weights = pd.DataFrame({"A": [1.0, -2.0, 0.0]}, index=index)
        returns = pd.DataFrame({"A": [0.10, -0.20, 0.30]}, index=index)
        stats = sizing.simulate_positions(
            weights, returns, _options(horizon=1, cost_one_way=0.01)
        )
        np.testing.assert_allclose(stats.turnover.to_numpy(), [0.0, 3.0, 2.0])
        np.testing.assert_allclose(stats.net_returns.to_numpy(), [0.10, 0.37, -0.02])

    def test_a_steady_position_costs_nothing_after_warmup(self) -> None:
        index = pd.date_range("2024-01-01", periods=6, freq="D")
        weights = pd.DataFrame({"A": [-1.0] * 6}, index=index)
        returns = pd.DataFrame({"A": [0.01] * 6}, index=index)
        stats = sizing.simulate_positions(weights, returns, _options(horizon=3))
        np.testing.assert_allclose(stats.turnover.to_numpy(), 0.0)
        assert stats.exposure == pytest.approx(1.0)

    def test_breakeven_cost_zeroes_the_net_return(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        weights = pd.DataFrame({"A": [1.0, -1.0, 1.0, -1.0, 1.0]}, index=index)
        returns = pd.DataFrame({"A": [0.02, 0.02, 0.02, 0.02, 0.02]}, index=index)
        stats = sizing.simulate_positions(weights, returns, _options(horizon=1))
        at_breakeven = sizing.simulate_positions(
            weights, returns, _options(horizon=1, cost_one_way=stats.breakeven_cost)
        )
        assert at_breakeven.net == pytest.approx(0.0, abs=1e-12)

    def test_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError, match="仓位路径为空"):
            _ = sizing.simulate_positions(
                pd.DataFrame(), pd.DataFrame(), _options(horizon=1)
            )

    def test_rejects_empty_intersection(self) -> None:
        weights = pd.DataFrame({"A": [1.0, 1.0]})
        returns = pd.DataFrame({"B": [0.1, 0.1]})
        with pytest.raises(ValueError, match="没有可用的交集"):
            _ = sizing.simulate_positions(weights, returns, _options(horizon=1))


class TestCompareSchemes:
    @staticmethod
    def _panels(
        periods: int = 40, symbols: int = 4, seed: int = 0
    ) -> tuple[pd.DataFrame, ...]:
        rng = np.random.default_rng(seed)
        days = pd.date_range("2024-01-01", periods=periods, freq="D")
        names = pd.Index([f"S{i}" for i in range(symbols)])
        # 概率均值恰为 0.5：否则「全下」的夏普会被样本均值带走，比较就没有意义
        probability = pd.DataFrame(
            rng.uniform(0.1, 0.9, (periods, symbols)), index=days, columns=names
        )
        side = pd.DataFrame(1.0, index=days, columns=names)
        # 收益与 (概率 - 0.5) 同向：高置信确实对应高收益
        return side, (probability - 0.5) * 0.02, probability

    def test_a_threshold_of_zero_reproduces_the_all_bets_scheme(self) -> None:
        side, forward, probability = self._panels()
        table = sizing.compare_schemes(
            side,
            forward,
            probability,
            options=sizing.SchemeOptions(horizon=1, thresholds=(0.0,)),
        )
        all_on, switch = table.iloc[0], table.iloc[1]
        assert switch["gross_bp"] == pytest.approx(float(all_on["gross_bp"]))
        assert switch["exposure"] == pytest.approx(float(all_on["exposure"]))

    def test_a_threshold_above_every_probability_gives_no_exposure(self) -> None:
        side, forward, probability = self._panels()
        table = sizing.compare_schemes(
            side,
            forward,
            probability,
            options=sizing.SchemeOptions(horizon=1, thresholds=(1.5,)),
        )
        assert float(table.iloc[1]["exposure"]) == pytest.approx(0.0)
        assert float(table.iloc[1]["gross_bp"]) == pytest.approx(0.0)

    def test_sizing_beats_equal_sized_bets_when_confidence_carries_information(
        self,
    ) -> None:
        """Ch 10 的核心论断，做成可证伪的断言：概率真含信息时，按它缩放胜过等权。

        构造数据里收益与 (p - 0.5) 同向。等权下正负抵消（均值≈0）；按概率定仓会把
        低置信那半边压到 0，于是每期都非负——夏普必然更高。
        """
        side, forward, probability = self._panels()
        table = sizing.compare_schemes(
            side,
            forward,
            probability,
            options=sizing.SchemeOptions(horizon=1, thresholds=(0.6,)),
        )
        sized = table.loc[table["scheme"] == "概率定仓"].iloc[0]
        assert float(sized["gross_sharpe"]) > float(table.iloc[0]["gross_sharpe"])
        assert float(sized["gross_sharpe"]) > 0.0
        # 定仓的暴露天然更小——所以比较夏普而不是比较净收益
        assert float(sized["exposure"]) < float(table.iloc[0]["exposure"])

    def test_a_zero_grid_makes_the_quantized_scheme_identical(self) -> None:
        """网格设成 0 就是不定量，最后一行必须与「概率定仓」逐字段相同。"""
        side, forward, probability = self._panels()
        table = sizing.compare_schemes(
            side,
            forward,
            probability,
            options=sizing.SchemeOptions(horizon=1, grid=0.0),
        )
        sized = table.loc[table["scheme"] == "概率定仓"].iloc[0]
        gridded = table.loc[table["scheme"].str.startswith("概率定仓 + 网格")].iloc[0]
        assert float(gridded["gross_bp"]) == pytest.approx(float(sized["gross_bp"]))
        assert float(gridded["net_bp"]) == pytest.approx(float(sized["net_bp"]))
        assert float(gridded["turnover"]) == pytest.approx(float(sized["turnover"]))
        assert float(gridded["exposure"]) == pytest.approx(float(sized["exposure"]))
