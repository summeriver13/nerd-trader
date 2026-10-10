"""分位组合的口径：价差与成本公式、只多腿、复利年化、市场基准与 BTC。

历史教训：成本若与换手口径不一致，扣出来的净收益就没有意义；算术年化会被波动
拖累高估实际收益；不比基准就分不清因子与市场 beta；span 用面板跨度而非组合实际
投资期，会把每年期数算少、所有年化跟着偏低。这些都用合成数据钉死。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.research.factors.library import FactorSpec
from src.research.factors.panel import Panel
from src.research.factors.portfolio import PortfolioOptions, leg_stats, portfolio_stats

QUANTILES = 4
STEP = 0.01  # 第 i 个标的每根收益 = STEP × i


def _alternating_spec(panel: Panel) -> FactorSpec:
    """每期排名翻转的因子：成分全换，用于钉住换手与成本公式。"""
    frame = pd.DataFrame(
        {
            name: [
                float(index) * (1 if bar % 2 == 0 else -1)
                for bar in range(len(panel.index))
            ]
            for index, name in enumerate(panel.symbols)
        },
        index=panel.index,
    )
    return FactorSpec(
        key="flip",
        label="翻转因子",
        family="测试",
        direction=1,
        warmup=0,
        compute=lambda _: frame,
    )


def test_quantile_spread_matches_hand_calculation(
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """价差 = 最高分位平均收益 - 最低分位平均收益；成分不变时换手为 0。"""
    panel, rates = ranked_panel
    stats = portfolio_stats(
        rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )
    names = len(panel.symbols)
    expected = float(
        np.mean(rates[-names // QUANTILES :]) - np.mean(rates[: names // QUANTILES])
    )

    assert stats.gross_spread == pytest.approx(expected)
    assert stats.net_spread == pytest.approx(expected)
    assert stats.turnover == pytest.approx(0.0)
    assert stats.periods == len(panel.index) - 1
    assert stats.quantile_returns.is_monotonic_increasing


def test_turnover_and_cost_formula(
    ranked_panel: tuple[Panel, list[float]],
) -> None:
    """成分每期全换：单腿换手 1，成本 = 4 × 换手 × 单边费率。"""
    panel, _ = ranked_panel
    cost_one_way = 0.001
    stats = portfolio_stats(
        _alternating_spec(panel),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=cost_one_way),
    )

    assert stats.turnover == pytest.approx(1.0)
    assert stats.net_spread == pytest.approx(stats.gross_spread - 4 * cost_one_way)


def test_long_only_net_equals_top_quantile_when_no_turnover(
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """成分不变时只多腿没有换手成本，等于最高分位的毛收益。"""
    panel, _ = ranked_panel
    stats = portfolio_stats(
        rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.001),
    )

    assert stats.long_turnover == pytest.approx(0.0)
    assert stats.long_only_net == pytest.approx(float(stats.quantile_returns.iloc[-1]))
    assert stats.long_only_annual == pytest.approx(
        stats.long_only_net * stats.periods / stats.span_years
    )


def test_long_only_leg_uses_its_own_turnover(
    ranked_panel: tuple[Panel, list[float]],
) -> None:
    """只多腿成本 = 2ρ（不是多空价差的 4ρ）：用错会把成本算高一倍。"""
    panel, _ = ranked_panel
    cost_one_way = 0.001
    stats = portfolio_stats(
        _alternating_spec(panel),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=cost_one_way),
    )

    assert stats.long_turnover == pytest.approx(1.0)
    assert stats.short_turnover == pytest.approx(1.0)
    assert stats.turnover == pytest.approx(1.0)  # 两腿均值
    assert stats.long_only_net == pytest.approx(
        float(stats.quantile_returns.iloc[-1]) - 2 * cost_one_way
    )


def test_annualization_uses_realized_periods(
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """年化按期数简单折算（spread 不是可复利曲线），期数少时不外推。"""
    panel, _ = ranked_panel
    stats = portfolio_stats(
        rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )

    assert stats.span_years > 0
    assert stats.annual_gross == pytest.approx(
        stats.gross_spread * stats.periods / stats.span_years
    )


def test_span_years_covers_portfolio_not_panel(
    panel_factory: Callable[[dict[str, list[float]]], Panel],
) -> None:
    """span_years 是组合的实际投资期，不是面板跨度。

    面板前半段没有可交易标的时不能算进去，否则每年期数被低估、所有年化跟着偏低。
    """
    closes = {
        f"N{index:02d}-USDT": [100.0 * (1 + STEP * index) ** bar for bar in range(12)]
        for index in range(20)
    }
    panel = panel_factory(closes)
    late = FactorSpec(
        key="late",
        label="延迟因子",
        family="测试",
        direction=1,
        warmup=8,
        compute=lambda p: pd.DataFrame(
            {name: float(index) for index, name in enumerate(p.symbols)},
            index=p.index,
        ),
    )
    stats = portfolio_stats(
        late,
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )

    assert stats.periods == 3  # 第 8、9、10 根有前向收益，第 11 根没有
    assert stats.span_years == pytest.approx(3 / 365.25, rel=1e-3)


def test_leg_stats_compounded_below_arithmetic_when_volatile() -> None:
    """复利年化必须低于算术折算：差就是波动拖累，高波动腿上可以到十几百分点。"""
    returns = pd.Series([0.5, -0.3] * 6)
    stats = leg_stats(returns, periods_per_year=12.0)

    assert stats.annual_compounded < float(returns.mean()) * 12.0
    assert stats.volatility > 0.0
    assert stats.max_drawdown < 0.0


def test_leg_stats_drawdown_from_compounded_curve() -> None:
    """最大回撤按复利曲线算：+100% 再 -50% 回到原点，回撤是 -50% 不是 -100%。"""
    stats = leg_stats(pd.Series([1.0, -0.5]), periods_per_year=2.0)

    assert stats.max_drawdown == pytest.approx(-0.5)
    assert stats.annual_compounded == pytest.approx(0.0)


def test_market_leg_is_equal_weight_all_names(
    ranked_panel: tuple[Panel, list[float]],
    rank_spec: Callable[..., FactorSpec],
) -> None:
    """市场基准 = 每期「等权全体」的平均前向收益：不减它就分不清因子与 beta。"""
    panel, rates = ranked_panel
    stats = portfolio_stats(
        rank_spec(),
        panel,
        panel.close.notna(),
        PortfolioOptions(horizon=1, quantiles=QUANTILES, cost_one_way=0.0),
    )

    assert stats.market_returns.size == stats.periods
    assert float(stats.market_returns.mean()) == pytest.approx(float(np.mean(rates)))
    # 最高分位收益高于全体均值，超额必须为正
    assert stats.excess_compounded > 0.0
