"""
tests/research/test_pairs.py
配对研究的口径测试：只测会「静默算错」的地方——选对是否真的拒掉随机漫步、
z 的统计量是否只用形成期、两腿的账是否与手算一致、成本与资金费的符号对不对。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.research.factors.panel import Panel, UniverseRule
from src.research.pairs.backtest import (
    PairsOptions,
    _positions,
    _trade_pair,
    daily_at_cost,
    run_pairs,
)
from src.research.pairs.funding import funding_cost
from src.research.pairs.selection import (
    Pair,
    SelectionOptions,
    fit_spread,
    half_life_of,
    random_pairs,
    select_pairs,
)

START = "2022-01-01"
SIGMA = 0.1


def _frame(values: dict[str, list[float]], start: str = START) -> pd.DataFrame:
    """按列构造宽表（每个标的等长）。"""
    index = pd.date_range(
        start, periods=len(next(iter(values.values()))), freq="1D", tz="UTC"
    )
    return pd.DataFrame(values, index=index)


def _panel(values: dict[str, list[float]]) -> Panel:
    """合成面板：收盘价即全部输入（volume/notional 只为池子规则存在）。"""
    close = _frame(values)
    return Panel(
        timeframe="1d",
        close=close,
        volume=close * 0.0 + 1.0,
        notional=close * 0.0 + 1e9,
    )


def _cointegrated_panel(bars: int = 240, seed: int = 3) -> Panel:
    """20 只随机漫步 + 一对协整（A = 1 + 1.2·B + AR(1) 噪声）。"""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.03, bars))
    values: dict[str, list[float]] = {}
    for i in range(20):
        walk = 20.0 + np.cumsum(rng.normal(0, 0.02, bars))
        values[f"W{i:02d}-USDT"] = np.exp(walk).tolist()
    log_b = 20.0 + common
    noise = np.zeros(bars)
    for i in range(1, bars):
        noise[i] = 0.93 * noise[i - 1] + rng.normal(0, 0.01)
    values["AAA-USDT"] = np.exp(1.0 + 1.2 * log_b + noise).tolist()
    values["BBB-USDT"] = np.exp(log_b).tolist()
    return _panel(values)


def _log_close(panel: Panel) -> pd.DataFrame:
    """对数价格宽表（np.log 的返回类型在 stubs 里是 ndarray，这里显式收敛成 DataFrame）。"""
    return cast(pd.DataFrame, np.log(panel.close))


def _test_rule() -> UniverseRule:
    """测试用的池子规则：合成面板只有几十根，默认的「上市 ≥180 天」会把它们全部剔掉。"""
    return UniverseRule(min_notional_usd=0.0, min_listed_days=0, liquidity_window=1)


def _pair_panel(bars: int = 60, formation: int = 40) -> Panel:
    """两标的、确定性价差路径：形成期小幅振荡，交易期挖一个约 −2.8σ 的坑再回来。"""
    rng = np.random.default_rng(7)
    log_b = 10.0 + np.cumsum(rng.normal(0, 0.02, bars))
    spread = 0.02 * np.sin(np.arange(bars) * 0.7)
    spread[formation : formation + 10] = np.linspace(0.02, -0.04, 10)
    spread[formation + 10 :] = np.linspace(-0.04, 0.0, bars - formation - 10)
    return _panel(
        {
            "AAA-USDT": np.exp(1.0 + log_b + spread).tolist(),
            "BBB-USDT": np.exp(log_b).tolist(),
        }
    )


def _pair(**overrides: float) -> Pair:
    """测试用的对子：β=1、价差均值 0、标准差 SIGMA。"""
    base: dict[str, float] = {
        "alpha": 0.0,
        "beta": 1.0,
        "corr": 0.9,
        "adf_p": 0.01,
        "half_life": 5.0,
        "spread_mean": 0.0,
        "spread_std": SIGMA,
    }
    base.update(overrides)
    return Pair(left="AAA-USDT", right="BBB-USDT", **base)


# === 选对：统计口径 ===


def test_fit_spread_recovers_alpha_and_beta() -> None:
    """OLS 必须还原真实参数：logA = 1 + 1.2·logB + 噪声。

    logB 从 0 附近起步：截距的识别误差会被 ``β 的误差 × x̄`` 放大，
    logB 量级在 20 时截距差 0.1 都算正常（不是估计器有问题）。
    """
    rng = np.random.default_rng(11)
    log_b = pd.Series(np.cumsum(rng.normal(0, 0.03, 300)))
    log_a = 1.0 + 1.2 * log_b + pd.Series(rng.normal(0, 0.01, 300))
    alpha, beta, spread = fit_spread(log_a, log_b)
    assert alpha == pytest.approx(1.0, abs=0.01)
    assert beta == pytest.approx(1.2, abs=0.01)
    assert float(spread.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(spread.std(ddof=1)) == pytest.approx(0.01, rel=0.15)


def test_half_life_matches_ar1_process() -> None:
    """AR(1) 的 λ 决定半衰期：0.93 对应约 9.5 天，估计不能差一个量级。"""
    rng = np.random.default_rng(5)
    values = np.zeros(2000)
    for i in range(1, values.size):
        values[i] = 0.93 * values[i - 1] + rng.normal(0, 0.01)
    half_life = half_life_of(pd.Series(values))
    assert half_life == pytest.approx(-np.log(2.0) / np.log(0.93), rel=0.2)


def test_half_life_rejects_random_walk() -> None:
    """随机漫步必须被上限拦掉：AR(1) 在有限样本上会给出一个有限但很大的半衰期。"""
    rng = np.random.default_rng(6)
    walk = pd.Series(np.cumsum(rng.normal(0, 0.02, 500)))
    half_life = half_life_of(walk)
    assert half_life == float("inf") or half_life > 30.0


def test_select_pairs_picks_cointegrated_pair_only() -> None:
    """随机漫步池里必须只挑出那一对协整：ADF 不是摆设。"""
    panel = _cointegrated_panel()
    mask = pd.DataFrame(True, index=panel.index, columns=panel.symbols)
    pairs = select_pairs(_log_close(panel), mask, 200, SelectionOptions(min_names=20))
    assert len(pairs) == 1
    assert {pairs[0].left, pairs[0].right} == {"AAA-USDT", "BBB-USDT"}
    assert pairs[0].adf_p < 0.05


def test_select_pairs_respects_pool_size_and_dedupe() -> None:
    """池子太小不选；选出来的对子里每个标的最多出现一次。"""
    panel = _cointegrated_panel()
    mask = pd.DataFrame(True, index=panel.index, columns=panel.symbols)
    assert (
        select_pairs(_log_close(panel), mask, 200, SelectionOptions(min_names=25)) == []
    )
    picked = select_pairs(_log_close(panel), mask, 200, SelectionOptions(min_names=20))
    symbols = [symbol for pair in picked for symbol in (pair.left, pair.right)]
    assert len(symbols) == len(set(symbols))


def test_random_pairs_skips_cointegration_filter() -> None:
    """随机对照必须**不查 ADF**：否则它就不是安慰剂，而是另一套筛选。"""
    panel = _cointegrated_panel()
    mask = pd.DataFrame(True, index=panel.index, columns=panel.symbols)
    options = SelectionOptions(min_names=20, top_pairs=3)
    picked = random_pairs(_log_close(panel), mask, 200, options, seed=1)
    assert picked, "随机选对应能选出对子"
    assert len(picked) <= options.top_pairs
    symbols = [symbol for pair in picked for symbol in (pair.left, pair.right)]
    assert len(symbols) == len(set(symbols))
    assert all(pair.adf_p > 0.05 for pair in picked), "随机对子不该普遍通过协整检验"


# === 交易：状态机与账 ===


def test_run_pairs_end_to_end_opens_and_closes() -> None:
    """端到端：选对 → 交易期建仓 → 收敛后平仓，组合与明细都要有产物。

    这条测试用的是一套宽松的选对口径（关掉 ADF/半衰期/相关性），因为要验的是
    「窗口滚动 → 选对 → 两腿成交 → 组合等权」这条链是否接通；筛选本身另有测试。
    """
    panel = _pair_panel()
    options = PairsOptions(
        selection=SelectionOptions(
            formation_bars=40,
            min_names=2,
            top_pairs=1,
            min_corr=0.0,
            max_adf_p=1.0,
            min_half_life=0.0,
            max_half_life=1e6,
        ),
        trading_bars=20,
        step_bars=20,
    )
    run = run_pairs(panel, options, rule=_test_rule())
    assert run.windows == 1
    assert len(run.trades) == 1
    assert run.rebalances >= 2, "价差挖坑又回归，至少一进一出"
    assert len(run.daily) == 20
    assert set(run.trades["left"]) == {"AAA-USDT"}
    assert float(run.trades["bars_held"].iloc[0]) > 0


def test_positions_enter_exit_and_stop() -> None:
    """z 阈值语义：±2σ 进、回到 ±0.5σ 内平仓（多头与空头两侧都要验）。"""
    options = PairsOptions(entry_z=2.0, exit_z=0.5, stop_z=4.0)
    z = pd.Series([0.0, -2.5, -1.0, 0.0, 3.0, -3.0, 0.0])
    pos, stopped = _positions(z, options)
    assert list(pos) == [0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    assert not stopped


def test_positions_stop_blocks_further_entries() -> None:
    """破 4σ 止损后本段交易期不再进场：关系断了，再进就是赌它回来。"""
    options = PairsOptions(entry_z=2.0, exit_z=0.5, stop_z=4.0)
    z = pd.Series([0.0, -2.5, -4.5, 0.0, 3.0, -2.5, 0.0])
    pos, stopped = _positions(z, options)
    assert list(pos) == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert stopped


def test_positions_keep_state_through_nan() -> None:
    """数据缺口维持原仓：既不开新仓，也不假装平价离场。"""
    options = PairsOptions()
    z = pd.Series([0.0, -2.5, float("nan"), -2.4, 0.0])
    pos, _ = _positions(z, options)
    assert list(pos) == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_trade_pair_accounts_two_legs_without_lookahead() -> None:
    """两腿的账要能手算，且信号下一根才生效（否则是拿当根价格换仓）。

    z 序列（σ=0.1）：0 → 0 → −2.5 → −1.0 → 0。
    - 第 3 根：z 冲到 −2.5 那根**不吃**（执行在下一根），吃到的是 −2.5σ→−1.0σ
      的价差收敛：``0.5 × (exp(−0.1)/exp(−0.25) − 1)``；
    - 第 4 根：z 回到 0 但目标仓在第 5 根才执行，仓位仍持有，继续吃涨；
    - 换手只有一次（开仓）；出场信号落在窗口外，不该凭空计费。
    """
    panel = _panel(
        {
            "AAA-USDT": [
                float(np.exp(value * SIGMA)) for value in (0.0, 0.0, -2.5, -1.0, 0.0)
            ],
            "BBB-USDT": [1.0] * 5,
        }
    )
    options = PairsOptions(entry_z=2.0, exit_z=0.5, cost_one_way=0.001)
    frame, record = _trade_pair(
        _pair(),
        _log_close(panel),
        panel.close.pct_change(fill_method=None),
        0,
        5,
        options,
        None,
    )
    assert float(frame["gross"].iloc[2]) == pytest.approx(0.0, abs=1e-12)
    convergence = 0.5 * (float(np.exp(-0.1)) / float(np.exp(-0.25)) - 1.0)
    assert float(frame["gross"].iloc[3]) == pytest.approx(convergence, rel=1e-9)
    assert float(frame["gross"].iloc[4]) == pytest.approx(
        0.5 * (1.0 / float(np.exp(-0.1)) - 1.0), rel=1e-9
    )
    assert float(frame["turnover"].sum()) == pytest.approx(1.0)
    assert int(cast(int, record["rebalances"])) == 1
    assert float(cast(float, record["cost_sum"])) == pytest.approx(0.001)


def test_daily_at_cost_is_linear_in_turnover() -> None:
    """成本只在换手上线性缩放：0 成本 = 毛收益，双倍成本 = 双倍拖累。"""
    contributions = pd.DataFrame(
        {
            "pair": ["A|B@x", "C|D@x"],
            "gross": [0.01, -0.02],
            "turnover": [1.0, 0.0],
            "funding": [0.0, 0.0],
            "exposure": [1.0, 0.0],
        },
        index=pd.DatetimeIndex(["2022-01-01", "2022-01-01"]),
    )
    free = daily_at_cost(contributions, 0.0)
    cheap = daily_at_cost(contributions, 0.001)
    pricey = daily_at_cost(contributions, 0.002)
    assert float(free.iloc[0]) == pytest.approx((0.01 - 0.02) / 2)
    assert float(cheap.iloc[0]) == pytest.approx((0.01 - 0.02) / 2 - 0.001 / 2)
    assert float(pricey.iloc[0]) == pytest.approx((0.01 - 0.02) / 2 - 0.002 / 2)


def test_funding_sign_convention() -> None:
    """资金费符号：多腿付正费率、空腿收；两腿费率相同时净额为零。"""
    index = pd.DatetimeIndex(["2022-01-01", "2022-01-02"])
    funding = pd.DataFrame(
        {"AAA-USDT": [0.002, 0.002], "BBB-USDT": [0.002, 0.002]}, index=index
    )
    pair = _pair()
    executed = pd.Series([1.0, 1.0], index=index)
    cost = funding_cost(pair, executed, funding, weight=0.5)
    assert np.allclose(cost, 0.0)
    mixed = pd.DataFrame(
        {"AAA-USDT": [0.002, 0.0], "BBB-USDT": [0.001, 0.0]}, index=index
    )
    cost = funding_cost(pair, executed, mixed, weight=0.5)
    assert float(cost[0]) == pytest.approx(0.5 * 0.002 - 0.5 * 1.0 * 0.001)
    short = pd.Series([-1.0, -1.0], index=index)
    cost = funding_cost(pair, short, mixed, weight=0.5)
    assert float(cost[0]) == pytest.approx(-0.5 * 0.002 + 0.5 * 1.0 * 0.001)


def test_idle_pairs_dilute_the_portfolio() -> None:
    """空仓的对也占资金：组合收益是「当期选中对数」的等权平均，不是持仓对的平均。"""
    contributions = pd.DataFrame(
        {
            "pair": ["A|B@x", "C|D@x"],
            "gross": [0.10, 0.0],
            "turnover": [0.0, 0.0],
            "funding": [0.0, 0.0],
            "exposure": [1.0, 0.0],
        },
        index=pd.DatetimeIndex(["2022-01-01", "2022-01-01"]),
    )
    daily = daily_at_cost(contributions, 0.0)
    assert float(daily.iloc[0]) == pytest.approx(0.05)
