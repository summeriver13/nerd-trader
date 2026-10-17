"""阈值市场公式层：one-touch 公式的正确性由「解析恒等式 + QuantLib PDE 对拍」双重钉住。

历史教训：反射原理公式多发一个 `+s/2` 项时，dip 阶梯会被全部算成 100¢ 而看不出错，
因为阶梯单调性依然成立、量级却全错；所以本文件既测形状（单调、极限），也测量级
（QuantLib 的独立数值解）。对拍期限由日期算出，手写年分数会凭空造出 2e-3 的假偏差。
"""

from __future__ import annotations

import math

import pytest

from src.research.polymarket.digital_pricing import (
    GbmContext,
    implied_sigma,
    normal_cdf,
    one_touch,
    terminal_digital,
)

QD = pytest.importorskip("QuantLib", reason="QuantLib 仅作为 dev 依赖做公式对拍")

SPOT = 81915.0
TODAY = QD.Date(14, 10, 2026)
EXPIRY = QD.Date(1, 1, 2027)
HORIZON = QD.Actual365Fixed().yearFraction(TODAY, EXPIRY)


def _quantlib_one_touch(strike: float, sigma: float, drift: float) -> float:
    """用 QuantLib 的 PDE 引擎独立求同一 GBM 下的一触碰概率。

    r=0 时「触碰即付」与「到期付」等价，故 `CashOrNothingPayoff` 的 NPV 就是触碰概率。
    q 的取值使对数漂移恰为 drift：nu = -q - sigma^2/2。
    """
    QD.Settings.instance().evaluationDate = TODAY
    day_count = QD.Actual365Fixed()
    dividend = -drift - sigma * sigma / 2.0
    process = QD.BlackScholesMertonProcess(
        QD.QuoteHandle(QD.SimpleQuote(SPOT)),
        QD.YieldTermStructureHandle(
            QD.FlatForward(TODAY, QD.QuoteHandle(QD.SimpleQuote(dividend)), day_count)
        ),
        QD.YieldTermStructureHandle(
            QD.FlatForward(TODAY, QD.QuoteHandle(QD.SimpleQuote(0.0)), day_count)
        ),
        QD.BlackVolTermStructureHandle(
            QD.BlackConstantVol(
                TODAY,
                QD.NullCalendar(),
                QD.QuoteHandle(QD.SimpleQuote(sigma)),
                day_count,
            )
        ),
    )
    upward = strike > SPOT
    payoff = (
        QD.CashOrNothingPayoff(QD.Option.Call, 0.5, 1.0)
        if upward
        else QD.CashOrNothingPayoff(QD.Option.Put, 10.0 * SPOT, 1.0)
    )
    option = QD.BarrierOption(
        QD.Barrier.UpIn if upward else QD.Barrier.DownIn,
        strike,
        0.0,
        payoff,
        QD.EuropeanExercise(EXPIRY),
    )
    option.setPricingEngine(QD.FdBlackScholesBarrierEngine(process, 400, 800, 0))
    return float(option.NPV())


def test_zero_drift_reduces_to_reflection_identity() -> None:
    """零漂移时 one-touch 退化为 `2 * Phi(-b / s)`，这是反射原理的手算锚点。"""
    context = GbmContext(spot=SPOT, drift=0.0, horizon=HORIZON)
    for strike, sigma in ((90000.0, 0.38), (120000.0, 0.5), (50000.0, 0.6)):
        barrier = math.log(strike / SPOT)
        scale = sigma * math.sqrt(HORIZON)
        assert one_touch(context, strike, sigma) == pytest.approx(
            2.0 * normal_cdf(-abs(barrier) / scale), rel=1e-12
        )


def test_strike_at_spot_counts_as_touched() -> None:
    """行权价等于现货即为已触碰，返回 1.0（不能落进上行/下行分支除零）。"""
    context = GbmContext(spot=SPOT, drift=-0.2, horizon=HORIZON)
    assert one_touch(context, SPOT, 0.4) == 1.0


def test_touch_is_above_terminal_digital() -> None:
    """触碰概率恒不低于同口径末次结算概率（事件包含），且近价档差近一半。"""
    context = GbmContext(spot=SPOT, drift=-0.003, horizon=HORIZON)
    for strike in (90000.0, 110000.0, 150000.0):
        assert one_touch(context, strike, 0.38) > terminal_digital(
            context, strike, 0.38
        )
    for strike in (70000.0, 55000.0, 45000.0):
        assert one_touch(context, strike, 0.38) > terminal_digital(
            context, strike, 0.38, below=True
        )
    touch = one_touch(context, 90000.0, 0.38)
    assert touch / terminal_digital(context, 90000.0, 0.38) == pytest.approx(
        1.99, abs=0.05
    )


def test_touch_is_monotone_decreasing_in_strike() -> None:
    """阶梯单调性（无模型无关套利）：每条阶梯内部触碰概率随行权价递减。

    上行档与下行档是不同事件（`P(max >= K)` vs `P(min <= K)`），跨现货不可比，
    因此只能分别验单调：reach 阶梯随 K 降、dip 阶梯随 K 升（即随 K 降而降）。
    """
    context = GbmContext(spot=SPOT, drift=-0.1, horizon=HORIZON)
    reach = [
        one_touch(context, float(k), 0.45) for k in (90000, 120000, 150000, 250000)
    ]
    dip = [one_touch(context, float(k), 0.45) for k in (70000, 55000, 45000, 25000)]
    assert reach == sorted(reach, reverse=True)
    assert dip == sorted(dip, reverse=True)
    assert dip[0] < reach[0]  # 14% 的下行触碰比 +10% 的上行触碰难


def test_implied_sigma_roundtrips_price() -> None:
    """价格 → 隐含波动率 → 价格 必须闭环，否则回解口径与正算口径不一致。"""
    context = GbmContext(spot=SPOT, drift=-0.1, horizon=HORIZON)
    for strike in (90000.0, 120000.0, 55000.0, 40000.0):
        for sigma in (0.25, 0.45, 0.9):
            price = one_touch(context, strike, sigma)
            recovered = implied_sigma(context, strike, price)
            assert one_touch(context, strike, recovered) == pytest.approx(
                price, abs=1e-9
            )


def test_zero_volatility_limits() -> None:
    """波动率趋零时路径变成确定轨道：够不到则 0，够得到则 1。"""
    flat = GbmContext(spot=SPOT, drift=0.0, horizon=HORIZON)
    assert one_touch(flat, 2.0 * SPOT, 0.01) == pytest.approx(0.0, abs=1e-12)
    assert one_touch(flat, 0.5 * SPOT, 0.01) == pytest.approx(0.0, abs=1e-12)
    up = GbmContext(spot=SPOT, drift=5.0, horizon=HORIZON)
    assert one_touch(up, 1.1 * SPOT, 0.01) == pytest.approx(1.0, rel=1e-9)
    down = GbmContext(spot=SPOT, drift=-5.0, horizon=HORIZON)
    assert one_touch(down, 0.9 * SPOT, 0.01) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize(
    ("strike", "sigma", "drift"),
    [
        (90000.0, 0.38, -0.003),
        (120000.0, 0.46, -0.46),
        (150000.0, 0.50, -0.10),
        (55000.0, 0.50, -0.10),
        (45000.0, 0.40, -0.10),
        (70000.0, 0.35, 0.20),
    ],
)
def test_quantlib_fd_crosscheck(strike: float, sigma: float, drift: float) -> None:
    """与 QuantLib 的 PDE 解对拍；实测最大偏差 2.3e-4（FD 逐点略低），容差留到 1e-3。"""
    context = GbmContext(spot=SPOT, drift=drift, horizon=HORIZON)
    assert one_touch(context, strike, sigma) == pytest.approx(
        _quantlib_one_touch(strike, sigma, drift), abs=1e-3
    )
