"""数字期权与一触碰（one-touch）定价：阈值型预测市场的公式层。

Polymarket 的「will-bitcoin-reach-K」是**触碰即结算**的单边障碍（one-touch），
不是末次结算数字期权 `N(d2)`：同一张 18 档阶梯用 `N(d2)` 拟合 RMS 45.0¢，
用本模块的 `one_touch()` 拟合 RMS 2.1¢——公式族选错的代价比价格本身还大。

漂移约定：`drift` 是**年化对数漂移** nu，即 `ln(S_T/S0) ~ N(nu*T, sigma^2*T)`。
风险中性下 `nu = r - q - sigma^2/2`；用远期表达时 `nu = ln(F/S0)/T - sigma^2/2`。

历史教训：
- 反射原理公式多发一个 `+s/2` 项（`s = sigma*sqrt(T)`）会把 dip 阶梯全算成 100¢；
- QuantLib 1.44 的 `AnalyticBinaryBarrierEngine` 在 Python 层对任何 payoff/障碍组合
  都抛 "payoff must be at expiry"，无法解析对拍；只能用 `FdBlackScholesBarrierEngine`
  （PDE）；两者最大偏差 2.3e-4（0.02¢，逐点同号，属 FD 自身偏置）；
- 对拍时期限必须由日期算出（`Actual365Fixed().yearFraction`）：手写 78/365 而日历是 79 天，
  会在公式正确的前提下凭空造出 2e-3 的“偏差”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_SIGMA = 0.005
MAX_SIGMA = 6.0
SIGMA_TOLERANCE = 1e-10
SIGMA_ITERATIONS = 100
EXPONENT_CAP = 300.0  # 反射项指数的封顶，防小波动率下溢出


@dataclass(frozen=True)
class GbmContext:
    """GBM 定价环境：现货、年化对数漂移（nu）与剩余期限（年）。"""

    spot: float
    drift: float
    horizon: float


def normal_cdf(value: float) -> float:
    """标准正态分布函数。"""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def terminal_digital(
    context: GbmContext, strike: float, sigma: float, *, below: bool = False
) -> float:
    """末次结算数字期权价格：默认 `P(S_T > strike)`（call 口径），below=True 给 `P(S_T < strike)`。

    下行阶梯必须用 below=True：K < 现货时 call 口径的数字价趋近 1，与一触碰概率不同量级。
    """
    scale = sigma * math.sqrt(context.horizon)
    above = normal_cdf(
        (context.drift * context.horizon - math.log(strike / context.spot)) / scale
    )
    return 1.0 - above if below else above


def one_touch(context: GbmContext, strike: float, sigma: float) -> float:
    """触碰即结算概率：`P(max S >= K)`（K > 现货）或 `P(min S <= K)`（K < 现货）。

    K 等于现货表示已在障碍上，返回 1.0。结果已裁剪到 [0, 1]。
    """
    if strike == context.spot:
        return 1.0
    barrier = math.log(strike / context.spot)
    scale = sigma * math.sqrt(context.horizon)
    drift_term = context.drift * context.horizon
    exponent = 2.0 * context.drift * barrier / (sigma * sigma)
    # 反射项先算指数，再用正态项收敛掉它，避免小 sigma 下溢出
    reflected = math.exp(min(EXPONENT_CAP, exponent)) * normal_cdf(
        (-barrier - drift_term) / scale
        if barrier > 0
        else (barrier + drift_term) / scale
    )
    direct = normal_cdf(
        (-barrier + drift_term) / scale
        if barrier > 0
        else (barrier - drift_term) / scale
    )
    return min(1.0, max(0.0, direct + reflected))


def implied_sigma(
    context: GbmContext,
    strike: float,
    price: float,
    *,
    low: float = MIN_SIGMA,
    high: float = MAX_SIGMA,
    tolerance: float = SIGMA_TOLERANCE,
    max_iterations: int = SIGMA_ITERATIONS,
) -> float:
    """由触碰价格反解隐含波动率（触碰概率对 sigma 单调，二分即可）。"""
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        if one_touch(context, strike, middle) < price:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return 0.5 * (low + high)
