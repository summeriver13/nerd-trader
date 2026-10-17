"""阈值市场三口径对照：市场价 / Deribit 风险中性 / 自持历史 bootstrap。

三个口径回答三个不同问题，缺一不可：

- **市场价**：交易者按什么价格成交（含点差与彩票溢价）；
- **Deribit**：同标的、同期限、同为「触碰即结算」的风险中性价，且是能真实对冲的一侧；
- **历史 bootstrap**：物理测度下同样行情再来一次的触碰频率（去漂移/含漂移两档）。

阶梯的装载与口径自检在 `polymarket_ladder.py`；本模块只做定价与对照。

价格语义已核实（docs/polymarket/README.md §七）：判定源是 Binance 的 **1 分钟 High/Low**、
`immediately resolve` 即触碰即结算，所以用 `digital_pricing.one_touch()`；离散监控的连续化
修正只有 0.12¢，忽略。窗口起点是各市场自己的创建时刻，但只要「至今未触碰」，马尔可夫性
就使条件分布等于从当前现货重新出发——定价只需现货与剩余期限，不需要路径条件修正。

判读口径：Polymarket 的 touch 与 Deribit 的到期结算**不同源**（BTC 跌破再反弹时一边赔 100
一边归零），所以两者之差是**相对价值**不是无风险套利；Deribit 一侧的 touch 是「逐档 IV
一阶近似」，误差带由微笑区间两端给出（`deribit_touch_low/high`）——只有市场价高出带上沿
（`beyond_deribit_band`）才算近似误差吃不掉的偏贵。

历史教训：定价锚点必须用 Deribit 指数/永续，不能用滞后十几小时的 Binance 1h 收盘
（1.8% 的锚点误差足以让近价区的隐含波动率结论完全失真）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.research.polymarket.digital_pricing import GbmContext, implied_sigma, one_touch
from src.research.polymarket.polymarket_ladder import LadderRow
from src.research.polymarket.price_bootstrap import PathStats
from src.utils.data_sources.deribit import DeribitChain

INITIAL_DRIFT = 0.4
DRIFT_ITERATIONS = 40
DRIFT_TOLERANCE = 1e-12

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Smile:
    """某一到期的波动率微笑：按 log-moneyness 线性插值（下方用 put、上方用 call）。

    同时带着虚值档的买卖价：卖尾要付的对冲成本就在这一侧，不列出来会高估可交易性。
    """

    forward: float
    strikes: FloatArray
    vols: FloatArray
    bids: FloatArray
    asks: FloatArray

    def iv_at(self, strike: float) -> float | None:
        """插值取该档隐含波动率；超出报价区间返回 None（不外推）。"""
        if self.strikes.size == 0 or not self.strikes[0] <= strike <= self.strikes[-1]:
            return None
        moneys = np.log(self.strikes / self.forward)
        return float(np.interp(math.log(strike / self.forward), moneys, self.vols))

    def iv_range(self, low: float, high: float) -> tuple[float, float]:
        """区间内波动率的上下沿——「逐档 IV 一阶近似」的误差带就取这个。"""
        mask = (self.strikes >= min(low, high)) & (self.strikes <= max(low, high))
        selected = self.vols[mask] if mask.any() else self.vols
        return float(selected.min()), float(selected.max())

    def relative_spread_at(self, strike: float) -> float | None:
        """离该档最近的虚值期权相对买卖价差（(ask−bid)/mid）；无有效报价返回 None。"""
        if self.strikes.size == 0:
            return None
        index = int(np.argmin(np.abs(self.strikes - strike)))
        bid, ask = float(self.bids[index]), float(self.asks[index])
        if bid <= 0.0 or ask < bid:
            return None
        return (ask - bid) / (0.5 * (ask + bid))


@dataclass(frozen=True)
class ComparisonRow:
    """三口径逐档对照结果，字段名即 CSV 表头。"""

    slug: str
    strike: float
    direction: str
    market_price: float
    market_iv: float | None
    deribit_touch: float | None
    deribit_touch_low: float | None
    deribit_touch_high: float | None
    deribit_iv: float | None
    market_relative_spread: float | None
    deribit_option_spread: float | None
    bootstrap_zero_drift: float | None
    bootstrap_historical_drift: float | None

    @property
    def diff_vs_deribit(self) -> float | None:
        """市场价减 Deribit 风险中性价（正 = 偏贵；单位是价格，不是 bp）。"""
        if self.deribit_touch is None:
            return None
        return self.market_price - self.deribit_touch

    @property
    def beyond_deribit_band(self) -> bool | None:
        """市场价是否高过 Deribit 一侧的误差带上沿——近似误差吃不掉的偏贵才是真偏贵。"""
        if self.deribit_touch_high is None:
            return None
        return self.market_price > self.deribit_touch_high


def smile_from_chain(chain: DeribitChain, expiry: str, forward: float) -> Smile:
    """用虚值一侧构造微笑：K ≥ F 取 call、K < F 取 put（虚值 IV 受买卖价影响小）。"""
    pairs = [
        (quote.strike, quote.mark_iv, quote.bid or 0.0, quote.ask or 0.0)
        for quote in chain.by_expiry(expiry)
        if quote.mark_iv is not None and quote.kind == ("C" if quote.strike >= forward else "P")
    ]
    ordered = sorted(pairs)
    return Smile(
        forward=forward,
        strikes=np.array([item[0] for item in ordered], dtype=float),
        vols=np.array([item[1] for item in ordered], dtype=float),
        bids=np.array([item[2] for item in ordered], dtype=float),
        asks=np.array([item[3] for item in ordered], dtype=float),
    )


def drift_from_forward(spot: float, forward: float, horizon: float, sigma: float) -> float:
    """风险中性对数漂移：`ln(F/S0)/T − σ²/2`（与 `one_touch` 的约定配套）。"""
    return math.log(forward / spot) / horizon - 0.5 * sigma * sigma


def implied_touch_sigma(
    *, spot: float, forward: float, horizon: float, strike: float, price: float
) -> float:
    """由触碰价格反解隐含 σ；漂移本身依赖 σ，所以用不动点迭代（不迭代会差 2~3pp）。"""
    sigma = implied_sigma(
        GbmContext(spot, drift_from_forward(spot, forward, horizon, INITIAL_DRIFT), horizon),
        strike,
        price,
    )
    for _ in range(DRIFT_ITERATIONS):
        drift = drift_from_forward(spot, forward, horizon, sigma)
        updated = implied_sigma(GbmContext(spot, drift, horizon), strike, price)
        if abs(updated - sigma) < DRIFT_TOLERANCE:
            return updated
        sigma = updated
    return sigma


def deribit_touch(
    *, spot: float, forward: float, horizon: float, strike: float, sigma: float
) -> float:
    """把该档 IV 喂给触碰公式（一阶近似：只用该档波动率水平，不含微笑形状）。"""
    context = GbmContext(spot, drift_from_forward(spot, forward, horizon, sigma), horizon)
    return one_touch(context, strike, sigma)


@dataclass(frozen=True)
class DeribitSide:
    """Deribit 一侧的对照列；该档没有有效报价时全为 None（不插值也不外推）。"""

    touch: float | None = None
    touch_low: float | None = None
    touch_high: float | None = None
    iv: float | None = None
    market_relative_spread: float | None = None
    option_relative_spread: float | None = None


def _deribit_columns(
    row: LadderRow, *, spot: float, horizon: float, smile: Smile
) -> DeribitSide:
    """Deribit 一侧的数列。

    误差带用微笑区间两端各算一次触碰价：障碍价对 σ 单调，故 [低σ价, 高σ价] 就是
    「逐档 IV 一阶近似」的上下沿（该用卡障碍的 σ 还是 ATM 的 σ，单点无法判定）。
    """
    iv = smile.iv_at(row.strike)
    mid = None if row.bid is None or row.ask is None else 0.5 * (row.bid + row.ask)
    market_spread = None if row.spread is None or not mid else row.spread / mid
    option_spread = smile.relative_spread_at(row.strike)
    if iv is None:
        return DeribitSide(
            market_relative_spread=market_spread, option_relative_spread=option_spread
        )
    low, high = smile.iv_range(spot, row.strike)
    keep = {"spot": spot, "forward": smile.forward, "horizon": horizon}
    touch_low = deribit_touch(**keep, strike=row.strike, sigma=low)
    touch_high = deribit_touch(**keep, strike=row.strike, sigma=high)
    return DeribitSide(
        # 中心取区间中点：单点没法知道该用「卡障碍的 σ」还是「ATM 的 σ」，不假装知道
        touch=0.5 * (touch_low + touch_high),
        touch_low=touch_low,
        touch_high=touch_high,
        iv=iv,
        market_relative_spread=market_spread,
        option_relative_spread=option_spread,
    )


def comparison_table(
    rows: Sequence[LadderRow],
    *,
    spot: float,
    horizon: float,
    smile: Smile,
    bootstrap: dict[str, PathStats] | None = None,
) -> list[ComparisonRow]:
    """逐档对照；`bootstrap` 的键约定为 zero / history（去漂移与含历史漂移两档）。"""
    stats = bootstrap or {}
    table: list[ComparisonRow] = []
    for row in rows:
        deribit = _deribit_columns(row, spot=spot, horizon=horizon, smile=smile)
        table.append(
            ComparisonRow(
                slug=row.slug,
                strike=row.strike,
                direction=row.direction,
                market_price=row.price,
                market_iv=_market_iv(row, spot=spot, horizon=horizon, smile=smile),
                deribit_touch=deribit.touch,
                deribit_touch_low=deribit.touch_low,
                deribit_touch_high=deribit.touch_high,
                deribit_iv=deribit.iv,
                market_relative_spread=deribit.market_relative_spread,
                deribit_option_spread=deribit.option_relative_spread,
                bootstrap_zero_drift=_rate(stats.get("zero"), math.log(row.strike / spot)),
                bootstrap_historical_drift=_rate(
                    stats.get("history"), math.log(row.strike / spot)
                ),
            )
        )
    return table


def _market_iv(
    row: LadderRow, *, spot: float, horizon: float, smile: Smile
) -> float | None:
    """市场价反解的隐含 σ；价格正好在 0/1 边界上无法反解，返回 None。"""
    if not 0.0 < row.price < 1.0:
        return None
    return implied_touch_sigma(
        spot=spot, forward=smile.forward, horizon=horizon, strike=row.strike, price=row.price
    )


def _rate(stats: PathStats | None, barrier_log: float) -> float | None:
    """某条障碍位上的历史触碰频率；没有那档 bootstrap 就留空。"""
    return None if stats is None else stats.touch_rate(barrier_log)
