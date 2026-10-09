"""交易成本口径的唯一出处。

研究侧（向量化引擎、maker 挂单执行、carry 研究）与仿真侧（事件驱动网关）都从这里
取费率，避免同一笔成本在不同模块各写一个数、算出来的绩效互不可比。

OKX Regular 档费率（已开衍生品账户；未开衍生品账户的现货是 maker 0.200% /
taker 0.350%），来源 https://www.okx.com/en-eu/help/eea-xperp-spot-trading-fees-comparison：

- 永续：maker 0.020% / taker 0.050%
- 现货：maker 0.080% / taker 0.100%

现货与永续必须分开取值：仿真网关是现货语义（1x 现金账户），拿永续费率会少算一半成本。
滑点只计在市价单上——挂单不吃滑点，代价是可能挂不上（见 execution.py 的实测）。
"""

from dataclasses import dataclass
from typing import Final

PERP_MAKER_FEE: Final[float] = 0.0002
PERP_TAKER_FEE: Final[float] = 0.0005
SPOT_MAKER_FEE: Final[float] = 0.0008
SPOT_TAKER_FEE: Final[float] = 0.0010
MARKET_SLIPPAGE: Final[float] = 0.0005


@dataclass(frozen=True)
class FeeSchedule:
    """单边费率（正 = 成本，负 = 返佣）。maker/taker 无默认值，必须显式选现货或永续。"""

    maker: float
    taker: float
    slippage: float = MARKET_SLIPPAGE

    def maker_net(self, offset: float = 0.0) -> float:
        """被动成交的单边净成本：手续费减去挂单价差带来的价格改善。"""
        return self.maker - offset

    def taker_net(self) -> float:
        """吃价成交的单边净成本：手续费加滑点。"""
        return self.taker + self.slippage

    def net(self, *, crosses: bool, offset: float = 0.0) -> float:
        """单边成本：吃价成交（taker）还是被动挂单成交（maker）由 ``crosses`` 决定。"""
        return self.taker_net() if crosses else self.maker_net(offset)


PERP_FEES: Final[FeeSchedule] = FeeSchedule(maker=PERP_MAKER_FEE, taker=PERP_TAKER_FEE)
SPOT_FEES: Final[FeeSchedule] = FeeSchedule(maker=SPOT_MAKER_FEE, taker=SPOT_TAKER_FEE)
