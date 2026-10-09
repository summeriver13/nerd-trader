from typing import override

import pandas as pd

from src.core.costs import MARKET_SLIPPAGE, PERP_TAKER_FEE
from src.research.domains.base import MarketDomain, TradeCost


class CryptoDomain(MarketDomain):
    """
    加密货币市场领域模型 (Crypto)

    规则特征：
    1. 结构简单：通常只有交易手续费，无印花税。
    2. 无最低限制：可以交易极小金额 (dust trade)。
    3. 双向收费：买入和卖出通常都收取手续费。
    """

    def __init__(
        self,
        fee_rate: float = PERP_TAKER_FEE,
        slippage: float = MARKET_SLIPPAGE,
    ):
        self.fee_rate = fee_rate
        self.slippage = slippage

    @override
    def calculate_cost(
        self, trade_amount: pd.Series, is_sell: pd.Series | bool
    ) -> TradeCost:
        """计算 Crypto 交易成本（只支持 Series：调用方传的都是逐根名义金额）。

        :param trade_amount: 交易金额 (绝对值)
        :param is_sell: 是否卖出（Crypto 双向同费率，保留参数只为接口一致）
        """
        commission = trade_amount * self.fee_rate  # 手续费，无最低门槛
        tax = pd.Series(0.0, index=trade_amount.index)  # Crypto 无印花税
        slippage_cost = trade_amount * self.slippage
        return TradeCost(
            commission=commission,
            tax=tax,
            slippage=slippage_cost,
            total=commission + tax + slippage_cost,
        )
