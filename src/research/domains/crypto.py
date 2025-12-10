import pandas as pd

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
        fee_rate: float = 0.0005,  # 万5 (Binance/OKX 常见 Taker 费率)
        slippage: float = 0.0005,  # 滑点 (流动性通常较好，设为万5)
    ):
        self.fee_rate = fee_rate
        self.slippage = slippage

    def calculate_cost(
        self, trade_amount: pd.Series, is_sell: pd.Series | bool
    ) -> TradeCost:
        """
        计算 Crypto 交易成本

        :param trade_amount: 交易金额 (绝对值)
        :param is_sell: 是否卖出 (Crypto中通常费率一致，此参数暂不影响计算，但保留接口一致性)
        """
        # 1. 交易手续费 (Commission/Fee)
        # 直接按比例计算，不设最低门槛
        commission = trade_amount * self.fee_rate

        # 2. 印花税 (Tax)
        # Crypto 无印花税
        if isinstance(trade_amount, pd.Series):
            tax = pd.Series(0.0, index=trade_amount.index)
        else:
            tax = 0.0

        # 3. 滑点 (Slippage)
        slippage_cost = trade_amount * self.slippage

        # 4. 汇总
        total_cost = commission + tax + slippage_cost

        if isinstance(trade_amount, pd.Series):
            return TradeCost(
                commission=commission, tax=tax, slippage=slippage_cost, total=total_cost
            )
        else:
            return TradeCost(commission, tax, slippage_cost, total_cost)
