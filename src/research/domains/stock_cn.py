import numpy as np
import pandas as pd

from src.research.domains.base import MarketDomain, TradeCost


class StockCNDomain(MarketDomain):
    """
    A股市场领域模型

    规则特征：
    1. 佣金 (Commission): 通常为万分之 2.5，但单笔最低 5 元 (不足 5 元按 5 元收)。
    2. 印花税 (Stamp Duty): 目前为万分之 5，仅在【卖出】时收取。
    3. 过户费: 极低，通常并入佣金或忽略，此处忽略。
    """

    def __init__(
        self,
        commission_rate: float = 0.00025,  # 万2.5
        min_commission: float = 5.0,  # 最低 5 元
        stamp_duty: float = 0.0005,  # 万5 (卖出收)
        slippage: float = 0.001,  # 滑点 (默认千1，A股冲击成本较Crypto小但也不可忽视)
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty = stamp_duty
        self.slippage = slippage

    def calculate_cost(
        self, trade_amount: pd.Series, is_sell: pd.Series | bool
    ) -> TradeCost:
        """
        计算 A 股交易成本

        :param trade_amount: 交易金额 (绝对值)
        :param is_sell: 是否卖出 (True=卖出, False=买入)
        """
        # 0. 预处理：确保 trade_amount 是 numpy array 或 series，方便计算
        # 如果传入的是 float，转为 series 统一处理会更安全，但这里假设引擎传入的是 Series

        # 1. 计算基础佣金 (Commission)
        raw_commission = trade_amount * self.commission_rate

        # 处理最低佣金逻辑：
        # 逻辑：如果交易额 > 0，则取 max(计算佣金, 5元)；如果交易额 = 0，则为 0。
        # 使用 np.where 避免对 0 交易额收取 5 元
        commission = np.where(
            trade_amount > 0, np.maximum(raw_commission, self.min_commission), 0.0
        )

        # 2. 计算印花税 (Stamp Duty) - 仅卖出收取
        # 逻辑：只有 (是卖出 AND 交易额 > 0) 时才收税
        tax = np.where(
            (is_sell) & (trade_amount > 0), trade_amount * self.stamp_duty, 0.0
        )

        # 3. 计算滑点 (Slippage)
        # 滑点通常双向都收，模拟成交价偏离
        slippage_cost = trade_amount * self.slippage

        # 4. 汇总
        total_cost = commission + tax + slippage_cost

        # 保持 Series 的索引与输入一致
        if isinstance(trade_amount, pd.Series):
            index = trade_amount.index
            return TradeCost(
                commission=pd.Series(commission, index=index),
                tax=pd.Series(tax, index=index),
                slippage=pd.Series(slippage_cost, index=index),
                total=pd.Series(total_cost, index=index),
            )
        else:
            # 兼容单点计算 (虽然向量化回测极少用到)
            return TradeCost(commission, tax, slippage_cost, total_cost)

    def adjust_signal(self, signals: pd.Series) -> pd.Series:
        """
        A 股特有的信号调整
        注意：在纯资金流回测中，通常不做 100 股取整，因为会引入较大的资金闲置误差。
        但如果是基于数量的回测，这里应该由 Engine 调用 logic。

        此处保留原样，或者您可以添加逻辑：
        将仓位百分比截断到对应的 100 股倍数 (比较复杂，需知当前价格)。
        """
        return signals
