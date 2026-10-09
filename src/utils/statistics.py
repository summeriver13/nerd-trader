import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm  # 修正: 使用 statsmodels.api 进行 OLS 回归

logger = logging.getLogger(__name__)


def calculate_z_score(series: pd.Series, window: int) -> pd.Series:
    """
    计算滚动 Z-Score (用于均值回归策略)
    公式: (当前价 - 均值) / 标准差
    """
    # 确保输入不为空，避免报错
    if series.empty:
        return pd.Series(dtype=float)

    roll = series.rolling(window=window)
    mean = roll.mean()
    std = roll.std(ddof=1)  # 明确指定自由度，通常样本标准差用 ddof=1

    # 处理标准差为 0 的情况 (虽然极少见)，避免除以 0
    z_score = (series - mean) / std
    return z_score.replace([np.inf, -np.inf], np.nan)


def calculate_half_life(series: pd.Series) -> float:
    """
    计算均值回归的半衰期 (Half-Life)
    基于 Ornstein-Uhlenbeck (OU) 过程: dx_t = -lambda * (x_t - mean) * dt + sigma * dW_t
    离散化形式: x_t - x_{t-1} = alpha + beta * x_{t-1} + epsilon
    其中 lambda = -beta (回归系数)
    半衰期 = -ln(2) / beta

    :return: float 半衰期 (单位: K线周期数)
    """
    # 1. 数据准备
    series = series.dropna()
    if len(series) < 10:
        return 0.0  # 数据太少无法计算

    # 构造滞后项 (Lagged Price) 和 差分项 (Delta Price)
    df_lag = series.shift(1)
    df_delta = series - df_lag

    # 2. 数据对齐与清洗 (非常重要)
    # 将两个序列合并为一个 DataFrame，以此确保 dropna 时同时删除对应行
    df_reg = pd.DataFrame({"y": df_delta, "x": df_lag})
    df_reg = df_reg.dropna()

    # 3. 运行 OLS 线性回归
    # 公式: Delta = const + beta * Lag
    # 我们需要添加常数项 (intercept)，否则假设均值为 0
    X = sm.add_constant(df_reg["x"])
    y = df_reg["y"]

    try:
        model = sm.OLS(y, X)
        res = model.fit()

        # 4. 提取回归系数 beta (即 x 的系数)
        # params[1] 是 x 的系数，params[0] 是常数项
        beta = res.params.iloc[1]
    except Exception as e:
        # 回归失败（数据完全平直等）退化为「无半衰期」：返回值语义与 beta>=0 一致，
        # 但原因必须打出来，否则「没有均值回归」与「算不出来」看起来一模一样
        logger.warning(f"半衰期回归失败，按无半衰期处理: {e}")
        return 0.0

    # 5. 计算半衰期
    # 均值回归的前提是 beta 必须为负数 (即价格偏离时会被拉回)
    # 如果 beta >= 0，说明是动量发散趋势，不存在半衰期
    if beta >= 0:
        return float("inf")  # 无穷大，表示没有均值回归特性

    half_life = -np.log(2) / beta
    return float(half_life)
