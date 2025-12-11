# [Bollinger Mean Reversion] 布林带均值回归

## 1. 核心逻辑 (Core Logic)
* **因子类别**：波动率 / 反转 (Volatility / Mean Reversion)
* **核心假设**：价格服从正态分布，当价格偏离均值超过一定标准差（$\sigma$）时，属于极端小概率事件，大概率会回归均值。

## 2. 数学定义 (Mathematical Formulation)

### 2.1 中轨 (Middle Band)
中轨通常为 $N$ 周期简单移动平均线 (SMA)：

$$
MB_t = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}
$$

### 2.2 标准差 (Standard Deviation)
计算过去 $N$ 周期的样本标准差：

$$
\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=0}^{N-1} (P_{t-i} - MB_t)^2}
$$

### 2.3 上下轨 (Upper & Lower Bands)
$$
UB_t = MB_t + K \cdot \sigma_t
$$
$$
LB_t = MB_t - K \cdot \sigma_t
$$

### 2.4 相对位置指标 %B (Percent B)
为了量化价格在通道中的位置：

$$
\%B_t = \frac{P_t - LB_t}{UB_t - LB_t}
$$

### 2.5 交易信号 (均值回归逻辑)
$$
Signal_t = 
\begin{cases} 
-1 (Short), & \text{if } P_t > UB_t \quad (\%B_t > 1) \\
1 (Long), & \text{if } P_t < LB_t \quad (\%B_t < 0) \\
0 (Hold), & \text{otherwise}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N$**：计算周期，默认 **20**。
* **$K$**：标准差倍数，默认 **2.0**（涵盖约 95% 的价格波动）。

## 4. 实战注意 (Notes)
* 在强趋势行情（Trend）中，价格可能长期沿着上轨运行（钝化），此时均值回归策略会失效并造成巨额亏损。
* 建议结合 **ADX** 指标或 **ATR** 过滤，仅在低波动率环境下启用此策略。
