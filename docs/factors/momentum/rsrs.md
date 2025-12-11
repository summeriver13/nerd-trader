# [RSRS] 阻力支撑相对强度

## 1. 核心逻辑 (Core Logic)
* **因子类别**：动量 / 结构 (Momentum / Structure)
* **核心假设**：阻力位（最高价）与支撑位（最低价）的相对变化速率隐含了市场强弱趋势。当支撑位的抬升速度快于阻力位时，市场处于强势。
* **原始出处**：光大证券《基于阻力支撑相对强度(RSRS)的择时策略》

## 2. 数学定义 (Mathematical Formulation)

### 2.1 线性回归模型
对过去 $N$ 日的最高价序列 $\{H_t\}$ 和最低价序列 $\{L_t\}$ 进行普通最小二乘法 (OLS) 线性回归：

$$
H_t = \beta \cdot L_t + \alpha + \epsilon_t
$$

其中斜率 $\beta$ 即为基础 RSRS 值：

$$
\beta = \frac{Cov(L, H)}{Var(L)}
$$

### 2.2 标准化处理 (Z-Score)
为了消除不同标的或不同时期的波动率差异，对斜率 $\beta$ 进行标准化处理。
设 $M$ 为历史回顾窗口（通常 $M=600$）：

$$
\mu_{\beta} = \frac{1}{M} \sum_{i=0}^{M-1} \beta_{t-i}
$$

$$
\sigma_{\beta} = \sqrt{\frac{1}{M-1} \sum_{i=0}^{M-1} (\beta_{t-i} - \mu_{\beta})^2}
$$

最终标准分因子值：

$$
RSRS_{std} = \frac{\beta_t - \mu_{\beta}}{\sigma_{\beta}}
$$

### 2.3 交易信号
$$
Signal_t = 
\begin{cases} 
1 (Buy), & \text{if } RSRS_{std} > S_{buy} \\
0 (Sell), & \text{if } RSRS_{std} < S_{sell}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N$ (Regression Window)**：回归周期，默认 **18**。
* **$M$ (History Window)**：标准化历史窗口，默认 **600**。
* **$S_{buy}$**：买入阈值，默认 **0.7**。
* **$S_{sell}$**：卖出阈值，默认 **-0.7**。

## 4. 实战注意 (Notes)
* 计算回归时应包含当日数据（t时刻）。
* 对于新上市股票（历史数据不足 M 天），需做填充处理或跳过。
