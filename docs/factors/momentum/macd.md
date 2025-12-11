# [MACD] 平滑异同移动平均线

## 1. 核心逻辑 (Core Logic)
* **因子类别**：动量 / 趋势 (Momentum / Trend)
* **核心假设**：
    1.  短期均线与长期均线的**乖离率 (Divergence)** 反映了当前价格动能的强弱。
    2.  当短期均线试图回归长期均线（收敛）时，意味着原有趋势减弱；当两者发散时，意味着趋势加强。
* **原始出处**：Gerald Appel (1970s)

## 2. 数学定义 (Mathematical Formulation)

### 2.1 指数移动平均 (EMA)
MACD 的计算基础是 EMA。设 $N$ 为周期，$P_t$ 为 $t$ 时刻收盘价，平滑系数 $\alpha = \frac{2}{N+1}$。

$$
EMA(N)_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA(N)_{t-1}
$$

### 2.2 差离值 (DIF / MACD Line)
即“快线”。计算短期 EMA 与长期 EMA 的差值。
设 $N_{short}$ 为短周期，$N_{long}$ 为长周期：

$$
DIF_t = EMA(N_{short})_t - EMA(N_{long})_t
$$

### 2.3 讯号线 (DEA / Signal Line)
即“慢线”。对 DIF 进行平滑处理，消除噪音。
设 $N_{mid}$ 为平滑周期：

$$
DEA_t = EMA(DIF, N_{mid})_t
$$

### 2.4 柱状图 (MACD Histogram)
用于衡量 DIF 与 DEA 之间的距离。
*(注：中国 A 股软件通常会乘以 2 以放大视觉效果，国际标准通常不乘)*

$$
Hist_t = (DIF_t - DEA_t) \times 2
$$

### 2.5 交易信号 (Signal Generation)
$$
Signal_t = 
\begin{cases} 
1 (Golden Cross), & \text{if } DIF_{t-1} < DEA_{t-1} \text{ and } DIF_t > DEA_t \\
-1 (Death Cross), & \text{if } DIF_{t-1} > DEA_{t-1} \text{ and } DIF_t < DEA_t \\
0 (Hold), & \text{otherwise}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N_{short}$**：短周期，标准值为 **12**。
* **$N_{long}$**：长周期，标准值为 **26**。
* **$N_{mid}$**：平滑周期，标准值为 **9**。

## 4. 实战注意 (Notes)
1.  **滞后性**：由于基于均线，MACD 信号在震荡市中会出现严重的滞后和频繁的假信号（左右打脸）。
2.  **背离 (Divergence)**：在量化中，**“顶背离”**（价格创新高，MACD 没创新高）和**“底背离”**（价格创新低，MACD 没创新低）的预测价值远高于简单的金叉死叉。
3.  **零轴意义**：
    * $DIF > 0$：多头市场（EMA12 > EMA26）。
    * $DIF < 0$：空头市场（EMA12 < EMA26）。
    * 零轴上方的金叉（水上金叉）通常比零轴下方的金叉（水下金叉）动量更强。
