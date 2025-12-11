# [KDJ] 随机指标 (Stochastic Oscillator)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：动量 / 反转 (Momentum / Reversion)
* **核心假设**：
    1.  在上升趋势中，收盘价倾向于接近周期内的**最高价**。
    2.  在下降趋势中，收盘价倾向于接近周期内的**最低价**。
    3.  当收盘价开始远离极值位时，意味着原有动量衰竭，可能发生反转。
* **原始出处**：George Lane (1950s) - *注：J线为后续改良，主要流行于亚洲市场。*

## 2. 数学定义 (Mathematical Formulation)

### 2.1 未成熟随机值 (RSV - Raw Stochastic Value)
计算当前收盘价在过去 $N$ 天波动范围内的相对位置。
设 $C_t$ 为今日收盘价，$L_N$ 为过去 $N$ 天的最低价，$H_N$ 为过去 $N$ 天的最高价。

$$
RSV_t = \frac{C_t - L_N}{H_N - L_N} \times 100
$$

### 2.2 K 线 (Fast Stochastic)
对 RSV 进行平滑处理（通常采用 SMA 权重，即 $1/3$ 的新值，$2/3$ 的旧值）。
若无前一日数据，通常设初始值为 50。

$$
K_t = \frac{2}{3} \cdot K_{t-1} + \frac{1}{3} \cdot RSV_t
$$

### 2.3 D 线 (Slow Stochastic)
对 K 线再次进行平滑处理。

$$
D_t = \frac{2}{3} \cdot D_{t-1} + \frac{1}{3} \cdot K_t
$$

### 2.4 J 线 (Divergence / Sensitive Line)
J 线是 KDJ 独有的指标，反映 K 值与 D 值的乖离程度。它本质上是一个**超敏信号**，可以超过 100 或低于 0。

$$
J_t = 3 \cdot K_t - 2 \cdot D_t
$$

### 2.5 交易信号 (Signal Generation)
$$
Signal_t = 
\begin{cases} 
-1 (Overbought/Sell), & \text{if } K_t > 80 \text{ or } D_t > 80 \text{ or } J_t > 100 \\
1 (Oversold/Buy), & \text{if } K_t < 20 \text{ or } D_t < 20 \text{ or } J_t < 0 \\
0 (Hold), & \text{otherwise}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N$**：计算周期，默认 **9**。
* **$M1$**：K 线平滑权重分母，默认 **3**。
* **$M2$**：D 线平滑权重分母，默认 **3**。

## 4. 实战注意 (Notes)
1.  **钝化 (Passivation)**：KDJ 最大的缺陷。在**单边暴涨/暴跌**行情中，K 值和 D 值会长期粘合在 80 以上或 20 以下（钝化）。此时如果按照“超买卖出、超卖买入”操作，会被严重轧空或深套。
    * *量化解法*：必须引入 **ADX** 或 **均线** 判断当前是否为强趋势。如果是强趋势，禁用 KDJ 反转信号。
2.  **J 线的用法**：
    * $J > 100$：极端超买，短期回调风险极大。
    * $J < 0$：极端超卖，短期反弹概率极大。
3.  **背离 (Divergence)**：
    * 价格创新高，但 J 线一波比一波低（顶背离），是比死叉更有效的离场信号。
