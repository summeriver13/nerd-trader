# [ATR] 平均真实波幅 (Average True Range)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：波动率 / 风控 (Volatility / Risk Management)
* **核心假设**：
    1.  **波幅即情绪**：价格波动的幅度反映了市场参与者的活跃度和情绪激烈程度。
    2.  **均值回归**：波动率具有周期性。长时间的低波动（死鱼行情）通常是暴涨暴跌的前兆（波动率压缩）；极端的的高波动往往意味着趋势的不可持续或反转。
    3.  **缺口修正**：相比于标准差（Std Dev），ATR 考虑了**跳空缺口 (Gap)**，更能真实反映隔夜或突发事件带来的风险。
* **原始出处**：J. Welles Wilder - *New Concepts in Technical Trading Systems* (1978)

## 2. 数学定义 (Mathematical Formulation)

### 2.1 真实波幅 (TR - True Range)
为了涵盖跳空缺口的影响，TR 取以下三者的最大值：
1.  今日最高价减最低价（日内波幅）。
2.  今日最高价减昨日收盘价的绝对值（向上跳空）。
3.  今日最低价减昨日收盘价的绝对值（向下跳空）。

$$
TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
$$

### 2.2 平均真实波幅 (ATR)
对 TR 进行移动平均处理。Wilder 原版使用的是修正式移动平均 (RMA/SMMA)，现代量化常用简单移动平均 (SMA) 或指数移动平均 (EMA)。

$$
ATR_t = \frac{1}{N} \sum_{i=0}^{N-1} TR_{t-i}
$$

*(注：Wilder 原版公式为 $ATR_t = \frac{(N-1) \cdot ATR_{t-1} + TR_t}{N}$)*

## 3. 因子应用场景 (Application)

ATR 在量化中极少单独作为买卖信号，而是作为**“基础建设”**因子：

### 3.1 动态止损 (Dynamic Stop Loss)
* **逻辑**：止损不应该设死固定点数（如跌 5%），而应该根据当前市场的波动率来定。
* **公式**：
  $$StopLoss\_Price = Entry\_Price - K \times ATR$$
  *(著名的“吊灯止损” Chandelier Exit 通常设 K=3)*

### 3.2 仓位管理 (Position Sizing)
* **海龟法则核心**：波动率越大的标的，买得越少；波动率越小的标的，买得越多。
* **公式**：
  $$Unit = \frac{Total\_Capital \times Risk\%}{ATR \times Contract\_Size}$$
  *(这被称为“波动率平价” Volatility Parity 的雏形)*

### 3.3 趋势过滤器 (Trend Filter)
* **逻辑**：趋势的启动往往伴随着 ATR 的上升。
* **信号**：$ATR > MovingAverage(ATR, 20)$，代表行情被激活，此时才允许开启趋势跟踪策略。

## 4. 参数说明 (Parameters)
* **$N$**：计算周期，默认 **14**。
* **$K$**：倍数系数，通常止损用 **2.0 ~ 3.0**，海龟法则用 **1.0 (即 1N)**。

## 5. 实战注意 (Notes)
1.  **绝对值问题**：ATR 是一个绝对价格数值（如 BTC 的 ATR 是 2000U，ETH 是 150U）。在做多因子比较时，必须将其**归一化**。
    * **NATR (Normalized ATR)**：$NATR = \frac{ATR}{Close} \times 100\%$。这样才能横向比较不同股票的波动率。
2.  **滞后性**：ATR 飙升时，通常暴跌或暴涨已经发生了一半。用它来做预测比较难，用来做**防御**（止损）最好。
