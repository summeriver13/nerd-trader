---
name: John Bollinger
born: 1950
role: CFA, CMT, President of Bollinger Capital Management
key_concepts: 
  - Bollinger Bands (布林带)
  - Volatility (波动率)
  - Standard Deviation (标准差)
  - Mean Reversion (均值回归)
  - BandWidth (带宽)
  - Percent B (%B)
---

# 约翰·布林 (John Bollinger)

> "Volatility is not a constant; it is dynamic."  
> **“波动率不是恒定的，它是动态的。”**

## 1. 人物简介

**约翰·布林**（John Bollinger）是美国著名的金融分析师、作家和技术分析大师，拥有 CFA 和 CMT 双重特许资格。

他在 1980 年代初期发明了 **布林带 (Bollinger Bands)**。在此之前，交易员通常使用固定百分比的“包络线” (Envelopes)。布林创造性地引入了统计学中的**标准差 (Standard Deviation)** 概念，使信封带能够随着市场波动率的变化自动收缩和扩张，解决了固定带宽无法适应市场“平静/剧烈”切换的难题。

---

## 2. 核心数学构造 (Mathematical Framework)

布林带的设计哲学是：**价格总是围绕价值中枢（均线）波动，而波动的边界由市场的波动率决定。**

### 2.1 三轨公式
布林带由三条轨道组成，默认参数通常为 $N=20, k=2$：

$$
\begin{aligned}
\text{Middle Band (中轨)} &= SMA(Close, N) \\
\text{Upper Band (上轨)} &= \text{Middle Band} + (k \times \sigma) \\
\text{Lower Band (下轨)} &= \text{Middle Band} - (k \times \sigma)
\end{aligned}
$$

* **$SMA$**：简单移动平均线（通常 20 日）。
* **$\sigma$ (Sigma)**：N 日内的价格标准差。
* **$k$**：标准差倍数（通常为 2）。

### 2.2 统计学逻辑：正态分布
布林带基于正态分布理论：
* 在统计学中，约 **95.4%** 的数值分布在平均值 $\pm 2\sigma$ 的范围内。
* **自适应性 (Adaptability)**：
    * **市场平静时**：$\sigma$ 变小 $\rightarrow$ 轨道收口 (Squeeze)。
    * **市场剧烈时**：$\sigma$ 变大 $\rightarrow$ 轨道张口 (Expansion)。

---

## 3. 核心衍生指标 (Derived Indicators)

为了更精确地描述布林带的状态，布林定义了两个关键指标，用于构建量化策略。

### 3.1 %B (位置指标)
描述当前价格在布林带中的相对位置（类似 KDJ 中的 J 值）。

$$
\%B = \frac{\text{Close} - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}
$$

* **$\%B > 1$**：价格突破上轨（极强趋势或超买）。
* **$\%B < 0$**：价格跌破下轨（极弱趋势或超卖）。
* **$\%B = 0.5$**：价格刚好在中轨。

### 3.2 BandWidth (带宽指标)
描述波动率的压缩程度，主要用于识别“变盘点”。

$$
\text{BandWidth} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{Middle Band}}
$$

---

## 4. 经典交易策略 (Key Strategies)

约翰·布林强调：**“触碰轨道本身并不是交易信号” (Tag of the band is not a signal)**。必须结合价格行为 (Price Action) 或其他指标。

### 4.1 波动率突破 (The Squeeze) — *趋势策略*
这是布林带最著名的用法，基于“波动率循环”理论：**极低的波动率往往预示着剧烈波动的爆发**。

* **识别条件**：`BandWidth` 降至 6 个月以来的最低点（轨道像香肠一样挤压）。
* **买入信号**：价格突破上轨 + 交易量放大。
* **卖出信号**：价格跌破下轨 + 交易量放大。
* **风险提示 (Head Fake)**：市场常出现“假突破”，即先向一个方向突破，然后迅速反向。必须设置止损或等待收盘确认。

### 4.2 均值回归 (Mean Reversion) — *震荡策略*
在没有明确趋势的震荡市中，布林带扮演“超买超卖”的边界。布林结合了 Arthur Merrill 的 W/M 形态理论。

* **W底 (W-Bottoms)**：
    1.  股价第一次探底，**跌破**下轨（$\%B < 0$）。
    2.  反弹至中轨附近。
    3.  第二次探底，**未跌破**下轨（$\%B > 0$），且价格高于前低。
    4.  **信号**：底背离形成，买入。
* **M顶 (M-Tops)**：
    1.  股价第一次冲高，**突破**上轨。
    2.  第二次冲高，**未突破**上轨，形成顶背离。
    3.  **信号**：卖出。

### 4.3 沿布林带行走 (Walking the Bands) — *强趋势*
新手最容易犯的错误是“破上轨就做空”。
* **现象**：在超强趋势中，价格会长时间“粘着”上轨上涨。
* **策略**：只要价格能收盘在上轨之外，或者回调不破中轨，说明趋势极强。**此时不能做空，反而应该持有**。

---

## 5. 科学解构：量化布林 (The Quant Perspective)

如果把布林带的思想翻译给计算机（Python/Pandas），它的本质是对**Z-Score**的应用。

### 5.1 数据标准化本质
布林带的核心其实是在计算价格的 **Z-Score (标准分数)**：

$$
Z = \frac{x - \mu}{\sigma} \approx \frac{\text{Close} - \text{MA20}}{\text{StdDev}}
$$

* 上轨其实就是 $Z = +2$ 的线。
* 下轨其实就是 $Z = -2$ 的线。
* **%B 指标** 其实就是将 $Z$ 值归一化到了 [0, 1] 区间。

### 5.2 策略代码逻辑 (Pseudo-code)
当你与 LLM 交互编写策略时，可以参考此逻辑：

```python
# 计算基础数据
df['SMA'] = df['close'].rolling(20).mean()
df['STD'] = df['close'].rolling(20).std()
df['Upper'] = df['SMA'] + 2 * df['STD']
df['Lower'] = df['SMA'] - 2 * df['STD']

# 策略 1: Squeeze (布林挤压)
# 筛选带宽处于历史低位的时刻
df['BandWidth'] = (df['Upper'] - df['Lower']) / df['SMA']
is_squeeze = df['BandWidth'] < df['BandWidth'].rolling(120).min()

# 策略 2: Walking the Bands (趋势跟随)
# 如果连续 3 天收盘价 > 上轨，则标记为强趋势
is_trending = (df['close'] > df['Upper']).rolling(3).sum() == 3
```

---

## 6. 经典语录 (Quotes)

> "The bands are a measure of highness and lowness on a relative basis."
> **“布林带是对‘相对高’和‘相对低’的一种度量。”**

> "Low volatility begets high volatility."
> **“低波动孕育高波动。”**
> *(注：这是对市场周期性最精辟的总结，也是 Squeeze 策略的灵魂。)*
