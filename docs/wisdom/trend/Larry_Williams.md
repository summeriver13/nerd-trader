---
name: Larry Williams
born: 1942
role: Trader, Author, Politician
key_concepts: 
  - Williams %R (威廉指标)
  - COT Report Analysis (COT持仓报告)
  - Volatility Breakout (波动率突破)
  - Money Management (资金管理/凯利公式)
  - Market Sentiment (市场情绪)
---

# 拉里·威廉姆斯 (Larry Williams)

> "I have developed a powerful inner feeling that the trade I am about to make will be a loser... that is the only way I can protect myself."  
> **“我培养出了一种强烈的内在感觉，即我即将进行的这笔交易会亏损……这是我保护自己的唯一方式。”**

## 1. 人物简介

**拉里·威廉姆斯**（Larry Williams）是著名的短线交易员、作家和技术分析指标的发明者。

他在交易界封神的战役发生在 **1987 年罗宾斯世界杯期货交易锦标赛 (Robbins World Cup)**。他使用基于波动率突破和趋势跟随的策略，将 **1 万美元的本金在一年内变成了 113 万美元**，收益率高达 **11,376%**。更有趣的是，他的女儿米歇尔·威廉姆斯（Michelle Williams，好莱坞著名女星）在 10 年后也以 1000% 的收益率赢得了该项比赛，证明了他策略的可复制性。

与纯粹的技术派不同，拉里非常看重**“商业逻辑”**和**“基本面数据”**（如 COT 报告）在短线交易中的应用。

---

## 2. 核心技术指标 (Technical Indicators)

拉里发明了数个广为流传的指标，其中最著名的是威廉指标 (%R)，它从数学上捕捉了市场的“超买/超卖”状态。

### 2.1 威廉指标 (Williams %R)

这是一个动量指标，与随机指标 (KDJ) 的原理相反。它测量收盘价相对于过去 N 天最高价的位置。

$$
\%R = \frac{H_n - C}{H_n - L_n} \times -100
$$

* **$C$**：当日收盘价。
* **$H_n$**：过去 N 日（通常 14 日）内的最高价。
* **$L_n$**：过去 N 日内的最低价。

**量化逻辑**：
* **数值范围**：0 到 -100。
* **0 到 -20**：视为**超买** (Overbought)，卖方主力可能入场。
* **-80 到 -100**：视为**超卖** (Oversold)，买方主力可能入场。
* **区别**：与 KDJ 不同，%R 对价格的反应更敏锐，更适合捕捉短线反转点。

---

## 3. 核心交易思想 (Trading Philosophy)

拉里·威廉姆斯的短线并非“高频剥头皮”，他的“短线”通常指持有几天到几周 (Swing Trading)。

### 3.1 波动率突破 (Volatility Breakout)
拉里认为：**“利润是由大幅度的价格波动产生的。”**
* **逻辑**：当价格波幅突然扩大时，往往意味着一个新的趋势开始。
* **策略模型**：
    * 计算过去几日的平均波动幅度 (Range)。
    * 如果今日价格突破了 `开盘价 + k × 昨日振幅`，则追涨买入。
    * 这是一种典型的**右侧交易**逻辑。

### 3.2 聪明钱的踪迹 (Follow the Smart Money)
他是最早大力推广 **COT 报告 (Commitment of Traders)** 的交易员之一。
* **商业头寸 (Commercials)**：这是由于业务需要（如农场主、矿商）进行套保的人，他们最懂现货的真实价值。
* **策略**：当“商业头寸”极度看多，而“散户投机者”极度看空时，往往是市场的历史大底。

### 3.3 关键日 (Smash Day)
这是他定义的特定价格形态，用于捕捉反转：
* **买入设置**：收盘价低于前一天的低点（看似要跌），但第二天价格反转并突破了前一天的高点。这实际上是一个**“空头陷阱”**。

---

## 4. 资金管理：交易的圣杯 (Money Management)

拉里·威廉姆斯之所以能创造 110 倍的收益，核心在于激进的资金管理（凯利公式的变体）。他著名的公式是：

$$
N = \frac{\text{Account Balance} \times \text{Risk \%}}{\text{Stop Loss amount}}
$$

* **核心教训**：他在 1987 年的比赛中，资金一度达到 200 万美元，但因为过度自信没有减少仓位，几天内回撤到 70 万美元，最终收在 113 万。
* **结论**：**“交易中最重要的是资金管理，而不是预测市场。”**

---

## 5. 科学解构：量化拉里 (The Quant Perspective)

拉里的策略非常适合编写成量化策略，因为他的规则大多是明确的“价格行为” (Price Action)。

### 5.1 波动率突破策略代码逻辑 (Pseudo-code)

这是一个基于他《短线交易秘诀》中提到的“开盘突破”逻辑的简化版：

```python
# 计算过去 N 天的真实波幅 (ATR 的简化版)
df['Range'] = df['high'] - df['low']
# 过去 3 天的平均振幅
df['Avg_Range'] = df['Range'].rolling(3).mean()

# 设定触发阈值 k (例如 0.5 到 0.8)
k = 0.6

# 交易信号
# 如果当前价格 > 开盘价 + k倍的平均振幅，则做多
df['Buy_Trigger'] = df['open'] + (df['Avg_Range'].shift(1) * k)
df['Signal'] = np.where(df['close'] > df['Buy_Trigger'], 1, 0)
```

### 5.2 周期性与季节性
拉里是**时间周期 (Time Cycle)** 和**季节性 (Seasonality)** 的坚定信徒。
* **量化点**：在编写策略时，加入 `Month_of_Year` 或 `Day_of_Week` 作为过滤因子。例如，他发现黄金往往在周一/周二形成低点。

---

## 6. 经典语录 (Quotes)

> "The moment you start 'hoping' in a trade, you are wrong."
> **“当你开始在交易中‘祈祷’的那一刻，你就已经输了。”**

> "Profitable trading does not come from predicting the future; it comes from reacting to what is happening now."
> **“盈利的交易不来自于预测未来，而来自于对当下发生的事情做出反应。”**
