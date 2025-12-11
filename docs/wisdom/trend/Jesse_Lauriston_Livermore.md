---
name: Jesse Lauriston Livermore
nickname: The Boy Plunger (拼命三郎), The Great Bear of Wall Street (华尔街大空头)
born: 1877
died: 1940
role: Speculator, Author
key_concepts: 
  - Pivot Points (关键点)
  - Pyramiding (金字塔加仓)
  - Line of Least Resistance (最小阻力线)
  - Top-Down Trading (自上而下交易法)
  - Time Element (时间要素/耐心)
---

# 杰西·利弗莫尔 (Jesse Livermore)

> "The game of speculation is the most uniformly fascinating game in the world. But it is not a game for the stupid, the mentally lazy, the person of inferior emotional balance, or the get-rich-quick adventurer. They will die poor."  
> **“投机是世界上最迷人的游戏。但这不适合愚蠢的人、懒惰的人、情绪失控的人，或者是想一夜暴富的冒险家。这些人终将死于贫穷。”**

## 1. 人物简介

**杰西·利弗莫尔**（Jesse Livermore）被公认为历史上最伟大的**投机之王**。他的一生是大起大落的传奇，曾四次破产，又四次东山再起。

他在 **1929 年的大崩盘** 中做空美股，获利超过 **1 亿美元**（相当于今天的 15-20 亿美元），成为当时世界上最富有的人之一。他的传记小说《股票作手回忆录》(Reminiscences of a Stock Operator) 被誉为交易员的“圣经”，是华尔街必读的经典。

他没有电脑，没有复杂的指标，仅靠一根纸带（Ticker Tape）和对**人性**的深刻洞察，定义了现代趋势交易的法则。

---

## 2. 核心交易哲学 (Trading Philosophy)

利弗莫尔的体系建立在对**价格行为**和**市场心理**的极致观察上。

### 2.1 最小阻力线 (Line of Least Resistance)
这是利弗莫尔对“趋势”的定义。
* **逻辑**：价格像水一样，总是沿着阻力最小的方向流动。
* **策略**：绝不逆势而为。在牛市中只做多，在熊市中只做空。如果价格横盘震荡，说明阻力线尚未清晰，此时应**空仓等待**。

### 2.2 关键点 (Pivot Points)
他不是在最低点买入，而是在“关键点”买入。
* **定义**：这是一个心理关口，通常是前期的阻力位或整数关口。
* **操作**：一旦价格带量突破关键点，意味着最小阻力线方向确立，此时是**最佳入场时机**。这本质上是现代**突破交易 (Breakout Trading)** 的原型。

### 2.3 领头羊理论 (Leaders)
* **逻辑**：在一个板块中，永远只交易最强的那两只股票。
* **应用**：如果领头羊开始下跌，整个板块乃至大盘往往都会随之转势。

---

## 3. 资金管理与执行 (Execution & Money Management)

利弗莫尔不仅是技术大师，更是资金管理大师。他的很多规则是用破产换来的教训。

### 3.1 金字塔加仓 (Pyramiding)
这是他赚取暴利的核心手段。
* **试探仓 (The Probe)**：先买入 20% 的仓位。
* **加仓规则**：只有当第一笔交易**盈利**时，才继续加仓。
* **逻辑**：**“不要向下摊平亏损，要向上摊平盈利。”** 凡是赚钱的仓位，证明你的判断是对的，所以要加注；亏钱的仓位证明你错了，必须止损。

### 3.2 10% 止损法则
* **规则**：任何一笔交易的亏损绝不能超过本金的 **10%**。
* **逻辑**：这是一个心理防线，也是资金防线。一旦亏损超过这个数字，人的心态会坏掉，不仅本金受损，判断力也会受损。

---

## 4. 科学解构：量化利弗莫尔 (The Quant Perspective)

虽然利弗莫尔是主观交易者，但他的策略完全可以翻译成现代的**趋势跟踪 (Trend Following)** 算法。

### 4.1 突破策略代码逻辑 (Pseudo-code)

利弗莫尔的“突破关键点”在量化中就是 Donchian Channel（唐奇安通道）或 N日高点突破。

```python
# 计算过去 N 天的最高价（关键点）
N = 50
df['Pivot_Point'] = df['high'].rolling(N).max().shift(1)

# 交易信号：价格突破关键点
# 这是一个典型的趋势突破逻辑
df['Signal'] = np.where(df['close'] > df['Pivot_Point'], 1, 0)

# 过滤震荡：结合波动率 (ATR)
# 如果 ATR 过低（死水一潭），即使突破也不入场，防止假突破
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['Valid_Breakout'] = df['Signal'] & (df['ATR'] > df['ATR'].rolling(20).mean())
```

### 4.2 浮盈加仓逻辑 (Pyramiding Logic)
这也是量化 CTA 策略中常用的技巧。

$$
\text{Position Size}_t = \begin{cases} 
\text{Initial Size}, & \text{if first entry} \\
\text{Position Size}_{t-1} + \Delta, & \text{if } P_t > P_{entry} + 1 \times \text{ATR} 
\end{cases}
$$

* **逻辑**：每当价格上涨一个 ATR 幅度，就加仓一次，同时移动止损位。

---

## 5. 经典语录 (Quotes)

> "It was never my thinking that made the big money for me. It was always my sitting."  
> **“赚大钱的从来不是我的思考，而是我的‘坐功’（持仓不动）。”** > *(注：强调在趋势确立后，要拿得住单子，不要被微小的波动洗出去。)*

> "Wall Street never changes, the pockets change, the suckers change, the stocks change, but Wall Street never changes, because human nature never changes."  
> **“华尔街从未改变。钱袋子会变，股票会变，但华尔街依旧，因为人性从未改变。”**

> "There is nothing new in Wall Street. There can't be because speculation is as old as the hills."  
> **“华尔街没有新鲜事。投机像山岳一样古老。”**
