# [VIX] 恐慌指数 / 隐含波动率 (Volatility Index)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：波动率 / 宏观状态 (Volatility / Macro Regime)
* **核心假设**：
    1.  **期权定价原理**：当投资者对未来感到恐慌时，会大量买入期权（尤其是看跌期权）进行避险，推高期权价格。VIX 反映的就是期权价格中蕴含的“隐含波动率”。
    2.  **负相关性**：VIX 与风险资产（如标普500、比特币）通常呈**显著负相关**。股市大跌，VIX 飙升。
    3.  **均值回归之王**：资产价格可以无限上涨，但 VIX 不可以。它具有极强的均值回归特性——飙升得越快，回落得也越快。
* **原始出处**：CBOE (芝加哥期权交易所) - 1993年推出，2003年修订算法。

## 2. 数学定义 (Mathematical Formulation)

### 2.1 概念公式 (Conceptual)
VIX 的计算逻辑并非简单的标准差，而是基于**方差互换 (Variance Swap)** 的无模型复现。
它利用一系列不同行权价 ($K$) 的标普 500 期权价格 ($Q(K)$)，加权计算出未来 30 天的预期波动率。

$$
\sigma^2 = \frac{2}{T} \sum_{i} \frac{\Delta K_i}{K_i^2} e^{RT} Q(K_i) - \frac{1}{T} \left( \frac{F}{K_0} - 1 \right)^2
$$

$$
VIX = 100 \times \sqrt{\sigma^2}
$$

* $\sigma^2$：方差率。
* $T$：距离到期时间。
* $Q(K_i)$：行权价为 $K_i$ 的期权中点价格。
* **注意**：个人开发者很难从头计算这个公式（需要全市场的期权链数据）。**实战中直接获取交易所发布的指数数据即可。**

### 2.2 Crypto 市场的对应物 (DVOL)
在加密货币市场，对应的是 **Deribit Volatility Index (DVOL)**。
其算法逻辑与 CBOE VIX 完全一致，基于 BTC 或 ETH 的期权链数据计算。

### 2.3 波动率风险溢价 (VRP - Volatility Risk Premium) *[进阶因子]*
这是量化机构常用的赚钱因子。
$$
VRP = Implied\_Vol (VIX) - Realized\_Vol (HV)
$$
* **逻辑**：隐含波动率 (预期) 通常高于 历史波动率 (实际)。这个差值就是卖出期权者赚取的“保险费”。

## 3. 信号生成 (Signal Generation)

VIX 通常不作为直接的买卖信号，而是作为**“过滤器” (Filter)** 或 **“开关”**。

### 3.1 状态过滤 (Regime Filter)
$$
Regime_t = 
\begin{cases} 
\text{Panic (恐慌)}, & \text{if } VIX_t > 30 \quad (\text{Crypto: } DVOL > 70) \\
\text{Normal (正常)}, & \text{if } 15 < VIX_t \le 30 \quad (\text{Crypto: } 45 < DVOL \le 70) \\
\text{Complacent (贪婪)}, & \text{if } VIX_t \le 15 \quad (\text{Crypto: } DVOL \le 45)
\end{cases}
$$

### 3.2 逆势抄底信号 (Contrarian Buy)
利用 VIX 的极值回归特性捕捉大底。
$$
Signal_{buy} = 1, \quad \text{if } VIX_t > \text{BollingerUpper}(VIX, 20, 2.5) \text{ AND } VIX_t < VIX_{t-1}
$$
*(当 VIX 冲破布林带上轨并开始回落时，往往是市场情绪的极值点)*

## 4. 参数说明 (Parameters)
* **观察标的**：
    * 美股策略：使用 **^VIX**。
    * 币圈策略：使用 **BTC-DVOL** (Deribit)。
* **阈值设置**：
    * 币圈的波动率通常是美股的 **2-3 倍**。不要直接套用美股的阈值（如 20），币圈的“风平浪静”可能就是 40。

## 5. 实战注意 (Notes)

### 5.1 不可直接交易性
* VIX 指数本身是不可交易的数学统计值。
* 交易工具：VIX 期货 (Futures) 或 VIX ETF (如 VXX, UVXY)。
* **巨坑预警**：VXX 等 ETF 长期来看是**归零**的。因为 VIX 期货结构通常是升水 (Contango)，长期持有会产生巨大的展期损耗 (Roll Decay)。**切勿长线持有 VIX 做多产品。**

### 5.2 牛熊不对称
* 股市/币市缓涨时，VIX 会缓慢阴跌。
* 股市/币市急跌时，VIX 会瞬间暴涨。
* 这种不对称性使得 **做空 VIX** 胜率很高（捡硬币），但赔率极差（被火车撞）；**做多 VIX** 胜率极低（买彩票），但赔率极高（中头奖）。

### 5.3 塔勒布式策略 (Tail Risk Hedging)
* 在组合中配置 **1%-2%** 的资金购买虚值看涨 VIX 期权（或 UVXY Call）。
* 平时这 1% 会亏光，但在类似 "3.12" 或 "5.19" 暴跌日，这 1% 的仓位能变成 50%，覆盖掉你现货的所有亏损。
