# [PB] 市净率 (Price-to-Book Ratio)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：价值 (Value)
* **核心假设**：
    1.  股票的底价由其**净资产**决定（清算价值）。
    2.  低 PB 意味着价格接近甚至低于公司的拆解价值（破净），下行风险有限（反脆弱）。
    3.  适用于重资产行业（银行、地产、工业），不适用于轻资产行业（互联网、服务业）。
* **原始出处**：Fama & French - *Three Factor Model* (1993)

## 2. 数学定义 (Mathematical Formulation)

### 2.1 基础公式
设 $P_t$ 为当前总市值，$BookValue_t$ 为最近财报的归属于母公司股东权益（净资产）。

$$
PB_{lf} = \frac{P_t}{BookValue_{latest}}
$$

### 2.2 账面市值比 (B/P - Book to Price) *[因子计算标准]*
同样为了处理数学分布的线性关系，量化中常采用倒数。

$$
BP = \frac{1}{PB} = \frac{BookValue_{latest}}{P_t}
$$

### 2.3 与 ROE 的关系
根据杜邦分析，PB 与 净资产收益率 (ROE) 存在内在联系：
$$
PB = PE \times ROE
$$
这意味着，**高 PB 必须有高 ROE 支撑才是合理的**。如果 PB 很高但 ROE 很低，则是典型的泡沫。

## 3. 参数说明 (Parameters)
* **无形资产调整**：对于部分行业，建议使用**有形资产净值 (Tangible Book Value)**，即从净资产中扣除商誉 (Goodwill) 和无形资产。
  $$PB_{tangible} = \frac{P_t}{NetAssets - IntangibleAssets - Goodwill}$$

## 4. 实战注意 (Notes)
1.  **资产质量**：极低 PB（如 < 0.5）可能不是机会，而是陷阱。这通常意味着市场认为该公司的资产是“虚”的（如卖不出去的存货、收不回的应收账款）。
2.  **行业局限**：对于科技公司（如 OpenAI），其核心资产是“人”和“代码”，这些不上资产负债表。用 PB 去估值科技股是刻舟求剑。
3.  **回购影响**：美股公司大量回购会导致净资产减少（甚至为负），导致 PB 极高或失效，需配合 Shareholder Yield 使用。
