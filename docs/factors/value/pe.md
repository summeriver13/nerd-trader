# [PE] 市盈率 (Price-to-Earnings Ratio)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：价值 (Value)
* **核心假设**：
    1.  买股票就是买公司的未来盈利能力。
    2.  低市盈率意味着投资者以较低的成本购买了单位盈利，具有较高的**安全边际**。
    3.  根据**均值回归**理论，被低估的低 PE 股票未来有上涨修复估值的动力。
* **原始出处**：Benjamin Graham - *Security Analysis* (1934)

## 2. 数学定义 (Mathematical Formulation)

### 2.1 静态市盈率 (PE - Lyr)
使用上一年度的年报数据。
$$
PE_{lyr} = \frac{P_t}{EPS_{last\_year}}
$$

### 2.2 滚动市盈率 (PE - TTM) *[量化推荐]*
Trailing Twelve Months。使用最近 4 个季度的财务数据，消除季节性影响且更具时效性。
设 $P_t$ 为 $t$ 时刻总市值，$NI_{q}$ 为最近第 $q$ 个季度的归母净利润。

$$
PE_{ttm} = \frac{P_t}{\sum_{i=0}^{3} NI_{t-i}}
$$

### 2.3 盈利收益率 (E/P - Earnings Yield) *[因子计算标准]*
为了解决 $PE$ 分母为负数（亏损）或接近 0 时数值发散的问题，量化模型通常使用其倒数。

$$
EP_{ttm} = \frac{1}{PE_{ttm}} = \frac{\sum_{i=0}^{3} NI_{t-i}}{P_t}
$$

### 2.4 交易信号 (Signal Generation)
通常用于截面多因子排序，而非简单的择时。
$$
Signal_t = 
\begin{cases} 
1 (Long), & \text{if } EP_{ttm} \in \text{Top } 10\% \text{ (High Value)} \\
-1 (Short), & \text{if } EP_{ttm} \in \text{Bottom } 10\% \text{ (Overvalued)}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **数据频率**：通常为日频价格 + 季频财报。
* **调整方式**：需使用**行业中性化 (Industry Neutralization)** 处理。因为银行股 PE 天生低 (5x)，科技股 PE 天生高 (50x)，直接比较没有意义。

## 4. 实战注意 (Notes)
1.  **周期股陷阱**：对于强周期行业（航运、钢铁），**"低 PE 往往是顶，高 PE 往往是底"**。因为在周期顶点盈利极好导致 PE 极低，但随后业绩会断崖式下跌。
2.  **非经常性损益**：建议使用**扣非净利润**计算 PE，排除卖房、政府补贴等一次性收入的干扰。
3.  **负值处理**：使用 $EP$ 因子时，亏损公司的 $EP$ 为负值，自然排在尾部，逻辑上是自洽的。
