# [ROE] 净资产收益率 (Return on Equity)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：质量 / 盈利能力 (Quality / Profitability)
* **核心假设**：
    1.  **巴菲特定律**：长期来看，股票的年化回报率约等于其 ROE 水平。
    2.  **复利效应**：高 ROE 的公司能更高效地利用留存收益进行再投资，从而实现指数级增长。
    3.  **抗跌性**：在熊市中，高 ROE 公司通常拥有更深的护城河和现金流，表现优于低质量公司。
* **原始出处**：DuPont Model (1920s) / Warren Buffett

## 2. 数学定义 (Mathematical Formulation)

### 2.1 基础公式 (Basic ROE)
设 $NI_t$ 为归属于母公司股东的净利润，$E_t$ 为归属于母公司股东权益（平均净资产）。

$$
ROE = \frac{NetIncome}{Average\ Equity} \times 100\%
$$

### 2.2 杜邦分析分解 (DuPont Analysis) *[量化核心]*
单纯看 ROE 数值容易被误导，量化模型通常将其拆解为三部分，以识别收益的来源：

$$
ROE = \underbrace{\frac{NetIncome}{Sales}}_{\text{销售净利率}} \times \underbrace{\frac{Sales}{TotalAssets}}_{\text{资产周转率}} \times \underbrace{\frac{TotalAssets}{Equity}}_{\text{权益乘数}}
$$

1.  **销售净利率 (Profit Margin)**：产品卖得贵不贵？（茅台模式）
2.  **资产周转率 (Asset Turnover)**：管理层运营快不快？（沃尔玛模式）
3.  **权益乘数 (Financial Leverage)**：杠杆加得高不高？（银行/地产模式）

### 2.3 信号生成 (Signal Generation)
* **高质量筛选**：通常要求 $ROE_{ttm} > 15\%$ 且连续 3 年保持。
* **趋势筛选**：$ROE_{this\_year} > ROE_{last\_year}$ (加速增长)。

## 3. 参数说明 (Parameters)
* **数据频率**：季频。通常使用 **ROE-TTM**（滚动4季度净利润 / 平均净资产）。
* **加权方式**：建议使用**加权平均净资产收益率**，以考虑增发、回购对净资产时间加权的影响。

## 4. 实战注意 (Notes)

### 4.1 杠杆陷阱 (The Leverage Trap)
* **现象**：某些垃圾公司 ROE 很高，不是因为赚得多，而是因为**负债极高**（分母 Equity 极小）。
* **风控**：在使用 ROE 因子时，**必须剔除高杠杆公司**。如果 ROE 的提升纯粹来自于权益乘数（借钱），这类公司在加息周期会暴雷。

### 4.2 一次性收益 (Non-recurring Items)
* **现象**：公司卖了一栋楼，净利润暴增，导致 ROE 瞬间飙升到 50%。
* **解法**：量化计算时，务必使用 **ROE (扣非)**，即扣除非经常性损益后的净利润。

### 4.3 PB-ROE 估值体系
* ROE 单独使用效果一般，**ROE + PB** 才是王炸。
* **逻辑**：我们希望买到“高 ROE”且“低 PB”的股票。
* **公式**：
  $$High\ Quality\ Value = \frac{ROE}{PB}$$
  *(这其实就是 PE 的倒数，但在逻辑上更强调质量与价格的匹配)*
