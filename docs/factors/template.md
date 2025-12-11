# [因子英文名] 因子中文名

## 1. 核心逻辑 (Core Logic)
* **因子类别**：[动量 / 价值 / 波动率 / 情绪 / 微观结构]
* **核心假设**：[简述因子为什么能赚钱，例如：价格围绕均值波动，偏离过大必回归]
* **原始出处**：[研报名称 / 论文引用]

## 2. 数学定义 (Mathematical Formulation)

### 2.1 基础变量定义
设 $P_t$ 为 $t$ 时刻的收盘价，$H_t$ 为最高价，$L_t$ 为最低价，$V_t$ 为成交量。
时间窗口为 $N$。

### 2.2 计算公式
$$
[在此处插入核心公式]
$$

### 2.3 信号生成逻辑
$$
Signal_t = 
\begin{cases} 
1 (Long), & \text{if } Indicator_t > Threshold_{upper} \\
-1 (Short), & \text{if } Indicator_t < Threshold_{lower} \\
0 (Neutral), & \text{otherwise}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N$**：计算周期，默认值 [例如 20]。
* **$K$**：阈值系数，默认值 [例如 2.0]。

## 4. 实战注意 (Notes)
* [数据清洗要求，例如：需使用复权价格]
* [适用标的范围]
