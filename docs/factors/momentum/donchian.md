# [Donchian] 唐奇安通道 (Donchian Channel)

## 1. 核心逻辑 (Core Logic)
* **因子类别**：动量 / 趋势 (Momentum / Trend)
* **核心假设**：
    1.  **新高即买入**：价格突破过去 N 天的最高价，意味着多头力量彻底压倒了空头，这不仅是以前的阻力位，更是新趋势的起点。
    2.  **价格位置**：收盘价在通道内的相对位置越高，动量越强。
* **原始出处**：Richard Donchian (1970s) - *The precursor to Turtle Trading*

## 2. 数学定义 (Mathematical Formulation)

### 2.1 基础轨道定义
设窗口期为 $N$（通常 20 或 55）。
注意：为了避免**未来函数**，计算上轨和下轨时通常使用**前一日**的数据（Lag 1），但在计算“突破因子”时，是拿**今日**收盘价与轨道比较。

$$
Upper_t = \max(H_{t-1}, H_{t-2}, \dots, H_{t-N})
$$

$$
Lower_t = \min(L_{t-1}, L_{t-2}, \dots, L_{t-N})
$$

$$
Mid_t = \frac{Upper_t + Lower_t}{2}
$$

### 2.2 因子化改造 (Factor Construction) *[重点]*
单纯的 $Upper_t$ 是一个绝对价格（比如 60000 美元），无法在不同股票间比较。我们需要构建**相对强度因子**：

#### 形式一：通道位置因子 (Donchian Channel Position)
类似于布林带的 %B 指标。衡量当前价格在过去 N 天极值范围内的位置。
$$
Factor_{pos} = \frac{Close_t - Lower_t}{Upper_t - Lower_t}
$$
* **值域**：通常在 $[0, 1]$ 之间。
* **含义**：越接近 1，说明离创新高越近，动量越强；越接近 0，说明离创新低越近。
* **突破**：如果 $>1$，说明今日创新高（最强动量）。

#### 形式二：均线距离因子 (Distance to Mid)
衡量价格偏离中轨的程度，并用通道宽度标准化。
$$
Factor_{trend} = \frac{Close_t - Mid_t}{Upper_t - Lower_t}
$$
* **值域**：$[-0.5, 0.5]$。正值代表多头，负值代表空头。

### 2.3 交易信号 (Signal Generation)
$$
Signal_t = 
\begin{cases} 
1 (Long), & \text{if } Close_t > Upper_t \text{ (Breakout)} \\
-1 (Short/Stop), & \text{if } Close_t < Lower_t \text{ (Breakdown)} \\
0 (Hold), & \text{otherwise}
\end{cases}
$$

## 3. 参数说明 (Parameters)
* **$N$**：回顾周期。
    * **20日**：经典短周期（月线级别）。
    * **55日**：经典长周期（季度级别，海龟法则常用）。

## 4. 实战注意 (Notes)
1.  **归一化的必要性**：如果你做截面多因子选股（Cross-sectional），必须使用 **2.2 中的因子公式**。绝对价格无法排序。
2.  **波动率过滤器**：唐奇安通道在波动率极低（通道极窄）时的突破往往是假突破。建议构建一个 **通道宽度因子 (Channel Width)**：
    $$Width = \frac{Upper_t - Lower_t}{Close_t}$$
    * 仅当 $Width$ 大于某个阈值（例如历史分位 20%）时的突破才有效。
