# [Model ID] 模型名称 (Model Name)

| 属性 | 描述 |
| :--- | :--- |
| **模型类型** | [Alpha预测 / 风险控制 / 交易执行] |
| **核心算法** | [OLS / XGBoost / Transformer / GARCH] |
| **输入维度** | [例如：158个因子] |
| **输出目标** | [例如：未来 5 日收益率 / 股票排名] |
| **最后更新** | 2025-12-11 |

## 1. 核心逻辑 (Architecture)
* **设计目的**：[例如：利用非线性算法挖掘因子间的交互作用]
* **算法原理**：[简述算法核心，如：基于梯度提升决策树 (GBDT)]
* **拓扑结构**：
    

[Image of neural network architecture diagram]

    *(如果是深度学习，最好放一张网络结构图)*

## 2. 特征工程 (Feature Engineering)
### 2.1 输入特征 ($X$)
* **因子列表**：[引用 docs/factors/ 中的因子，如 RSRS, MACD, PE]
* **预处理**：
    * **缺失值**：[例如：填充均值 / 0]
    * **去极值**：[例如：3倍标准差截断 (Winsorize)]
    * **标准化**：[例如：Z-Score Standardize]

### 2.2 预测目标 ($Y$ / Label)
* **定义**：$Return_{t+5} = \frac{Close_{t+5}}{Close_t} - 1$
* **处理**：[例如：按行业去均值，或转换为 0-9 的排序分 Rank]

## 3. 训练流程 (Training Process)
* **数据划分**：
    * 训练集：2015-2020
    * 验证集：2021
    * 测试集：2022-2024
* **损失函数 (Loss Function)**：[例如：MSE (均方误差) / NDCG (排序误差)]
* **超参数 (Hyperparameters)**：
    * `learning_rate`: 0.01
    * `max_depth`: 6
    * `n_estimators`: 1000

## 4. 性能评估 (Performance Metrics)
*(这里记录模型本身的统计学表现，而不是回测赚钱的表现)*
* **IC (Information Coefficient)**：预测值与真实值的相关系数。 [例如：0.05]
* **Rank IC**：排序相关系数。 [例如：0.06]
* **准确率 (Accuracy)**：[仅适用于分类模型]

## 5. 部署与调用 (Deployment)
* **模型文件路径**：`data/models/xgb_v1.json`
* **推理代码示例**：
    ```python
    model = XGBRegressor()
    model.load_model("data/models/xgb_v1.json")
    score = model.predict(current_factors)
    ```
