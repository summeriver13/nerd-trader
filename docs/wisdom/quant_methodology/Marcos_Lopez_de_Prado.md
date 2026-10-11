---
name: Marcos López de Prado
nickname: The Backtest Skeptic (回测怀疑论者)
born: 1968
role: Principal at AQR (Head of Machine Learning), Research Fellow at Lawrence Berkeley National Laboratory, Professor at Cornell
key_concepts:
  - Deflated Sharpe Ratio (DSR，通缩夏普比)
  - Probability of Backtest Overfitting (PBO，回测过拟合概率)
  - Combinatorially Symmetric Cross-Validation (CSCV)
  - Purged K-Fold CV & Embargo (净化交叉验证与禁运期)
  - Sample Uniqueness (样本唯一性权重)
  - Dollar Bars (美元棒) 与信息驱动采样
  - Triple-Barrier Labeling & Meta-Labeling (三重障碍标签与元标签)
  - Fractional Differentiation (分数阶差分)
  - The 7 Reasons Most Machine Learning Funds Fail
---

# 马科斯·洛佩斯·德·普拉多 (Marcos López de Prado)

> "The first rule of backtesting: **backtesting is not a research tool. Feature importance is.**"
> **“回测不是研究工具，特征重要性才是。”**

> "In finance, the hardest problem is not prediction. **It is validation.**"
> **“金融里最难的问题不是预测，是验证。”**

## 1. 人物简介

双博士（金融经济学 2003、数学金融 2011），前 Guggenheim Partners 高级董事总经理，管理 130 亿美元机构资金；
现 AQR 机器学习主管，劳伦斯伯克利国家实验室研究员，康奈尔 ORIE 5256 课程主讲。
SSRN 最高被引作者之一，7 项算法交易国际专利。

**他的独特性在于**：绝大多数量化作者写“怎么找信号”，他写**“怎么知道你的信号是假的”**。
《Advances in Financial Machine Learning》（2018，中译《金融机器学习》）本质上不是一本讲 ML 的书，
而是一本讲**金融数据为什么会让通用 ML 方法系统性失效**的书。

对以预测为业的研究者，他的价值不是提供 alpha，而是**提供否决权**。

---

## 2. 核心命题：金融 ML 不是通用 ML

通用 ML 的前提是 IID 样本。金融数据**处处违反**这个前提：

| 通用 ML 假设 | 金融现实 |
| --- | --- |
| 样本独立同分布 | 标签由重叠的未来收益决定，天然不独立 |
| 特征与标签一一对应 | 一个标签横跨多期观测，无法归因 |
| 训练集与测试集可分 | 序列相关 + 标签重叠 → 泄漏 |
| 特征平稳 | 价格有记忆但不平稳，收益平稳但无记忆 |
| 可以反复用同一份数据 | 反复用同一份数据 = 数据挖掘 |

**所以金融 ML 的失败不是因为模型不够强，而是因为验证环节被系统性破坏。**
他的整套方法论（DSR、PBO、purge/embargo、唯一性权重）都是为修复验证环节而设计的。

---

## 3. 七宗罪：逐条对照本仓库

> 来源：《The 7 Reasons Most Machine Learning Funds Fail》(2017, SSRN 3031282，即书中第 22 章)。
> 下表右列是**本仓库的真实状态**，不是泛泛而谈。

| # | de Prado 的七宗罪 | 本仓库现状 | 判定 |
| --- | --- | --- | --- |
| 1 | **Sisyphus paradigm**：每人被要求独立产出一个策略，结果要么是过拟合的假阳性，要么是“有学术支持的拥挤标准因子” | 八条研究线七条判死，全部建立在 OHLCV + 资金费率这份人人能下载的最拥挤数据上；六个标准因子全部跑输 BTC 买入持有 | **命中**（形态一致） |
| 2 | **Integer differentiation**：收益率平稳但无记忆，价格有记忆但不平稳；d=1 是“过度差分（无记忆）序列” | 所有因子都是整数窗口差分（`momentum(w)`/`volatility(w)`）或水平量；**从未试过分数阶差分** | **命中** |
| 3 | **Inefficient sampling**：信息不按恒定熵率到达；时间棒每个观测的信息含量相差极远。解药：tick/volume/**dollar**/volatility/order-imbalance/entropy bars | 全部是时间棒（1d/1h/1m），无一例外 | **命中** |
| 4 | **Wrong labeling**：固定时间跨度标签（a）时间棒统计性质差（b）阈值不看波动率（c）**完全忽略价格路径**。解药：三重障碍 + 元标签 | 所有研究都用固定期限前向收益（h=1/3/30 根） | **命中** |
| 5 | **Weighting of non-IID samples**：“血样被倒进后面 9 根试管”——标签重叠时无法归因。解药：按标签唯一性加权 | `non_overlapping_t` 用 `series.iloc[::horizon]` 只取非重叠子样本：**有效但不是他的做法**——丢掉 h−1/h 的数据（h=30 时丢 97%），而唯一性加权用全部数据加权 | **部分命中** |
| 6 | **Cross-validation leakage**：k-fold 在金融里失败是因为观测非 IID；解药 purge（剔除标签时间重叠的训练样本）+ embargo（禁运测试集后紧邻观测，h≈0.01T） | 仓库有 CSCV（`overfitting.py`，用于 `strategy_suite`/`carry_ranking`/`pairs_study`），但**分块直切、无 purge 也无 embargo**；因子路径（`factor_suite`/`factor_sweep`）**连 CSCV 都没用** | **部分命中** |
| 7 | **Backtest overfitting**：不顾理论依据地搜索经验发现会放大问题。解药：通缩夏普比（DSR），控制多重试验膨胀 + 数据挖掘 + 非正态 + 样本长度 | 有 DSR（六策略在 BTC/ETH 上全 FAIL，最好 0.875），但 **N 从未被诚实计数**——DSR 的 N 是“独立试验数”，仓库从未统计过历史上总共试过多少次 | **部分命中** |

**5 条命中、2 条部分命中。** 这不是书评，是体检结论。

---

## 4. 核心工具（按本仓库的优先级排）

### 4.1 PBO / CSCV —— 唯一可识别的“我过拟合了吗”

**做法**：把 N 个候选策略的收益矩阵切成 S 块；每种“S/2 块作训练、其余作测试”的组合，
找出训练集里最优的候选，看它在测试集里的排名是否跌破中位数。
**PBO = 跌破中位数的比例。**

**为什么它比“校正门槛”重要**：见 §5。FDR 无法从样本内统计量识别，但 PBO 是**可估的**——
它不需要知道“有多少个真效应”，只需要问“选出来的那个在没见过的数据上还排得上号吗”。

### 4.2 Purged K-Fold CV + Embargo

purge：把标签时间区间与测试集重叠的训练样本剔除。
embargo：再禁运测试集之后紧邻的一小段（h≈0.01T），因为序列相关会让边界处的样本泄漏。
**本仓库完全没有这一层。**

### 4.3 Dollar Bars

**为什么时间棒最差**：信息不是按恒定熵率到达的。按固定时间采样，每个观测的信息含量相差极远，
导致（a）收益序列自相关（b）异方差（c）统计性质差。
**为什么美元棒最好**：按成交金额累积采样，采样频率天然稳定（作者用 E-mini S&P 500 三种棒对比过）。

**对本仓库的特殊意义**：1m 数据已在本地（30 只永续、6100 万根），美元棒**可以直接构建**，
且它能回答一个当前无法回答的问题——**“3 根反转”这个尺度是信息尺度还是日历尺度？**

### 4.4 Triple-Barrier + Meta-Labeling

三重障碍：止盈、止损、时间三道障碍，先碰到哪道就取哪道的标签（把**路径**纳入标签）。
元标签：主模型只定**方向**，次模型定**要不要出手、出多大手**（0 到 1 的仓位）。

**对本仓库的特殊意义**：核心结论是“反转有边际但被 taker 成本吃掉，maker 条件化挂单能救回一部分”。
元标签回答的正是“给定主信号，这次该不该出手”——与条件化挂单**是同一个问题**，
但有一套现成的建模框架，而不是手写一个挂单探针。

### 4.5 Sample Uniqueness

平均唯一性 ūᵢ = 标签存活期内 1/concurrency 的均值；样本权重再按可归因的绝对对数收益加权。
**对比本仓库**：`iloc[::horizon]` 是“丢弃”，唯一性加权是“打折”——后者不丢数据、且**没有偏移自由度**。

### 4.6 Fractional Differentiation

把价格差分到 d<0.5，得到**既平稳又保留记忆**的序列。作者原话：
“多数金融研究基于 d=1 的收益率，意味着几十年来大部分金融研究建立在**过度差分（无记忆）**的序列上，
导致虚假预测与过拟合。”**本仓库从未试过。**

---

## 5. 对本仓库最锋利的一刀：我刚建的因子扫描器错在哪

**背景**：`src/research/factors/sweep.py` 扫了 468 个候选，用“有效试验数”（从 IC 序列相关矩阵的
特征值谱算 N_eff = 3.3）把门槛从名义 Bonferroni 的 3.87 降到 2.46，然后报出 33 个“平台”。

**问题**：de Prado & Fabozzi (2026)《The False Discovery Rate in Finance: **Identification Failure** and
Search-Adjusted Estimation》证明：

> FDR **cannot be identified from in-sample statistics alone** when reported results arise from a
> search-and-selection process.
> —— 当结果来自搜索-选择过程时，FDR 无法仅从样本内统计量识别。

并明确指出：此前那些“从已发表统计量的横截面推断出低 FDR”的方法**不成立，会严重低估真实假发现率**。

**逐条对照**：

1. **不可识别**：N_eff 完全由样本内 IC 相关矩阵导出，正属于被证明不可识别的那一类。
   而且论文说这类方法**低估**假发现率——我的校正方向恰好是**放松**门槛（3.87 → 2.46），即低估方向。
2. **N 不是“我试过的次数”**：N_eff 只数了本次扫描**内部**的冗余，没数历史上试过的所有东西
   （六个因子 × 2 市场 × 多期限、carry、配对、meme、上新币、maker……）。
   DSR 的 N 是“独立试验数”，而“独立”指的是**统计上的独立**，不是“同一次脚本里跑的”。
3. **没有样本外**：平台判据是稳健性的**代理**。§4.1 的 PBO/CSCV 才是**可识别**的那个量。
4. **零假设校准是真的，但它是模拟、不是识别**：两个纯噪音族 6 个组合零平台，这是真证据；
   但它依赖**我自己选的**零假设族（iid 噪音 + 滑动平均噪音），换一个带截面结构的零假设，结论可能不同。

**最讽刺的一点**：仓库**早就有** PBO/CSCV 与 DSR（`src/research/overfitting.py`，实现与
§4.1 一致，且 `print_overfitting` 里已经写了诚实提示“若历史上还试过其他参数/窗口，
请用 `--trials` 声明真实试验次数”），`strategy_suite`、`carry_ranking`、`pairs_study` 都在用。
**我的扫描器把整套现成机制绕过去了，自己造了一个没有理论保证的门槛。**
这正是 de Prado 批评的“研究流程没有质检环节”——不是因为缺工具，是因为没接上。

**结论**：扫描器的**发现**（`vol` > `mom` > `liq` > `dth` 的稳健性排序）可能是对的，
但 **“门槛 2.46”这个数字没有理论保证**，不该被当作“已校正的显著性”来读。

---

## 6. 必读著作（Library 索引）

> *建议配合 `docs/library/books/` 阅读*

1. **《Advances in Financial Machine Learning》**（2018，中译《金融机器学习》）
   *定位*：**总纲**。§3 的七宗罪、§4 的六个工具全出自此书。配套讲义免费：
   <https://quantresearch.org/Lectures.htm>（Cornell ORIE 5256，10 讲）。
2. **《Machine Learning for Asset Managers》**（2020，Cambridge）
   *定位*：**精简版**。作者自称“本书是给没时间读完 400 页的人的”。
   核心贡献：**聚类特征重要性**（特征相关时重要性会虚高，须在聚类层面算）、HRP 分层风险平价。
3. **《The 7 Reasons Most Machine Learning Funds Fail》**（2017，SSRN 3031282）
   *定位*：**体检表**。45 页幻灯片，就是 §3 那张表。
4. **《The Probability of Backtest Overfitting》**（Bailey, Borwein, López de Prado, Zhu）
   *定位*：**PBO/CSCV 原始论文**。参考实现：<https://www.quantresearch.org/Software.htm>（`CSCV_*.py`）。
5. **《The Deflated Sharpe Ratio》**（2014）
   *定位*：**多重试验校正的实操公式**。参考实现同上（`DSR.py`）。
6. **《The False Discovery Rate in Finance》**（2026, ADIA Lab，与 Fabozzi）
   *定位*：**最新、最锋利**。证明 FDR 的不可识别性，推翻一批“大多数因子是真的”的结论。
   复现代码：<https://github.com/lopezdeprado/FDR-in-Finance>。

---

## 7. 对我们的启示

如果说 Ernest Chan 是**工程手册**（怎么把回测写对），de Prado 就是**质检标准**（怎么知道回测结果是假的）。

**他的方法与我们的处境有一处尖锐的错配**：他的工具为**管理大资金、需要向投资人解释**的机构设计
（DSR 用于投委会、PBO 用于尽调、purge/embargo 用于模型上线）。我们是一个人做研究，
不需要向谁解释。所以**不要照抄全部**——§4 里真正该立刻做的是：

1. **PBO/CSCV**（§4.1）——它直接回答“我刚做的扫描算不算数据挖掘”，成本低，且是**唯一可识别**的量。
2. **Dollar Bars**（§4.3）——1m 数据已在本地，它回答“反转的尺度是信息尺度还是日历尺度”，
   而这个问题**当前所有结论都建立在未经验证的日历尺度上**。
3. **诚实计数 N**（§3 第 7 条）——把历史上试过的所有试验记进一个总数，DSR 才有意义。

**暂时不做的**：FFD 与三重障碍标签在**现有数据上**属于“换个特征表示”，
而我们已经知道现有数据没有漏掉的新东西（见 `docs/factors/README.md` 第三节）——
它们值得做，但排在“换数据源”（清算流自采）之后。
