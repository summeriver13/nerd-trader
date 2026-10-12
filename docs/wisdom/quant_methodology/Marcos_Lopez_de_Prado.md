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

> 来源：《The 7 Reasons Most Machine Learning Funds Fail》(2017, SSRN 3031282)。
> **注意：它不是书里的某一章**——那是 2017 年独立的演示，也是 Cornell 讲义的
> Lecture 10/10；书的第 22 章是「High-Performance Computational Intelligence」。
> 同一份材料已扩写成 **《The 10 Reasons Most Machine Learning Funds Fail》**
> （GARP 白皮书），新增的 Pitfall #6 是「**同时学方向和仓位**」——正是本仓库
> maker 那条线的核心问题，尚未逐条审计。
> 下表右列是**本仓库的真实状态**，不是泛泛而谈。

| # | de Prado 的七宗罪 | 本仓库现状 | 判定 |
| --- | --- | --- | --- |
| 1 | **Sisyphus paradigm**：每人被要求独立产出一个策略，结果要么是过拟合的假阳性，要么是“有学术支持的拥挤标准因子” | 八条研究线七条判死，全部建立在 OHLCV + 资金费率这份人人能下载的最拥挤数据上；六个标准因子全部跑输 BTC 买入持有 | **命中**（形态一致） |
| 2 | **Integer differentiation**：收益率平稳但无记忆，价格有记忆但不平稳；d=1 是“过度差分（无记忆）序列” | 所有因子都是整数窗口差分（`momentum(w)`/`volatility(w)`）或水平量；**从未试过分数阶差分** | **命中** |
| 3 | **Inefficient sampling**：信息不按恒定熵率到达；时间棒每个观测的信息含量相差极远。解药：tick/volume/**dollar**/volatility/order-imbalance/entropy bars | 全部是时间棒（1d/1h/1m），无一例外 | **命中** |
| 4 | **Wrong labeling**：固定时间跨度标签（a）时间棒统计性质差（b）阈值不看波动率（c）**完全忽略价格路径**。解药：三重障碍 + 元标签 | 所有研究都用固定期限前向收益（h=1/3/30 根） | **命中** |
| 5 | **Weighting of non-IID samples**：“血样被倒进后面 9 根试管”——标签重叠时无法归因。解药：按标签唯一性加权 | `non_overlapping_t` 用 `series.iloc[::horizon]` 只取非重叠子样本：**有效但不是他的做法**——丢掉 h−1/h 的数据（h=30 时丢 97%），而唯一性加权用全部数据加权 | **部分命中** |
| 6 | **Cross-validation leakage**：k-fold 在金融里失败是因为观测非 IID；解药 purge（剔除标签时间重叠的训练样本）+ embargo（禁运测试集后紧邻观测，h≈0.01T） | 仓库有 CSCV（`overfitting.py`，用于 `strategy_suite`/`carry_ranking`/`pairs_study`），但**分块直切、无 purge 也无 embargo**；因子路径（`factor_suite`/`factor_sweep`）**连 CSCV 都没用** | **部分命中** |
| 7 | **Backtest overfitting**：不顾理论依据地搜索经验发现会放大问题。解药：通缩夏普比（DSR），控制多重试验膨胀 + 数据挖掘 + 非正态 + 样本长度 | 有 DSR（六策略在 BTC/ETH 上全 FAIL，最好 0.875），但 **N 从未被诚实计数**——DSR 的 N 是“独立试验数”，仓库从未统计过历史上总共试过多少次（本轮已修：`src/research/trials.py` 账本，N=6625） | **部分命中 → 已修** |

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

**一处必须澄清的错配**（本节早期版本的说法已撑销）：曾经写过「他的工具为需要向
投委会解释的机构设计，我们不需要」——那是给自己跳过难的部分发许可证。核对原文后
的结论相反：机构的激励是夸大，投委会与尽调是反向约束；**单人研究者的激励一模一样，
却没有任何反向力量**。所以「不需要向投委会解释」不是「不需要校正」的理由，
而是校正**更没人替你执行**的理由。真正属于机构专属的只有组织方案（research factory、
流水线拆子任务），统计校正（DSR、PBO/CSCV、purge/embargo）完全适用且对单人更适用。

**已完成**（本轮）：

1. **PBO/CSCV**（§4.1）——已接进因子路径（`src/research/factors/verification.py`），
   并且区分了「平台代表窗口池」与「全候选池」两个问题。
2. **Dollar Bars**（§4.3）——已建（`src/research/bars.py`，30 只 / 42,421 根），
   尺度之争已答：**反转是日历尺度的现象，不是信息尺度的**
   （同池日历 IC 0.0354 → 同池信息 0.0048，见 `docs/factors/README.md` 第九节）。
3. **purge / embargo**（§4.2）——已进 CSCV（`src/research/overlap.py`）。
   实测：非重叠采样下 purge 恒为空操作（标签首尾相接），但 **embargo 真的改数字**
   （现货 h=1 从 23.8% → 31.0%、h=2 从 15.9% → 20.2%），且方向正确——原值偏乐观。
   全候选池已用新口径重算：**裁决零翻转**。
4. **SADF 结构突变**（§4.7，Ch 17）——已建（`src/research/breaks.py`，
   与 `statsmodels.adfuller` 逐位对拍，前缀和把每个窗口降到 O(1)、快 79 倍）。
   信号层三重控制全过、参数平台漂亮、组合层年化 45~53%——
   但 **PBO 52.4/60.3/85.0% 全不过、DSR 全 0，不采信**。详见
   `docs/factors/README.md` 第十节。这是「看上去很好、验证层不过」的完整演示。
5. **三重障碍**（§4.4，Ch 3）——已建（`src/research/labels.py`）。障碍按波动率标度、
   首次触及定 `t1`、标签是触及时刻收益的符号；与朴素逐事件循环逐位对拍。
6. **唯一性权重**（§4.5，Ch 4）——已建（`overlap.uniqueness_from_ends`，
   可变长度版）。**已接线并实测**，但结论是它 **不能当结论**：
   20 棵树冒烟跑里 C 胜 B、200 棵树全量跑里 B 胜 C，差异在噪声量级。
7. **诚实计数 N**（§3 第 7 条）——已建（`src/research/trials.py`，append-only 账本
   + `scripts/trials_ledger.py`）。接线后 `verify()` 的 N 默认取账本累计数，
   scope 为空且未声明 `--trials` 时**打印警告**（低估从此是可见的）。
   当前账本 **N = 6625**（扫描器 3276×2 + SADF 58 + 元标签 15）。
   对已有结论的影响是**把门槛抬高、裁决不变**：那些 DSR 本来就在最宽松的 N 下
   （本批 3~7 或 468）不通过，换成 6625 只会更低。账本是**下界**——建账前的零星
   尝试（carry、配对、maker、美元棒、反转）没有完整网格，一律不计。
   诚实说清的弱点：`data/` 整个被 `.gitignore` 忽略，账本**不防篡改**，
   靠「代码自动记录 + 每次打印 N」保证看得见，而不是靠改不动。

7. **Bet Sizing**（§4.5，Ch 10）——已建（`src/research/sizing.py`）。**结论是负的**：
   连续定仓在三个变体上一致输给硬阈值（净夏普 0.135~0.312 对 0.469~0.627）——
   概率**在阈值上有信息、阈值内部没有排序信息**。唯一站得住的细化是网格量化（降换手）。
   顺带抓到一个单位陷阱：把 h 日前向收益当每期收益会让夏普被重叠抬高（927→429bp/期）。

8. **特征重要性**（Ch 8）——已建（`src/research/importance/` 包：MDA / 聚类 MDA /
   MDI / SFI）。跑在元标签模型上，最干净的一条是 **SFI：21 个（7 特征 × 3 变体）
   读数全部为负**，单特征模型单独用时 log loss 中位 0.69367、比什么都不看的
   0.69315 还差——**没有任何特征能单独工作**。而 MDA 在完整模型上给动量簇 t 3.12、
   MDI 给每个特征都分重要性。SFI 按定义看不到联合效应，所以它排除的是「模型靠某个
   特征」，不是「模型有信息」。另有一条稳的负读数：`age_days` 单独用 t −5.7
   （灵活模型 + 弱特征 = 只有过拟合）。**PCA 正交也已做**（`src/research/importance/pca.py`）：
   三个变体的 τ（有监督 MDA 排序 vs 逆 PCA 排名）分别是 **−0.679 / −0.562 / −0.344**，
   **确认性检验不通过**——数据的主变异方向（`dth_60`、`sadf`，恰好是 MDA 判负的两个）正是
   模型说没用的方向。Ch 8 的工具到此齐了（MDI / MDA / 聚类 MDA / SFI / PCA 正交）。

**待做**（按性价比）：

1. **FFD**（§4.6）——现有数据上属于「换个特征表示」，排在换数据源之后。

**一条被推翻的施工计划**：曾把「唯一性加权（Ch 4）」列为可独立做的下一步，
**错了**。实测在固定期限标签下，`Σ唯一性` 与抽稀的 `T/h` **逐行相等**
（h=2 时 1500.2 vs 1500.0、h=30 时 100.5 vs 100.0）；美元棒这种不规则采样下
唯一性的变异系数也只有 0.017~0.095（因为平均每天一根，h 根标签的并发数就是 h）。
根因：唯一性只在**标签长度可变**时才有内容，而固定期限标签长度恒为 h。
**让标签长度可变的正是 Ch 3 的三重障碍**——所以 Ch 4 不能独立实现，
它是 Ch 3 的推论，顺序必须先 Ch 3 后 Ch 4。这个判断已被实测证实：
接上三重障碍后唯一性的标准差从 **0 变成 0.1737**，Ch 4 才真的有东西可算。
