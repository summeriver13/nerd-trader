# 预测市场（Polymarket）：数据可达性侦察与前向采集

本页回答一个前置问题——Polymarket 的历史与实时数据到底拿不拿得到，以及拿不到的部分
有没有替代口径。结论决定这条线值不值得开，以及开的话该采什么。侦察方式为端点直探
（2026-10-14，经本机代理，`http://127.0.0.1:7897`），不落依赖、先不写采集代码。

## 结论速览

- **历史全封**：已结算市场的价格序列**拿不到**——实测 7/7 只返回 0 点（显式 `startTs/endTs`、
  fidelity 60/1440 全试过），而价格历史只服务活跃市场、只回溯约 30 天。
- **回填的第二条路也断了**：官方 2026-04-28 迁 v2 合约后**不再提供 subgraph**
  （Goldsky 文档明说公开端点会返回不完整或错误数据）；`/trades` 的 `before`/`endTs`/`to`/`until`/`maxTs`
  等**时间过滤参数全部被静默忽略**（8 个参数返回逐条相同的页），没有按时间翻页的入口。
- **于是只剩前向采集**：今天起每天抓活跃市场，等它们在采集期内结算，才有了标签。
- **唯一的相对优势切片是 crypto 阈值市场**：「BTC 某日前突破 $X」等价于数字期权，
  无套利界与模型价可以用本项目自有的加密数据算出来；政治/体育方向是信息战与基建战，不属量化。
- **规则口径已核实**（见 §七）：判定源是 Binance 的 **1 分钟 High/Low**、`immediately resolve`
  即触碰即结算（one-touch）、窗口起点是各市场的创建时刻。离散监控修正只有 0.1¢ 级，
  所以「当前现货 + 剩余期限」的连续监控公式就是正确口径。
- **费率已经不是零**：`Fee = Θ × C × p × (1−p)` 的二次型——买 50¢ 的合约约 3.85% 名义额，
  买 95¢ 的约 0.35%，极端价这一侧（也正是长尾偏差 FLB 所在的一侧）费率便宜一个数量级。
- **产出定位是校准研究，不是可交易研究**：长尾市场点差最宽、单市场深度有限，
  且历史 L2 盘口拿不到，成本只能估上界。
- **三口径面板已固化**（见 §八）：上行阶梯 8/8 档都比 Deribit 贵 0.4~3.2¢ 且全在模型
  误差带之外；下行阶梯反而比 Deribit 便宜 1.2~5.7¢、但比历史 bootstrap 贵 2.4~6 倍——
  夹在两个参照之间，无一致方向。

## 一、端点实测

| 能力 | 端点 | 实测 |
| --- | --- | --- |
| 事件与市场元数据 | `gamma-api.polymarket.com/events?tag_slug=crypto` | 含标签、内嵌市场、费率（`feesEnabled`/`takerBaseFee`/`makerBaseFee`）、流动性 ✓ |
| 枚举上限 | Gamma `/markets` 单查询 | `offset ≤ 2000`（2100 → HTTP 422）；月度窗口切片可绕过 |
| 价格历史 | `clob.polymarket.com/prices-history` | 只有**活跃**市场、约 **30 天**、逐小时；窗口太长返回 400 `interval is too long` |
| 已结算市场价格 | 同上 | **任何窗口 0 点** ✗（7/7） |
| 成交明细 | `data-api.polymarket.com/trades` | `limit ≤ 1000`，单市场 offset 上限约 1~2 万笔 ✓ |
| 时间窗过滤 | `/trades` 的 8 个候选参数 | **全部被静默忽略** ✗ |
| 子图 | Goldsky / 官方 README | 迁移 v2 后不再提供；README 只给本地开发端点 ✗ |
| 本机连通性 | 直连 vs 代理 | 直连不通、走代理 200（与 OKX/Binance 同一条规矩）；单请求 2~4 秒，偶发 SSL EOF 需重试 |

拿得到的三样东西构成了采集口径：**活跃市场的价格历史（逐小时）、成交明细（含方向与
价格）、市场元数据（含费率与结算字段）**。

## 二、为什么只剩一条窄路

已结算市场没有价格序列，就无法按「价格 → 结算频率」重建历史校准面板。剩下的口径是
用**成交明细 + 结算结果**重建：单市场成交 ≤1~2 万笔可以完整取回，而 FLB（长尾偏差，
longshot-favorite bias）只需要「谁在什么价格买、最后结算成什么」。四个必须先钉死的风险：

1. **长尾 = 点差最宽 = 可交易性最差** → 这套数据能回答「定价是否系统性偏离」，
   不能回答「扣成本后是否赚钱」；
2. **巨型市场只能看到最近 1~2 万笔**（选举、NBA 等）→ 样本会偏向流动性差的市场；
3. **`negRisk` 多结果市场的分组口径**：按父事件分组 vs 按子市场分组会把结论符号翻转
   （文献里 −6.3¢ ↔ +4.1¢），不钉死就是自欺；
4. **替代做法是前向采集**：今天开始每天抓，半年后才有足够已结算样本。

## 三、数据契约（raw 只增不改）

```
data/storage/polymarket/raw/
    events/<YYYY-MM-DD>.jsonl     # 每轮运行的事件快照（含标签），去重键 (id, updatedAt)
    markets/<YYYY-MM-DD>.jsonl    # 市场快照（含费率/流动性/结算字段），键 (conditionId, updatedAt)
    prices/<slug>.jsonl           # 逐小时价格点（token/outcome/t/p），键 (token, t)
    trades/<slug>.jsonl           # 成交明细，键 (transactionHash, asset, timestamp, price, size)
data/manifests/polymarket_raw_{events,markets,prices,trades}.json
```

- 路径统一从 `src/config/paths.py` 取；自持不入湖的依据是容器 ADR-0007（同 ADR-0006 的 crypto）。
- 字段名保留上游原样（camelCase），只做白名单裁剪：一条市场里 `icon`/`events` 嵌套合计 50KB+，
  一条成交里 `icon`/`profileImage`/`bio` 之类展示字段合计约 900B（40 只市场一天的增量就能写成 50MB）。
  `description`/`resolutionSource` 例外保留（约 1.2KB/市场/天）：结算规则只写在这里，
  它是「按 one-touch 口径定价」的凭证（§七）。
- 快照类数据集保留 `updatedAt` 的历史行（同一市场每轮一行），用于回看「当时看到的是什么」。
- manifest 只对本次触及的文件重算 sha256，其余沿用旧条目。

## 四、复现命令

```bash
# 只看会抓哪些市场（不取数）
uv run python scripts/fetch_polymarket.py --list-only --event-pages 3

# 小样本端到端（8 只、1 页成交，约 3 分钟）
uv run python scripts/fetch_polymarket.py --max-markets 8 --event-pages 2 --trade-pages 1

# 日常采集（40 只、3 页成交、6 并发；超过 10 分钟必须后台跑）
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 nohup uv run python -u scripts/fetch_polymarket.py \
    --max-markets 40 --event-pages 4 --trade-pages 3 --workers 6 \
    > logs/polymarket_fetch_$(date -u +%Y-%m-%d).log 2>&1 &

# 枚举抖动时重取明细：直接用当天快照选池，不再打 Gamma
uv run python scripts/fetch_polymarket.py --from-snapshot --skip-prices

# 验收闸（致命项清零才算采集成活；告警逐条要能解释）
uv run python scripts/audit_polymarket.py
```

选池口径：`tag_slug=crypto` 的活跃市场里**成交额前 N**。价格历史默认 7 天窗口、
逐小时粒度，窗口重叠靠去重兜底；成交明细从最新往前翻，翻到「整页都已落盘」即停。

实测（2026-10-14，40 只、并发 6、本机代理 2~4 秒/请求）：**首轮 11 分钟**，价格点
1.35 万行（4.1MB，覆盖 7 天）、成交 5.7 万行（白名单后 539B/行，28MB，回溯到 2026-02-24）；
**第二轮增量 5 分钟**，每只只多 2 个价格点、成交整页命中水位后立即停（这就是每天该有的成本）。

## 五、验收闸与语义校验

`scripts/audit_polymarket.py` 的口径：

| 判定 | 检查项 |
| --- | --- |
| 致命 | JSONL 解析失败、业务键重复、价格越界（不在 (0,1]）、成交量非正、manifest 与磁盘文件数/行数不一致 |
| 告警 | 价格点不足、最新价格点过期（默认 >2 天）、二元市场同一小时桶 YES+NO 偏离 1 超过 0.02、成交行数恰为页大小整数倍（达页上限、更早历史未取）、孤儿 slug |

语义校验（2026-10-14，8 只市场小样本 + 40 只全量）：两个 token 的价格序列按**小时桶**对齐后
`YES + NO = 1.0000`（161 个公共桶，min = max = 1.0000）；逐小时间隔中位 3600~3602 秒；
价格点跨度 6.97 天（7 天窗口）；成交明细价格落在 `[0.003, 0.997]`、买卖两侧都有。

首轮采集的失败率约 5%（丢一条腿：某 token 的价格历史或某页成交连续 5 次请求失败），
所以采集内建**整只重试一次**；重试写盘靠业务键去重，只会补缺口、不会写重复行。
第二轮增量把两个缺腿的市场补齐后，验收闸的致命项与价格告警均清零。

## 六、局限与待办

- **采集中断是不可逆损失**：价格历史只有约 30 天窗口，错过就永久缺失，故审计的过期项
  是「回补机会正在关闭」的提醒；
- **样本要等**：市场结算后才有标签，头几个月的读数样本量不足，任何结论都要标样本期；
- **未接定时任务**：当前按需手动/后台运行，接入 Windows 计划任务属于基建变更，另开决策；
- **容量与成本口径未定**：历史 L2 拿不到，成本只能按点差上界估，回测必然乐观。

## 七、结算规则口径（实测，2026-10-15）

定价公式的合法性取决于结算规则的**确切文字**，而它只写在 Gamma 的 `description` 里（此前被白名单裁掉）。
用 `GET gamma-api.polymarket.com/markets?condition_ids=<conditionId>` 取回 20 只 BTC 阈值市场的
原始记录（原文落 `data/results/polymarket/gamma_raw_rules.json`），得到三条事实。

**1. 判定源与价格类型**（原文，`will-bitcoin-reach-100000-by-december-31-2026-…`）：

> This market will **immediately** resolve to "Yes" if any Binance **1 minute candle** for Bitcoin
> (BTC/USDT) between November 24, 2025, 14:00 and December 31, 2026, 23:59 in the ET timezone has a
> final **"High"** price equal to or greater than the price specified in the title. … The resolution
> source for this market is Binance, specifically the BTC/USDT "High" prices available at
> <https://www.binance.com/en/trade/BTC_USDT>, with the chart settings on "1m" for one-minute candles…

→ 判定用 **1 分钟 K 线的 High/Low**（不是收盘价、不是其他交易所）；`immediately resolve` 确认
**触碰即结算**，即 `src/research/digital_pricing.py` 的 `one_touch()` 口径，而不是末次结算 `N(d2)`。

**2. 离散监控的连续性修正可忽略**（Broadie-Glasserman-Kou 1997：把障碍位移 `β·σ·√Δt`，β = 0.5826）：

| 监控粒度 | 90k 档 | 140k 档 | 55k 档 |
| --- | --- | --- | --- |
| **1 分钟（实测口径）** | **+0.12¢** | **+0.007¢** | **+0.05¢** |
| 1 小时 | +0.92¢ | +0.05¢ | +0.36¢ |
| 日收盘 | +4.45¢ | +0.24¢ | +1.68¢ |

即用连续监控公式的偏差是 0.1¢ 级；此前列为待验证项的「≈4.7¢ 口径差」是**日收盘**粒度的量级，
已被规则文字排除。

**3. 窗口起点 = 各市场自己的创建时刻，但不影响定价。** 100k 档的 `createdAt = 2025-11-24`
与文本里写死的 `November 24, 2025, 14:00` 吻合；90k 档（`createdAt = 2026-02-06`）的文本改口为
`between the creation of this market and …`——两条独立证据一致，写死的日期只是那批市场创建当天的文本。

已触碰检验（自持 1h 数据；口径等价，因为 1h 的 High 就是该小时内所有 1m High 的最大值）：

| 区间 | 最高 | 最低 | 结论 |
| --- | --- | --- | --- |
| 全样本（2019-09 起） | 126,208（2025-10-06） | 3,622 | BTC 历史 ATH 在窗口**之前** |
| 2025-11-24 起（主阶梯窗口） | 97,932 | 57,759 | 全部 reach ≥ 100k、dip ≤ 55k 档**均未触碰** |

第三条独立证据来自价格本身：reach-120k 报价 0.065——若窗口包含 2025-10-06（BTC 触 126,208），
该档应早已是 1.00；它没被结算，即窗口晚于 ATH。90k 档按主阶梯起点算**已触碰**
（97,932 ≥ 90,000，应值 1.00）、按 creation（2026-02-06）算未触碰，而市场报价 0.565 站在后者一侧。
三条证据合起来把「起点 = creation、且不影响未来定价」钉死。

**定价含义**：全部档位都是「窗口已过去 11 个月且至今未触碰」，而 GBM 是马尔可夫的，
条件分布就是从当前现货重新出发的分布，所以 `one_touch(S0, K, 剩余 78 天)` 正是该用的式子，
**不需要**任何路径条件修正；起点只决定「过去是否触碰过」。

回到阶梯构造：`will-bitcoin-dip-to-60k-…-from-july-21`（`createdAt = 2026-07-21`）与
`…-70k-…-from-august-24` 是后开的**同族档位**，价位不与主阶梯重叠——这就是它们不能混进同一
拟合的原由（此前只从 slug 猜出，现在有 `createdAt` 与文本两条依据）。

第四个族别要注意：`will-bitcoin-hit-150k-by-december-31-2026`（`createdAt = 2025-08-07`，价 0.0275）
的文本**没有窗口起点**（`…has a final "High" price … by 11:59PM ET on the date specified in the title`），
即窗口无下界。它碰不上是因为 BTC 历史 ATH（126,208）本就低于 150k。
按 2026-10-15 快照统计，这批 40 只里有 **bitcoin dip-to 9 只、reach 11 只、hit 1 只、ethereum dip-to 4 只、
reach 1 只**，其余 14 只是数值盘（FDV/法案等），不属阈值数字期权。

## 八、三口径对照面板（2026-10-15 固化）

把会话里 heredoc 跑的分析固化成可复现链路：`src/utils/data_sources/deribit.py`（链与远期）、
`src/research/polymarket_ladder.py`（阶梯装载与口径自检）、`src/research/polymarket_digital.py`
（定价与对照）、`src/research/price_bootstrap.py`（路径 bootstrap），
`scripts/polymarket_digital_study.py` 只做取数与落盘。

```bash
uv run python scripts/polymarket_digital_study.py                  # BTC，自动选档位最多那张阶梯
uv run python scripts/polymarket_digital_study.py --list           # 只列阶梯清单
uv run python scripts/polymarket_digital_study.py --offline        # 用最近落盘的链，不联网
uv run python scripts/polymarket_digital_study.py --ladder bitcoin/dip-to/2025-11-24
uv run python scripts/polymarket_digital_study.py --block-hours 168 --vol-target 0.335
```

阶梯清单（2026-10-15 快照，键 = 资产/族别/窗口起点）：

| 阶梯 | 档数 | 说明 |
| --- | --- | --- |
| `bitcoin/reach/2025-11-24` | 10 | 主阶梯，100k~1M，行权价带 |
| `bitcoin/dip-to/2025-11-24` | 7 | 15k~55k |
| `bitcoin/dip-to/creation` | 2 | `from-july-21` 的 60k 与 `from-august-24` 的 70k |
| `bitcoin/reach/creation` | 1 | 90k（2026-02-06 开市） |
| `bitcoin/hit/unbounded` | 1 | 150k，文本无窗口起点 |

**三个口径的读法**：市场价是成交价；Deribit 侧是「同样碰即结算」的风险中性价，但它只能是
**逐档 IV 一阶近似**（单点无法判定该用卡障碍的 σ 还是 ATM 的 σ），所以给一条误差带：
取微笑在「现货到障碍」区间内的 σ 上下沿各算一次，中心取区间中点；只有市场价高过带上沿
（脚本里打 `*`）才算近似误差吃不掉的偏贵。历史 bootstrap 是物理测度参照（去漂移与
含历史漂移两档，市场价常夹在两者之间）。

**读盘（2026-10-15，BTC 指数 80,819、25DEC26 目标远期 81,773、剩余 77 天）**：

| 档位 | 市场¢ | 市场σ | 德σ | 德比带 | 差(中) | 历史0 | 市宽% | 德宽% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100k | 24.50 | 40.6% | 39.0% | 21.1~22.8 | +2.59 | 27.0 | 4.1 | 4.4 |
| 110k | 11.50 | 43.5% | 41.1% | 7.1~9.7 | +3.10 | 11.7 | 8.7 | 6.7 |
| 120k | 6.50 | 47.8% | 43.8% | 2.1~4.5 | +3.20 | 4.5 | 15.4 | 24.4 |
| 130k | 3.40 | 50.1% | 46.8% | 0.5~2.4 | +1.93 | 1.7 | 11.8 | 36.4 |
| 140k | 3.05 | 57.3% | 50.0% | 0.13~1.43 | +2.27 | 0.5 | 3.3 | 28.6 |
| 150k | 2.65 | 63.3% | 51.1% | 0.03~0.70 | +2.28 | 0.2 | 3.8 | 50.0 |
| 200k | 0.95 | 80.2% | 66.5% | 0.00~0.21 | +0.84 | 0.0 | 10.5 | 66.7 |
| 250k | 0.45 | 91.9% | 79.0% | 0.00~0.12 | +0.39 | 0.0 | 22.2 | 无报价 |
| 500k / 1M | 0.25 / 0.15 | 145% / 197% | — | 微笑只到 250k | — | 0.0 | 40 / 67 | — |

（上行阶梯 8/8 档都带 `*`：市场价高过整个误差带，且每一档的市场隐含 σ 都高于微笑 IV。
下行 6 档反着——市场σ 每档都**低于**微笑 IV，比 Deribit 便宜 1.18~5.67¢，但差额落在
很宽的误差带内，而比历史 bootstrap 贵 2.4~6 倍、方向不一致。）

**一个要纠正的算术**：上一轮拿「Deribit 远尾报价的相对宽度 46~86%」去对冲 1.9¢ 的边际，
等于拿**全价差**扣**对 mid 的差额**（重复计数）。正确是「差额 − 市半价差 − 对冲腿半价差」，
130k~150k 段净剩 1.5~2.1¢/张。但两个理由使这仍只能算相对价值：one-touch 的对冲腿是
**一篮子期权**而不是单档（成本是单档的数倍），且两边 payoff 不同源（BTC 跌破再反弹时
一边赔 100、一边归零）。

**迭代中抓到的三个坑**（都已进单测）：

| 坑 | 症状 | 修法 |
| --- | --- | --- |
| 结算文本混用直引号与弯引号 | 弯引号的 6 档抽不出规则，全被归成 `unbounded` | 价格类型正则改用 `["“”]` 字符集 |
| 到期码的日可是一位数 | `6NOV26` 解析失败（`pd.Timestamp.strptime` 对不上） | 改用 `^(\d{1,2})([A-Z]{3})(\d{2})$` 正则 |
| 永续被当成到期合约 | `BTC-PERPETUAL` 进 carry 分母（T 无穷大） | `is_expiry_code()` 拦住，carry 只用到期期货 |
