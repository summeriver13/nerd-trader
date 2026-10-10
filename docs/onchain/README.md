# 链上聪明钱：数据源可行性侦察

「聪明钱量化」在 crypto 里通常指两件事：跟踪高胜率钱包的链上动作，或跟踪大户/机构
的持仓方向。本页不实现任何策略，只回答一个前置问题——**这些数据到底拿不拿得到、
历史多深、成本多少**。结论决定了「聪明钱」这条研究方向值不值得投入。

侦察方式为端点直探（2026-10-11，经本机代理），不落任何依赖、不写采集代码。

## 结论速览

- **交易所持仓类数据做不了研究**：Binance 期货数据硬限制 30 天（`startTime` 传 60 天前
  直接报 `-1130 invalid`），OKX Rubik 日线 180 天。180 个样本连噪音都分不出来。
- **链上数据免费拿得到**：BTC 走 mempool.space，ETH 主网走 Blockscout 与 Routescan，
  Solana 走公共 RPC，均无需 key、无需注册。
- **但标签层拿不到**：Etherscan V2、Blockchair、Covalent、Dune、Glassnode、CryptoQuant、
  Helius、Solscan 全部需要 key。**没有标签层就没有「谁是聪明钱」的答案。**
- **链覆盖有缺口**：Base / BSC / Arbitrum / Polygon 无免费源。
- **吞吐瓶颈是代理不是源站**：串行约 1.2 req/s，并发 10 直接超时。
- **可行路径只有一条**：不试图「发现」聪明钱，改为跟踪**公开已知**的地址
  （交易所热钱包、项目方地址、做市商地址）——无选择偏差，且已实测可用。

## 一、交易所持仓类数据：历史太短，做不了研究

这是最标准的「聪明钱」口径——大户持仓量多空比、主动买卖量比、持仓量历史。实测深度：

| 源 | 端点 | 实测深度 |
| --- | --- | --- |
| OKX Rubik 多空持仓人数比 | `/api/v5/rubik/stat/contracts/long-short-account-ratio` | 5m→**2 天**、1H→**30 天**、1D→**180 天** |
| OKX Rubik 持仓量与成交额 | `/api/v5/rubik/stat/contracts/open-interest-volume` | 5m→2 天 |
| OKX Rubik 主动买卖量 | `/api/v5/rubik/stat/taker-volume` | 5m→2 天 |
| OKX Rubik 借币比例 | `/api/v5/rubik/stat/margin/loan-ratio` | 5m→2 天 |
| Binance 大户持仓量多空比 | `/futures/data/topLongShortPositionRatio` | **30 天** |
| Binance 大户账户数多空比 | `/futures/data/topLongShortAccountRatio` | **30 天** |
| Binance 主动买卖量比 | `/futures/data/takerlongshortRatio` | **30 天** |
| Binance 持仓量历史 | `/futures/data/openInterestHist` | **30 天** |

Binance 的 30 天不是 `limit` 限制，是**硬限制**：`limit=500` 也只返回 30 条，且
`startTime` 传 60 天前直接报 `parameter 'startTime' is invalid`（code `-1130`）。

**为什么这判了死刑**：对照本仓库既有的策略研究——3195 根日线、6 个策略，DSR 多重检验
校正后**全部 FAIL**（最好 0.875，门槛 0.95）。180 个样本连噪音都分不出来。这类数据
只能做**实时监控**，不能做历史研究。

## 二、免费无 key 可用的链上源

| 源 | 覆盖 | 能力 | 实测 |
| --- | --- | --- | --- |
| **mempool.space** | BTC | 地址详情 + 完整交易历史（分页） | ✅ 200 |
| **Blockscout** | ETH 主网 / Gnosis / Optimism | 地址交易 + 代币转账，游标分页 | ✅ 200 |
| **Routescan** | **仅 ETH 主网** | Etherscan 兼容 API（`module=account&action=txlist`） | ✅ 200 |
| **Solana 公共 RPC** | Solana | `getSignaturesForAddress` + `getTransaction` | ✅ 200 |

Blockscout 实例覆盖实测：`eth.blockscout.com` ✅、`gnosis.blockscout.com` ✅、
`optimism.blockscout.com` ✅、`base.blockscout.com` ❌（Cloudflare 403 挑战页）。

Routescan 链覆盖实测：chainid=1（Ethereum）✅；chainid=8453（Base）、56（BSC）、
42161（Arbitrum）、137（Polygon）一律返回 `{"status":"0","message":"chain not supported"}`。

Solana 公共 RPC（`https://api.mainnet-beta.solana.com`）实测可用方法：`getHealth`、
`getSlot`、`getSignaturesForAddress`（返回签名 + `blockTime` + `slot`）、
`getTransaction`（`encoding=jsonParsed` 返回解析后的 `meta`/`transaction`/`blockTime`）。

## 三、需要 key 的（全部不可用）

以下端点直探均返回 401 / 无 key 拒绝，无一例外：

| 平台 | 实测响应 |
| --- | --- |
| Etherscan V2 | `{"status":"0","message":"NOTOK","result":"Missing/Invalid API Key"}` |
| Blockchair | 430 `Your IP address is temporary blacklisted`（免费额度不足以支撑研究） |
| Covalent / GoldRush | 401 |
| Dune | 401 `invalid API Key` |
| Glassnode | 401 |
| CryptoQuant | 401 `Token does not exists` |
| Helius | 401 `Unauthorized` |
| Solscan | 404 |
| Solana FM | 502 |

公共 RPC 侧：`ethereum.publicnode.com` ✅、`eth.drpc.org` ✅；`eth.llamarpc.com` 525（SSL
握手失败）、`cloudflare-eth.com` 返回 `-32046 Cannot fulfill request`、`rpc.ankr.com` 要求认证。

## 四、吞吐实测：瓶颈是代理，不是源站

| 项 | 实测 |
| --- | --- |
| Blockscout 串行延迟中位 | 0.81 s → 约 **1.2 req/s** |
| Solana RPC 串行延迟中位 | 0.95 s → 约 **1.1 req/s** |
| Blockscout 游标分页 | 50 条/页，6 页 300 条耗时 14.9 s → 约 **2.5 s/页** |
| 并发 10 | **直接 ReadTimeout**，代理扛不住 |

延迟来自本机 Clash 代理（约 0.8 s 固定开销），不是源站限流——因此**并发这条路被堵死**，
只能串行慢跑。这是本机网络条件，换环境可改善。

## 五、决定性缺口：没有标签层

「聪明钱量化」必须先回答**「谁是聪明钱」**。免费层的现状是：

- ✅ 给原始交易（from / to / value / timestamp / token）
- ❌ **不给地址标签**（谁是谁、谁在赚钱）

标签层（Nansen、Arkham、Glassnode 的实体标签）全部付费。这意味着只能自己从原始数据里
筛「谁在赚钱」——

> **而这正是幸存者偏差最容易咬人的地方：跟「从 29 只里挑 LIT」是同一个错误，
> 只是候选池从 29 变成几百万个地址。**

三个具体陷阱：

1. **幸存者偏差**：事后看到的「高胜率钱包」是从海量地址里筛出来的，筛选本身就会挑出运气好的人。
2. **反身性**：一旦某地址被广泛标记为聪明钱，跟踪者本身成为流动性，**信号自我毁灭**。
3. **前视偏差**：链上时间戳是交易时间，但你看到它、确认它、跟随成交全在之后，价格可能已经动完。

## 六、可行路径：跟踪公开已知的地址

绕开陷阱的办法是**用「公开已知」代替「事后筛选」**：

| | 做法 | 问题 |
| --- | --- | --- |
| ❌ | 按历史胜率筛地址 | 幸存者偏差 + 反身性 |
| ✅ | **用公开已知的地址** | **无选择偏差** |

公开已知的地址类别：交易所热钱包、项目方/基金会地址、做市商地址、名人/机构公开地址。

**交易所净流入**是这条路上最成熟的版本，已实测可用：

```text
Binance 14   0x28C6c06298d514Db089934071355E5743bf21d60   余额 48,171 ETH
Binance 16   0xDFd5293D8e347dFe59E90eFd55b2956a1343963d   余额 21,064 ETH
Coinbase 10  0x503828976D22510aad0201ac7EC88293211D23Da   余额      9 ETH
Kraken       0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2   余额      0 ETH
```

代币转账端点 `/api/v2/addresses/{addr}/token-transfers` 返回字段完整：
`from`、`to`、`total.value`、`token.symbol`、`timestamp`、`method`、`transaction_hash`。
样例：30,000,000 USDT @ 2026-10-10T17:38:11Z。

**为什么这条最干净**：地址公开已知（无选择偏差）、不依赖付费标签层、经济含义明确
（大额转入交易所 = 准备卖出，转出 = 囤币），且可回测。

## 七、成本估算（按实测吞吐）

| 研究规模 | 请求数 | 预计耗时 |
| --- | --- | --- |
| 交易所净流入：200 地址 × 500 笔 | 2,000 页 | 约 **1.4 小时** |
| Solana 钱包跟踪：50 钱包 × 200 笔 | 10,000 次（每笔一次请求） | 约 **2.6 小时** |

两者都在可接受范围内，前提是接受串行慢跑。Solana 的成本远高于 EVM——因为
`getSignaturesForAddress` 只给签名，每笔交易的详情要单独调一次 `getTransaction`。

## 八、口径与限制

- 本页所有深度与吞吐为 2026-10-11 单次实测，未做多次采样；交易所侧深度限制可能随
  政策变化，链上免费源可能随时收紧或加 key。
- 吞吐数字含本机 Clash 代理开销（约 0.8 s/请求），**不代表源站上限**。
- 未实测：免费源的实际限流阈值（本轮未触发任何 429）、数据完整性与缺块率、
  地址标签的第三方免费替代品（如 ENS 反向解析，Blockscout 响应里含 `ens_domain_name` 字段但多为空）。
- 「交易所净流入」路径的经济含义是**假设**（转入 = 准备卖出），本页只验证了数据可得性，
  未验证其预测力。任何研究都必须先过 DSR 多重检验校正。
- 本页不涉及任何新依赖、采集代码或数据落地。

## 九、端点清单（复现用）

```text
# 链上免费源
https://mempool.space/api/address/{btc_addr}/txs
https://eth.blockscout.com/api/v2/addresses/{addr}/transactions
https://eth.blockscout.com/api/v2/addresses/{addr}/token-transfers
https://api.routescan.io/v2/network/mainnet/evm/1/etherscan/api?module=account&action=txlist&address={addr}
https://api.mainnet-beta.solana.com          # JSON-RPC: getSignaturesForAddress / getTransaction

# 交易所持仓类（深度不足，仅记录）
https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy=BTC&period=1D
https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=1d&limit=500
```
