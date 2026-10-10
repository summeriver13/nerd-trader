# nerd-trader

面向加密市场的量化研究工作台，两套引擎分工：

- **向量化研究回测**（`src/research/`）：按市场规则（手续费/滑点/杠杆）快速验证信号。
- **事件驱动模拟盘**（`src/simulation/`）：事件队列 + 撮合 + 账户回放，贴近实盘执行。

行情统一经 `src/utils/data_loader.py` 调度：从交易所拉取缺失区间，落地为本地 Parquet 缓存。

## 快速开始

依赖管理用 [uv](https://docs.astral.sh/uv/)，要求 Python >= 3.12。

```bash
uv sync                    # 创建 .venv 并安装依赖
uv run python run.py       # 事件驱动模拟盘 demo（离线，自动生成测试数据）
uv run python research.py  # 交互式研究回测
```

研究回测也支持非交互模式（参数省略则回落交互菜单），适合脚本与定时任务：

```bash
uv run python research.py --symbol BTC/USDT --strategy bollinger --bars 365 --timeframe 1d
uv run python research.py --symbol ETH/USDT --strategy rsrs --bars 730 --leverage 2
```

`--strategy` 可填菜单序号 `1`-`5`，也可填别名 `dual-ma` / `trend-reversion` / `bollinger` / `rsrs` / `donchian`。
每次回测都会把资金曲线图写到 `data/results/<SYMBOL>_<策略名>.png`，同时把净值/基准/回撤序列写成同名 `_equity.csv`（pi 扩展里的字符曲线读它），不需要额外确认。

行情来自 OKX（ccxt 公共接口），国内网络需先配代理：

```dotenv
OKX_USE_PROXY=true
OKX_PROXY_PORT=7890
```

## 环境变量

`.env` 由 `src/config/settings.py` 读取（缺失时全部取默认值）。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OKX_API_KEY` / `OKX_SECRET` / `OKX_PASSPHRASE` | 空 | OKX 私有接口凭据，公共行情不需要 |
| `OKX_USE_PROXY` | `false` | 是否走 HTTP 代理 |
| `OKX_PROXY_PORT` | `7890` | 代理端口，地址固定 `http://127.0.0.1:<port>` |
| `OKX_SANDBOX` | `false` | `true` = OKX 模拟盘 |

## 入口脚本

| 命令 | 用途 |
| --- | --- |
| `uv run python research.py [--symbol ... --strategy ...]` | 向量化回测：带参数进入非交互模式，不带参数进入交互菜单，自动落盘图表 |
| `uv run python research.py --robust --bars 2900 --segments 4` | 稳健性检验：全样本 + 一年滚动窗口分布 + 分段稳定性 + 多重检验校正（默认 BTC/ETH） |
| `uv run python research.py --carry --symbol c1,c2,SOL/USDT` | 资金费率 carry 研究：做多现货 + 做空等额永续的毛收益与净收益（Binance 历史，`--carry-days` 控研究窗口） |
| `uv run python research.py --symbol c1 --maker` | 用 maker 挂单执行替代市价单：挂单价与成交明细见下文「执行成本」 |
| `uv run python run.py` | 事件驱动模拟盘 demo（离线，自动生成测试数据）：落 `trades_*.parquet` 与 `<标的>_<策略>_equity.csv`，净值按每根收盘价 mark-to-market，成交按 `src/core/costs.py` 计费（**按是否吃价**：市价单与挂在市场同侧的限价单 taker+滑点，只有被动挂单才 maker） |
| `uv run python scripts/crosscheck_crypto_bars.py --symbol BTC/USDT --bars 100` | 数据对账：同源复抓要求 OHLCV 逐根相等，跨源（Binance）只比差异分布 |
| `uv run python scripts/reconcile_execution.py --market spot [--place-orders --side sell]` | 执行对账：拉回单逐笔比费率，可选在模拟盘下探针单（实盘拒绝） |

回测类参数（`--robust` 与单次回测共用）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--bars` | 单次 730 / 稳健 2900 | 评估区间根数（最近 N 根）；预热根数留在窗口之外，抓数深度按「评估+预热」换算成日历天 |
| `--segments` | 4 | 稳健性检验的分段数 |
| `--trials` | 本次候选数 | 多重检验校正的试验次数 N，历史试过的参数/窗口越多应越大 |
| `--vol-target` | 关闭 | 波动率目标仓位，只给开关则用 40%；把 0/1 信号缩放成 0~100% 暴露 |
| `--vol-lookback` / `--vol-rebalance` | 30 / 7 | 波动率回看根数与调仓间隔（控制换手成本） |
| `--maker` | 关闭 | 用 maker 挂单执行替代市价单（详见「执行成本」） |
| `--maker-patience` | 1 | 挂单耐心根数，超时回退市价（默认挂市价外 1bp，需等 1 根） |
| `--carry-days` | 2900 | `--carry` 的研究窗口（日历天，资金费率历史），与 `--bars` 无关；窗口外数据不参与统计 |

## 稳健性检验

`--robust` 输出四段：

| 段落 | 回答的问题 |
| --- | --- |
| 全样本 + 一年滚动窗口分布 | 持有任意一年的收益/回撤分布如何（窗口最差、正收益窗占比、窗口回撤中位） |
| 分段稳定性 | 收益是否只在某一段成立（正收益段、跑赢基准段、最差段） |
| 多重检验校正 | 在试过 N 个候选后夏普是否还算技能：DSR ≥ 0.95 且 PBO < 50% 才算过关 |
| 资金费率 carry | carry 扣掉保证金占用与换手成本后还剩多少 |

DSR（Deflated Sharpe Ratio）与 PBO（CSCV）实现见 `src/research/overfitting.py`；波动率目标仓位见 `src/research/position_sizing.py`。

## 执行成本

费率只有一个出处：`src/core/costs.py`，现货与永续分开（`SPOT_FEES` / `PERP_FEES`），且
**按「这笔有没有吃价」判定而不是按订单类型**：市价单、以及挂在市场同侧的限价单
（买单 ≥ 卖一）都算吃价，按 taker + 滑点计；只有挂到市价之外、被 `low`/`high` 触及的
挂单才算被动成交，按 maker 计。

`--maker` 把下单从市价单换成挂单，实现在 `src/research/execution.py`。默认挂在市价外
1bp（`offset=1bp`、耐心 1 根）——**挂在市价（`offset=0`）当根就以市价成交，本质是吃价、
必须按 taker 计费**，拿 maker 费率是错误口径。

挂单价差与触价率来自 1m 数据实测（BTCUSDT 12.9 万根）：

| 挂单价差 | 触价率 | 净优势 | 说明 |
| --- | --- | --- | --- |
| 0bp（挂在市价） | 100% | 0（按 taker） | 当根即成，拿不到 maker 费率 |
| **1bp（默认）** | 86.6% | **约 +6.1bp/边** | 已扣掉挂不上时回退市价的成本 |
| 5bp | 61.9% | +4.6bp/边 | 挂不上比例更高 |

**maker 与市价不是「同样交易只换费率」**：挂单要等一根才生效，仓位生效时点不同，所以
结果永远打印 maker − market 差额。实测布林带均值回归 `--bars 200`（4 笔交易）：maker
5.13% 对市价 5.24%，即 −0.10%——费率上限只值 +0.36%（4 笔 × 0.09%/边），延迟一根入场
就把它吃掉了。另一面是择时偏差：跨 139 标的 × 4 策略，挂 1bp 的仓位偏离标准差 25–37pp
（突破型 RSRS 净效应 −13.82pp），挂在市价则偏离为 0、代价是全额 taker。急涨（前 15 根
> +50bp）时挂单会被吃穿，此时买单按吃价计。

### 对账：模型 vs 交易所真实回单

`scripts/reconcile_execution.py` 在 OKX 模拟盘下三笔最小单，把回单费率与模型逐笔比对
（判据：单边误差 < 1bp）：

| 探针单 | OKX 标记 | 实际 | 模型 | 差 |
| --- | --- | --- | --- | --- |
| 市价卖 | taker | 10.000bp | 10.000bp | +0.000 |
| 限价卖@买一（挂在市场同侧） | taker | 10.000bp | 10.000bp | +0.000 |
| 限价卖@卖一外 1bp（被动） | maker | 8.000bp | 8.000bp | −0.000 |

滑点：市价单实测 +0.01bp（$20 名义额）而模型常数是 5bp，**对小单偏保守**；真实滑点与
规模相关，那个常数应读作「按打算下的规模」。`--place-orders` 只在 `OKX_SANDBOX=true`
时允许，实盘直接拒绝。

## 绩效指标

回测摘要与图表标题统一输出下列指标（均由 `src/research/engine.py` 计算）：

| 指标 | 定义 |
| --- | --- |
| 总收益 / 年化 | 资金曲线首尾变化；年化按 365 天复利折算 |
| 基准 | 同区间买入持有收益 |
| 卡尔玛 (Calmar) | 年化收益 / \|最大回撤\|，衡量单位回撤换来的收益 |
| 夏普 (Sharpe) | 净收益均值 / 标准差 × √年化根数 |
| 索提诺 (Sortino) | 净收益均值 / 下行偏差（MAR=0 口径）× √年化根数 |
| 最大回撤 | 资金曲线相对历史峰值的最大跌幅 |
| 交易回合 / 胜率 / 盈亏比 | 按连续非零持仓段聚合，一进一出算一个回合 |
| 平均持仓 / 费用 | 回合平均持有 K 线根数；含手续费与滑点的总成本 |

## 数据与配置

- 缓存路径：`data/storage/crypto/<exchange_id>/<market>/<timeframe>/<SYMBOL>.parquet`（`market` = `spot` | `swap`），`data/` 不入版本库。
- 研究/回测产物：数值表格一律落 parquet（`compression="snappy"`），不写 CSV；唯一例外是 `data/results/<策略>_equity.csv`，它是 pi 扩展字符曲线的数据源（零依赖 TS 解析）。
- 复现指纹：每次成功运行写 `data/results/provenance/<run_id>.json`，`repro_hash` = sha256(配置快照 + 数据指纹 + git HEAD/脏标记)，同一命令必得同一 hash（`src/research/provenance.py`）。
- 资金费率缓存：`data/storage/crypto/binance/funding/8h/<SYMBOL>.parquet`（OKX 只保留约 100 天，长历史取 Binance，回溯至 2019-09）。
- 数据源：ccxt，交易所实例由调用方注入（行情为 OKX，资金费率为 Binance 公共接口）。

## 目录结构

```text
data/              数据湖（不入版本库）：storage/ 行情与资金费率缓存、results/ 回测产物与复现指纹
scripts/           数据与执行对账：批量回填（fetch_crypto_bars）、验收闸（audit_crypto_bars）、manifest 生成、行情交叉校验、成本/滑点对账、挂单成交率研究（maker_edge_study）、信号扫描（signal_scan）、趋势对比（trend_duel）
src/analytics/     绩效指标（单一出处）
src/cli/           研究回测的命令行与交互菜单
src/config/        环境配置读取、路径常量、实盘参数表
src/core/          事件引擎、数据类型、BarGenerator/ArrayManager、成本模型
src/research/      向量化回测：engine、domains（市场规则）、strategies、robustness、overfitting、position_sizing、carry、execution（maker 挂单执行）、provenance（复现指纹）
src/simulation/    事件驱动：engine、mock_gateway、replay_loader
src/strategies/    事件驱动策略（dual_ma）
src/utils/         数据调度与数据源
run.py / research.py  两个薄壳入口（逻辑在 src/cli/ 与 src/simulation/）
docs/              因子/模型/书目/交易者笔记/执行与信号研究/趋势研究
```

## 依赖与源

- 依赖锁定在 `uv.lock`。
- 默认 PyPI 源配成清华镜像（`pyproject.toml` 的 `[[tool.uv.index]]`）；换回官方源改这段后重新 `uv lock`。

## 文档索引

| 目录 | 内容 |
| --- | --- |
| `docs/factors/` | 因子笔记：动量、价值、波动率、风险、质量 |
| `docs/models/` | 模型笔记 |
| `docs/library/` | 书籍与视频剪藏 |
| `docs/wisdom/` | 交易者思想摘录 |
| `docs/src.md/` | 策略源码说明 |
| `docs/execution/` | 执行与信号研究：被动挂单成交率与逆向选择、1m 方向性信号扫描 |
| `docs/trend/` | 趋势追踪研究：经典唐奇安与自适应双均线的对比 |
