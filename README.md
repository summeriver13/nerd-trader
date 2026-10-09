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
uv run python research.py --symbol BTC/USDT --strategy bollinger --days 365 --timeframe 1d
uv run python research.py --symbol ETH/USDT --strategy rsrs --days 730 --leverage 2
```

`--strategy` 可填菜单序号 `1`-`4`，也可填别名 `dual-ma` / `trend-reversion` / `bollinger` / `rsrs`。
每次回测都会把资金曲线图写到 `data/results/<SYMBOL>_<策略名>.png`，不需要额外确认。

`research.py` 按提示选择标的、周期与策略（双均线 / 趋势+反转 / 布林带均值回归 / RSRS），
结果直接打印收益率、最大回撤与夏普，可选出图存到 `data/results/`。

行情来自 OKX（ccxt 公共接口），国内网络需先配代理：

```dotenv
OKX_USE_PROXY=true
OKX_PROXY_PORT=7890
```

## 环境变量

`.env` 由 `config/settings.py` 读取（缺失时全部取默认值）。

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
| `uv run python research.py --robust --days 2900 --segments 4` | 稳健性检验：全样本 + 一年滚动窗口分布 + 分段稳定性 + 多重检验校正（默认 BTC/ETH） |
| `uv run python research.py --carry --symbol c1,c2,SOL/USDT` | 资金费率 carry 研究：做多现货 + 做空等额永续的毛收益与净收益（Binance 历史） |
| `uv run python run.py` | 事件驱动模拟盘 demo（离线，自动生成测试数据） |
| `uv run python analyze_result.py --capital 1000000` | 读 `data/results/trades_*.csv` 估算资金曲线与回撤，`--capital` 需与引擎初始资金一致 |

回测类参数（`--robust` 与单次回测共用）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--days` | 单次 730 / 稳健 2900 | 回测窗口长度：评估区间取最近 N 根，指标预热在窗口之外 |
| `--segments` | 4 | 稳健性检验的分段数 |
| `--trials` | 本次候选数 | 多重检验校正的试验次数 N，历史试过的参数/窗口越多应越大 |
| `--vol-target` | 关闭 | 波动率目标仓位，只给开关则用 40%；把 0/1 信号缩放成 0~100% 暴露 |
| `--vol-lookback` / `--vol-rebalance` | 30 / 7 | 波动率回看根数与调仓间隔（控制换手成本） |

## 稳健性检验

`--robust` 输出四段：

| 段落 | 回答的问题 |
| --- | --- |
| 全样本 + 一年滚动窗口分布 | 持有任意一年的收益/回撤分布如何（窗口最差、正收益窗占比、窗口回撤中位） |
| 分段稳定性 | 收益是否只在某一段成立（正收益段、跑赢基准段、最差段） |
| 多重检验校正 | 在试过 N 个候选后夏普是否还算技能：DSR ≥ 0.95 且 PBO < 50% 才算过关 |
| 资金费率 carry | carry 扣掉保证金占用与换手成本后还剩多少 |

DSR（Deflated Sharpe Ratio）与 PBO（CSCV）实现见 `src/research/overfitting.py`；波动率目标仓位见 `src/research/position_sizing.py`。

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

- 缓存路径：`data/storage/crypto/<exchange_id>/<timeframe>/<SYMBOL>.parquet`，`data/` 不入版本库。
- 资金费率缓存：`data/storage/crypto/binance/8h/<SYMBOL>.parquet`（OKX 只保留约 100 天，长历史取 Binance，回溯至 2019-09）。
- 数据源：ccxt，交易所实例由调用方注入（行情为 OKX，资金费率为 Binance 公共接口）。

## 目录结构

```text
config/            环境配置读取、实盘参数表
research.py        交互式研究回测入口
run.py             事件驱动模拟盘 demo
analyze_result.py  成交记录 → 资金曲线/回撤
src/core/          事件引擎、数据类型、BarGenerator/ArrayManager
src/research/      向量化回测：engine、domains（市场规则）、strategies、robustness、overfitting、position_sizing、carry
src/simulation/    事件驱动：engine、mock_gateway、replay_loader
src/strategies/    事件驱动策略（dual_ma）
src/utils/         数据调度与数据源
docs/              因子/模型/书目/交易者笔记
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
