# Deribit 期权链（自持数据集）

## 为什么自持

Polymarket 的 BTC/ETH 阈值市场是**路径依赖的 one-touch**（结算源是 Binance 1 分钟 High/Low，
正文含 `immediately resolve`，见 [`../polymarket/README.md`](../polymarket/README.md) §七），
所以它的定价锚不能是历史波动率，必须是**期权微笑**——而 BTC/ETH 期权的深流动性公开场所只有 Deribit：

- 给 Polymarket 阶梯做「市场价 vs 风险中性价」对照（已做到 0.4~3.2¢ 的量化，见 polymarket 文档 §八）；
- 将来真要对冲远尾时所需要报价与深度也在这条链上（永续只对冲 delta，补不上障碍附近的凸性）。

crypto 数据按容器 ADR-0006 由本项目自持，故落在本项目 `data/storage/deribit/`，
不进容器数据湖，也不新增外部数据源依赖。

## 数据集契约

```
data/storage/deribit/raw/options/<YYYY-MM-DD>.jsonl
data/storage/deribit/raw/futures/<YYYY-MM-DD>.jsonl
data/manifests/deribit_raw_{options,futures}.json
```

| 项 | 值 |
| --- | --- |
| 业务键 | `(instrument, fetched_at)` —— 一天可多轮快照、各留一行；同快照重跑写 0 行 |
| 期权字段 | `instrument, expiry, kind, strike, mark_price, mark_iv, bid, ask, open_interest, underlying` + `currency, index_price, fetched_at` |
| 期货字段 | `instrument, expiry（永续为 null）, mark_price` + `currency, index_price, fetched_at` |
| 单位 | `mark_iv` 是**小数**（0.374 不是 37.4）；`mark_price` 是**币本位权利金**（BTC 计价，若换成美元乘 `index_price`）；期货 `mark_price` 是 USD |
| 行内自带 | `index_price`（定价锚点 S0）与 `fetched_at`（快照时刻），下游无需再取 |
| 来源 | `www.deribit.com`（`public/get_index_price` + 两个 `public/get_book_summary_by_currency`） |

期货只存 `mark_price`：carry 锚只用 mark 外推远期，bid/ask/oi 现有链路一个也不消费，
将来真要用后续快照自然会带上（不猜未消费的字段）。

## 采集与验收

```bash
uv run python scripts/fetch_deribit.py                  # BTC + ETH 各一轮快照
uv run python scripts/fetch_deribit.py --list-only      # 只看链的规模，不落盘
uv run python scripts/audit_deribit.py                  # 验收：致命项退出码 1
uv run python scripts/audit_deribit.py --stale-hours 30
```

建议频度**日频**：微笑的日内变化远小于其它误差源（S0 用 Deribit 永续 mark、逐档 IV 是一阶近似、
两边 payoff 不同源），多轮快照只在做对冲报价研究时才有意义。

首轮实测（2026-10-15）：BTC 930 档 + ETH 812 档期权、各 13 张期货，约 800 KB/天。

## 验收口径

致命项（退出码 1）：JSONL 解析失败、业务键重复、`mark_price ≤ 0`、`strike ≤ 0`、
`mark_iv ∉ (0, 5]`、买卖价倒挂、manifest 文件数/行数与磁盘不符。

告警项：快照距今超过 `--stale-hours`、单文件期权档数少于 `--min-options`。

## 已经踩到的坑

| 坑 | 症状 | 处置 |
| --- | --- | --- |
| 近到期档上游回 `mark_price = 0` | 验收闸每次都报 6 条致命项 | **采集侧**丢掉零价/零 IV 档（不是脏数据，但对定价无用）。与容器里 QMT「成交额正常但 OHLC 全 0」是同一条教训：过滤放采集侧，别入库后再清 |
| 永续不是到期合约 | `BTC-PERPETUAL` 进 carry 分母会让期限发散 | `expiry` 记 `null`，`forward_curve()` 只看到期期货 |
| 到期码的日可一位数 | `6NOV26` 用 `%d%b%y` 解析失败 | 正则 `^(\d{1,2})([A-Z]{3})(\d{2})$` |
| 管道会吃掉退出码 | `... | tail -6; echo $?` 拿到的是 `tail` 的状态，曾据此误报「验收通过」 | 验收脚本独立跑或看 `PIPESTATUS`，不拿管道尾部的状态当结论 |
