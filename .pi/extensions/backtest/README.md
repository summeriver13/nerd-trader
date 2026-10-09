# backtest（pi 扩展）

把 `research.py` 接进 pi：结构化参数跑回测、实时进度、结果卡片、运行历史与后台任务。

```text
backtest 工具    跑回测：单次 / 稳健性 / carry，前台流式进度，后台立即返回
/runs            历史记录列表 → 运行明细（含净值曲线）→ 两次运行对比
/jobs            后台任务列表 → 日志尾部 → 终止
```

## 为什么

命令行跑稳健性检验要十几分钟，进度只能靠 `tail` 日志；跑完的结论散在终端里，
想对比两次运行得手动翻记录。这个扩展把「发起 → 进度 → 结果 → 历史 → 对比」放进 pi 内。

## 用法

```text
# 单次回测（前台，约 3 秒，命中缓存时不发网络请求）
backtest 工具: { mode: "backtest", symbol: "c1", strategy: "bollinger", days: 365 }

# 稳健性检验（后台，约 5-10 分钟）
backtest 工具: { mode: "robust", symbol: "c1,c2", days: 2900, segments: 4, background: true }

# 资金费率 carry（含逐年衰减）
backtest 工具: { mode: "carry", symbol: "c1,c2" }
```

参数（省略即用 research.py 的默认值）：

| 参数 | 说明 |
| --- | --- |
| `mode` | `backtest`（默认）/ `robust` / `carry` |
| `symbol` | 标的，如 `BTC/USDT` 或预设键 `c1`/`c2`；稳健性与 carry 支持逗号分隔多个 |
| `strategy` | 仅单次回测：序号 1-4 或别名 `dual-ma`/`trend-reversion`/`bollinger`/`rsrs` |
| `days` | 评估窗口根数（映射到 `--bars`）：单次默认 730、稳健性默认 2900；`carry` 模式下映射到 `--carry-days`，按日历天解释 |
| `segments` / `trials` | 稳健性分段数（默认 4）与多重检验声明的试验次数 N |
| `volTarget` | 波动率目标仓位：`0.4`=40%，`true`=默认 40%，`false`=关闭 |
| `volLookback` / `volRebalance` | 波动率回看根数（默认 30）与调仓间隔（默认 7） |
| `background` | 后台运行，完成时通知 |

快捷键：

| 面板 | 按键 |
| --- | --- |
| `/runs` | `↑↓` 选择 · `Enter` 明细 · `c` 标记并对比 · `e` 展开/收起 · `Esc` 返回或关闭 |
| `/jobs` | `↑↓` 选择 · `Enter` 看日志尾部 · `x` 终止任务 · `r` 刷新 · `Esc` 返回或关闭 |

## 净值曲线

单次回测的结果卡片与 `/runs` 明细里会画净值曲线（策略 vs 买入持有 + 回撤带）：

```text
净值 ×0.92   基准 ×1.36   最深回撤 -30.0%
 ×2.07┤                              ⢀⣠⣄ ⣀    ⢰⢦
      │                              ⣸ ⠈⠛⠉⢳⣀⡖⠲⠼⠘⡆⣀
 ×0.92┤⠉⠉⠉⠉⠁                             ⠘⠒⠒⠒⠒⠒⠒⠒⠒⠒⠚⠳⠖⠦⢶
      └─────────────────────────────────
       2024-10-10             2026-10-09
回撤   ▂  ▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂  ▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▄▄▄▄▄▄▄▄▄▄▄▄▄▅▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▆▇▇██  最深 -30.0%
```

用 braille 字符（U+2800 区）画，任何终端都能显示，并按面板宽度自适应（宽于 28 列才画）。
数据来自 `research.py` 落盘的净值 CSV（`chart.ts` 按 mtime 缓存）。

## 产物

```text
data/results/runs.json          已归档的运行记录（/runs 的数据源，最多 300 条）
data/results/jobs/<id>.json     任务状态（由 runner.mjs 写）
data/results/logs/<id>.log      research.py 的完整标准输出
data/results/*.png              单次回测的图表（research.py 自己生成）
data/results/*_equity.csv       单次回测的净值/基准/回撤序列（字符曲线的数据源）
```

运行 id 形如 `20261009-105113-backtest`，按时间排序。

## 实现

| 文件 | 职责 |
| --- | --- |
| `index.ts` | 装配：工具、命令、会话事件、后台任务轮询 |
| `spec.ts` | 数据契约与路径约定（`data/results/` 下的一切） |
| `parse.ts` | 解析 research.py 输出（表格从右往左切列，避免中文名被切开） |
| `proc.ts` | 任务启动、进程树终止、日志与运行记录的读写 |
| `runner.mjs` | 子进程执行器：跑 `uv run python research.py`，落日志与退出状态 |
| `ui.ts` | 结果卡片、进度行、表格与图表组件 |
| `chart.ts` | 净值曲线：读 equity CSV、按宽度画 braille 字符图 |
| `panels.ts` | `/runs` 与 `/jobs` 的状态机面板 |

执行链路：工具 → `runner.mjs`（Node 子进程）→ `uv run python research.py`。
扩展只轮询日志文件与状态文件，不持有管道；因此后台任务在 pi 退出后仍会继续，
重开 pi 也能通过 spec 里的 pid 判断任务是否还活着。

## 边界

- 只在 nerd-trader 目录内生效：从 `cwd` 向上找 `research.py`，找不到就提示。
- 后台任务靠 5 秒轮询归档，pi 退出期间结束的任务会在下次打开 pi 时补归档并通知。
- 被强杀（`taskkill /T /F`）的任务来不及写状态文件，扩展报「进程已消失，未归档」而不是伪造结果。
- 网络失败（代理不可达等）会如实记录为 `failed` 与退出码，不重试。
- 净值曲线用字符画，任何终端可用；内联 PNG 只在支持 kitty / iTerm2 协议的终端显示（Windows Terminal 不支持，`Image` 会退化成一行路径文字，故不再显示）。
- 改完扩展代码需在 pi 里执行 `/reload` 才生效。
- 项目级扩展首次加载需要 project trust 确认。

关闭方式：环境变量 `NERD_TRADER_BACKTEST_DISABLE=1`，或删除本目录。
