# nerd-trader

**English** · [中文](README.zh-CN.md)

A quantitative research workbench for crypto markets, split across two engines:

- **Vectorized research backtest** (`src/research/`): validate signals fast under market rules (fees / slippage / leverage).
- **Event-driven paper trading** (`src/simulation/`): event queue + matching + account replay, close to live execution.

Market data is dispatched through `src/utils/data_loader.py`: missing ranges are fetched from the exchange and cached locally as Parquet.

> CLI output and the research notes under `docs/` are in Chinese; column names, metric names and the Python API are ASCII. This file is the English entry point.

## Contents

- [Research findings](#research-findings) — one line per research line, read this first
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [Entry points](#entry-points)
- [Robustness checks](#robustness-checks)
- [Execution costs](#execution-costs)
- [Performance metrics](#performance-metrics)
- [Data & configuration](#data--configuration)
- [Repository layout](#repository-layout)
- [Dependencies](#dependencies)
- [Agent tooling](#agent-tooling)
- [Docs index](#docs-index)
- [License](#license)

## Research findings

Thirteen lines, one verdict each; every number is reproducible from the linked note (in Chinese) along with the exact command and data window. For a code map — module, CLI entry, criteria page and output directory per research line — see [`docs/研究线索引.md`](docs/研究线索引.md).

The thread running through the whole repo is **real signal, no monetization**: on daily price/volume cross-sections nothing clears both the cost wall and the beta wall, and the only reading that beats BTC on a risk-adjusted basis is time-series momentum.

| Research line | One-line verdict | Note |
| --- | --- | --- |
| **Cross-sectional factors** | `vol_30` is a real signal (IC t = 14.27, six falsification tests passed), yet all six factors **lose to buy-and-hold BTC** — the best implementation is to hold BTC instead of altcoins. Alpha101's `decay_linear` operator cuts long-only turnover cost from 15–22pp/year to a third with the signal unchanged, but stable variants still lag BTC by 1.3–9pp | [`docs/factors/README.md`](docs/factors/README.md) |
| **Time-series momentum** | Simple wins: ~60 lines of Donchian beat 292 lines of adaptive dual-MA. Multi-asset TSMOM reaches Sharpe 1.11 with a −17.1% drawdown, the only reading that beats BTC risk-adjusted (at the cost of ~10% average exposure; leveraged capacity $1M–$10M; the paper-trading forward ledger opened 2026-10-15, runs on a daily scheduled job, retirement criteria pre-registered). At portfolio level it is positioned as an **overlay on 20–30% of equity** (annualized 23.17% → 24.56%, drawdown −77.4% → −63.3%), insensitive to execution (3× costs lose only 1pp) — but DSR passes only at the loosest bucket (@6 = 0.977), PBO passes one of two pools, and the paired difference vs BTC includes 0: a candidate, not a validated strategy | [`docs/trend/README.md`](docs/trend/README.md) |
| **Portfolio** | The single-strategy champion changes with the asset (trend+reversal on BTC, Donchian on ETH); measured correlation tops out at 0.53, so the four trend strategies are not "four ways to write the same trade" | [`docs/portfolio/README.md`](docs/portfolio/README.md) |
| **Funding-rate carry** | Cross-source substitution only holds for large caps; the equal-weight portfolio decays to −2.08% over the last 365 days (Sharpe −7.86), and only 1 of 29 survives multiple-testing correction | [`docs/carry/README.md`](docs/carry/README.md) |
| **Volatility targeting** | A volatility-reduction device, not a return-improvement device: cutting volatility is reliable, raising Sharpe is not significant, and both assets fail in sync from 2024-08 | [`docs/vol_target/README.md`](docs/vol_target/README.md) |
| **New-listing effect** | The post-listing decline (206 coins, 1-year median −72.13%) is the cleanest effect in the repo, but it is untradeable — and now quantified: only 50.49% of coins have a perp on day 1, and after applying the channel mask all 9 short portfolios get **worse** — the half you could profit from shorting is exactly the half you cannot bet on | [`docs/listing/README.md`](docs/listing/README.md) |
| **Meme coins** | Lottery distribution (median −69.7%, 8/12 negative) and leveraged BTC (beta 1.62, R² 0.27); they tend to reverse after a spike and do not bounce after a crash | [`docs/meme/README.md`](docs/meme/README.md) |
| **Cointegrated pairs** | The first structure with beta ≈ 0 (0.012, correlation 0.063), but the edge is near zero: main specification annualized −0.4%, best variant DSR 0.261, fails | [`docs/pairs/README.md`](docs/pairs/README.md) |
| **Passive limit execution** | Pure passive resting orders are a negative-expectancy business: posting deeper only lowers fill frequency without improving P&L (post-fill drift is a constant −1.1 to −1.4bp) | [`docs/execution/README.md`](docs/execution/README.md) |
| **On-chain smart money** | Data-source reconnaissance: on-chain data is freely available, the **label layer is not**; the only viable path left is tracking publicly known addresses | [`docs/onchain/README.md`](docs/onchain/README.md) |
| **Prediction markets** | Data-source reconnaissance: **history is fully sealed** (settled markets keep no price series, the subgraph is gone, `/trades` has no time-window pagination), so only forward collection works. Settlement rules verified (Binance 1-minute High/Low, touch = settle). Three-spec panel: the upside ladder is 0.4–3.2¢ more expensive than Deribit on all 8/8 buckets and outside the error band, while the downside sits between Deribit and history | [`docs/polymarket/README.md`](docs/polymarket/README.md) |
| **Tokenized stocks** | Specification reconnaissance: "tokenized SanDisk" is **two products** — an on-chain token spot `XSNDK` (pays dividends reinvested into fractions via a multiplier, but the top 20 ask levels total only $10K) and a USDT perp `SNDK-USDT-SWAP` (no dividend, 75x maps to just 175 USDT notional, 24/7 but weekend volume is 2.3%, and the index does not freeze off-hours so there is no external anchor). Liquidity is entirely in the perp (140 spot pairs $29.0M vs 198 perps $994.3M, 34×), and the missing dividend is priced back through funding (SNDK +6.40% annualized paid by longs, KO −5.26% received by longs) | [`docs/tokenized-stocks/README.md`](docs/tokenized-stocks/README.md) |
| **Copy-trading leaderboard** | Data-line reconnaissance: the leaderboard "return" is cumulative on the trader's **own invested capital** (`pnlRatio ≈ pnl / investAmt`), while the copier's P&L lives in another field — of 59 traders with AUM ≥ $10k the self-reported median is +19.2%, but the copier-basis median is only +6.0% (aggregate +6.5%). Among 246 traders, 113 have no copiers, 26% are cumulatively negative, the top 10 positive-P&L names take 47%, and the median cumulative copier count among those with ≥100 cumulative copiers is only 4.5%; official profit share caps at 30% (onboarding threshold 100k USDT). The leaderboard is a survivor snapshot, archived daily since 2026-10-17 for forward validation (criteria pre-registered) | [`docs/copy-trading/README.md`](docs/copy-trading/README.md) |

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python >= 3.12.

```bash
uv sync                    # create .venv and install dependencies
uv run python run.py       # event-driven paper-trading demo (offline, generates test data)
uv run python research.py  # interactive research backtest
```

Research backtests also run non-interactively (omitted arguments fall back to the interactive menu), which is what scripts and scheduled jobs use:

```bash
uv run python research.py --symbol BTC/USDT --strategy bollinger --bars 365 --timeframe 1d
uv run python research.py --symbol ETH/USDT --strategy rsrs --bars 730 --leverage 2
```

`--strategy` accepts a menu index `1`-`5` or an alias: `dual-ma` / `trend-reversion` / `bollinger` / `rsrs` / `donchian`.
Every run writes the equity chart to `data/results/<SYMBOL>_<strategy>.png` and the equity / benchmark / drawdown series to `<name>_equity.csv` (read by the pi extension's ASCII curve) without asking for confirmation.

Market data comes from OKX (ccxt public endpoints). **Mainland China users need a proxy first**; elsewhere no proxy is required:

```dotenv
OKX_USE_PROXY=true
OKX_PROXY_PORT=7890
```

## Environment variables

`.env` is read by `src/config/settings.py` (missing values fall back to defaults). The template is `.env.example`: `cp .env.example .env` and fill in what you need.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OKX_API_KEY` / `OKX_SECRET` / `OKX_PASSPHRASE` | empty | OKX private-endpoint credentials; public market data does not need them |
| `OKX_USE_PROXY` | `false` | Route requests through an HTTP proxy |
| `OKX_PROXY_PORT` | `7890` | Proxy port; the host is fixed to `http://127.0.0.1:<port>` |
| `OKX_SANDBOX` | `false` | `true` = OKX demo trading |

## Entry points

| Command | Purpose |
| --- | --- |
| `uv run python research.py [--symbol ... --strategy ...]` | Vectorized backtest: arguments enter non-interactive mode, no arguments open the interactive menu; charts are saved automatically |
| `uv run python research.py --robust --bars 2900 --segments 4` | Robustness: full sample + rolling 1-year window distribution + segment stability + multiple-testing correction (BTC/ETH by default) |
| `uv run python research.py --carry --symbol c1,c2,SOL/USDT` | Funding-rate carry research: long spot + short equal perp, gross and net (Binance history; `--carry-days` sets the window) |
| `uv run python research.py --symbol c1 --maker` | Replace market orders with maker resting orders; see [Execution costs](#execution-costs) for quoting price and fills |
| `uv run python run.py` | Event-driven paper-trading demo (offline, generates test data): writes `trades_*.parquet` and `<symbol>_<strategy>_equity.csv`, marks the account to market at each bar close, and charges fees via `src/core/costs.py` (**by whether the order took liquidity**: market orders and same-side limit orders pay taker + slippage, only genuinely passive rests pay maker) |
| `uv run python scripts/crosscheck_crypto_bars.py --symbol BTC/USDT --bars 100` | Data reconciliation: same-source refetch must match OHLCV bar by bar, cross-source (Binance) compares the difference distribution only |
| `uv run python scripts/reconcile_execution.py --market spot [--place-orders --side sell]` | Execution reconciliation: compare per-fill fees against the model, optionally placing probe orders on demo trading (live is rejected) |

Backtest arguments (shared by `--robust` and single runs):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--bars` | 730 single / 2900 robust | Number of evaluation bars (most recent N); warmup bars stay outside the window, and the fetch depth is converted from evaluation + warmup into calendar days |
| `--segments` | 4 | Number of segments in the robustness check |
| `--trials` | this run's candidates | Number of trials N for multiple-testing correction; the more parameters and windows you have tried historically, the larger this should be |
| `--vol-target` | off | Volatility-targeted sizing; a bare flag means 40%, scaling a 0/1 signal into 0–100% exposure |
| `--vol-lookback` / `--vol-rebalance` | 30 / 7 | Volatility lookback bars and rebalance interval (controls turnover cost) |
| `--maker` | off | Use maker resting orders instead of market orders (see [Execution costs](#execution-costs)) |
| `--maker-patience` | 1 | Bars to wait for a rest before falling back to market (default quote 1bp away, wait 1 bar) |
| `--carry-days` | 2900 | Research window for `--carry` in calendar days (funding history), independent of `--bars`; data outside the window is excluded |

## Robustness checks

`--robust` prints four blocks:

| Block | Question it answers |
| --- | --- |
| Full sample + rolling 1-year window distribution | What does holding for any given year look like (worst window, share of positive windows, median window drawdown) |
| Segment stability | Does the return exist only in one segment (positive segments, segments beating the benchmark, worst segment) |
| Multiple-testing correction | After N tried candidates, is the Sharpe still skill: DSR ≥ 0.95 and PBO < 50% |
| Funding-rate carry | What survives after funding, margin and turnover costs |

DSR (Deflated Sharpe Ratio) and PBO (CSCV) live in `src/research/validation/overfitting.py`; volatility-targeted sizing lives in `src/research/backtest/position_sizing.py`, with its measured behaviour and falsification in `src/research/trend/vol_target.py`.

## Execution costs

There is exactly one source of fees: `src/core/costs.py`, with spot and perp separated (`SPOT_FEES` / `PERP_FEES`). Fees are decided **by whether this fill took liquidity, not by order type**: market orders and same-side limit orders (buy limit ≥ best ask) take liquidity and pay taker + slippage; only orders resting outside the market and touched by `low`/`high` count as passive and pay maker.

`--maker` switches order placement from market to resting orders, implemented in `src/research/backtest/execution.py`. The default quotes 1bp away (`offset=1bp`, patience 1 bar) — **quoting at the market (`offset=0`) fills within the same bar and therefore takes liquidity and must pay taker**; charging maker there is wrong.

Quote offsets and touch rates come from 1m data (129k BTCUSDT bars):

| Offset | Touch rate | Net edge | Note |
| --- | --- | --- | --- |
| 0bp (at market) | 100% | 0 (taker) | Fills in the same bar, so no maker fee |
| **1bp (default)** | 86.6% | **≈ +6.1bp/side** | Net of falling back to market when the rest is missed |
| 5bp | 61.9% | +4.6bp/side | Higher miss rate |

**Maker vs market is not "the same trade with a different fee"**: a resting order takes effect one bar later, so the position timing differs, and the output always prints the maker − market difference. Measured on Bollinger mean reversion over 200 bars (4 trades): maker 5.13% vs market 5.24%, i.e. −0.10% — the fee ceiling was worth only +0.36% (4 trades × 0.09%/side) and the one-bar entry delay ate it. The other side is timing bias: across 139 symbols × 4 strategies, resting 1bp away moves positions with a 25–37pp standard deviation (breakout-style RSRS nets −13.82pp), while quoting at market has zero deviation at the cost of paying full taker. During sharp rallies (previous 15 bars > +50bp) rests get run over and buys are charged as taking liquidity.

### Reconciliation: model vs real exchange fills

`scripts/reconcile_execution.py` places three minimal orders on OKX demo trading and compares fill fees against the model (criterion: per-side error < 1bp):

| Probe order | OKX tag | Actual | Model | Diff |
| --- | --- | --- | --- | --- |
| Market sell | taker | 10.000bp | 10.000bp | +0.000 |
| Limit sell at best bid (same side) | taker | 10.000bp | 10.000bp | +0.000 |
| Limit sell 1bp beyond best ask (passive) | maker | 8.000bp | 8.000bp | −0.000 |

Slippage: measured +0.01bp for a $20 notional market order against a 5bp model constant, i.e. **conservative for small orders**; real slippage scales with size, so read that constant as "for the size you plan to trade". `--place-orders` is only allowed with `OKX_SANDBOX=true`; live trading is rejected.

## Performance metrics

Backtest summaries and chart titles emit the metrics below, all computed in `src/research/backtest/engine.py`:

| Metric | Definition |
| --- | --- |
| Total / annualized return | First-to-last equity change; annualized as 365-day compounding |
| Benchmark | Buy-and-hold return over the same window |
| Calmar | Annualized return / \|max drawdown\| |
| Sharpe | Mean net return / standard deviation × √annualization |
| Sortino | Mean net return / downside deviation (MAR = 0) × √annualization |
| Max drawdown | Deepest peak-to-trough decline of the equity curve |
| Trades / win rate / profit factor | Aggregated over contiguous non-zero position runs; one entry plus one exit is one trade |
| Average hold / costs | Mean bars per trade; total fees and slippage |

## Data & configuration

- Cache path: `data/storage/crypto/<exchange_id>/<market>/<timeframe>/<SYMBOL>.parquet` (`market` = `spot` | `swap`); `data/` is not committed.
- Research outputs: numeric tables are always written as parquet (`compression="snappy"`), not CSV; the single exception is `data/results/<strategy>_equity.csv`, the data source for the pi extension's ASCII curve (zero-dependency TS parsing).
- Provenance: every successful run writes `data/results/provenance/<run_id>.json`, where `repro_hash` = sha256(config snapshot + data fingerprint + git HEAD/dirty flag), so the same command always yields the same hash (`src/research/validation/provenance.py`).
- Funding cache: `data/storage/crypto/binance/funding/8h/<SYMBOL>.parquet` (OKX keeps only ~100 days; long history comes from Binance back to 2019-09).
- Data source: ccxt, with the exchange instance injected by the caller (OKX for bars, Binance public endpoints for funding).
- Scheduled collection: five datasets/tasks are wired into Windows Task Scheduler (`nerd-trader-polymarket-hourly` at minute 7 hourly, `nerd-trader-deribit-daily` at 16:05, `nerd-trader-bars-daily` at 16:20 refreshing OKX spot daily bars + rebuilding the manifest + audit, `nerd-trader-paper-daily` at 16:45 appending the trend forward ledger, `nerd-trader-copy-daily` at 23:40 snapshotting the OKX copy-trading leaderboard + audit). Tasks call only `scripts/scheduled_run.sh <dataset>`; register/query/remove with `scripts/register_tasks.sh`; logs land per day in `logs/scheduled/<dataset>-<date>.log`.
- Prediction markets (Polymarket) self-hosted: `data/storage/polymarket/raw/<events|markets|prices|trades>/<name>.jsonl` (raw is append-only and de-duplicated), manifests in `data/manifests/polymarket_raw_*.json`; forward collection only (history is measured to be sealed — container ADR-0007), audit gate `scripts/audit_polymarket.py`, details in `docs/polymarket/README.md`.
- Deribit option chain self-hosted: `data/storage/deribit/raw/<options|futures>/<YYYY-MM-DD>.jsonl` (business key `(instrument, fetched_at)`, several rounds per day allowed), manifests in `data/manifests/deribit_raw_*.json`; it is the risk-neutral pricing anchor for the threshold ladder and a future hedge leg, fetched by `scripts/fetch_deribit.py`, audit gate `scripts/audit_deribit.py` (container ADR-0006), details in `docs/deribit/README.md`.
- OKX copy-trading leaderboard self-hosted: `data/storage/okx_copy/raw/lead_traders/<YYYY-MM-DD>.jsonl` (business key `(uniqueCode, dataVer)`, same-version reruns write no duplicate rows), manifest in `data/manifests/okx_copy_raw_lead_traders.json`; fetched by `scripts/fetch_okx_copy.py`, audit gate `scripts/audit_okx_copy.py`. Leaderboard returns are on the trader's own capital while copier P&L is separate; details in `docs/copy-trading/README.md`.

## Repository layout

```text
data/              data lake (not committed): storage/ bar and funding caches, results/ backtest outputs and provenance
scripts/           data and execution reconciliation: bulk backfill (fetch_crypto_bars), audit gates (audit_crypto_bars), manifest building, bar cross-checks, cost/slippage reconciliation, maker fill-rate study (maker_edge_study), signal scan (signal_scan), trend duel (trend_duel), per-strategy suite (strategy_suite), overlap and portfolio (strategy_correlation), funding fetch (fetch_funding_history), carry ranking (carry_ranking), cross-sectional factor suite (factor_suite), volatility-target study (vol_target_study), listing study (listing_study), forward paper ledger (tsmom_paper) and execution-sensitivity/overlay scan (tsmom_overlay_study); prediction-market forward collection (fetch_polymarket), audit gate (audit_polymarket) and three-spec comparison (polymarket_digital_study), option-chain snapshots (fetch_deribit) with its audit gate (audit_deribit), scheduled-task entry points (scheduled_run, register_tasks)
src/analytics/     performance metrics (single source of truth)
src/cli/           command line and interactive menu for research backtests
src/config/        env config, path constants, live parameter table
src/core/          event engine, data types, BarGenerator/ArrayManager, cost model
src/research/      vectorized backtest: engine, domains (market rules), strategies, robustness, overfitting, position_sizing, carry, factors, vol_target, listing, execution (maker rests), provenance
src/simulation/    event-driven: engine, mock_gateway, replay_loader
src/strategies/    event-driven strategies (dual_ma)
src/utils/         data dispatch and data sources
run.py / research.py  two thin entry shells (logic lives in src/cli/ and src/simulation/)
docs/              factors/models/library notes/trader notes/execution and signal research/trend research/portfolio research/carry research/onchain research/prediction markets
```

## Dependencies

- Dependencies are locked in `uv.lock`.
- The default index in `pyproject.toml` (`[[tool.uv.index]]`) points at the Tsinghua mirror, which is fast from mainland China and may be slow elsewhere. To use the official index, delete that block and re-run `uv lock`.

## Agent tooling

`.pi/extensions/backtest/` wires `research.py` into **pi** (a terminal coding agent): structured arguments to launch a backtest, streaming progress, result cards, run history and two-run comparison. Without pi installed nothing on this path executes, and deleting the directory breaks no command; see that directory's `README.md`.

## Docs index

The thirteen research lines above link to their own notes; the remaining directories:

| Directory | Contents |
| --- | --- |
| `docs/研究线索引.md` | **Code map**: module / CLI entry / criteria page / output directory per research line, plus the six-item checklist for adding a line |
| `docs/wisdom/` | Trader idea excerpts and methodology notes (chapter-by-chapter audit of de Prado against this repo, one "what it changed here" per chapter) |
| `docs/polymarket/` | Prediction markets: endpoint measurements, forward-collection contract and audit criteria (container ADR-0007) |
| `docs/deribit/` | Option-chain dataset: contract, fetch/audit commands and pitfalls (container ADR-0006) |
| `docs/copy-trading/` | OKX copy-trading leaderboard: field semantics, fetch/audit jobs and pre-registered forward-tracking criteria |
| `docs/models/` | Model notes |
| `docs/library/` | Book and video clippings |
| `docs/src.md/` | Strategy source notes |

## License

[Apache-2.0](LICENSE): commercial use, modification and redistribution are permitted, with a patent grant and a requirement to keep the copyright and disclaimer notices.

Copyright 2026 Xia Jiang
