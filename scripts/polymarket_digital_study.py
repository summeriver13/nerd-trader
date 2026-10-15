"""三口径对照面板：Polymarket 阈值阶梯 vs Deribit 风险中性 vs 自持历史 bootstrap。

回答「这批阶梯里哪些档位相对两个参照同时偏贵/偏便宜」，并给出每档的隐含波动率、
Deribit 一侧的报价宽度、以及卖尾方向必须扣掉的成本。分析逻辑在
`src/research/polymarket_digital.py`，本脚本只负责取数、装配与落盘。

判读口径：两侧 payoff 不同源（Deribit 无 touch 合约），差额是**相对价值**不是无风险套利。

用法：
    uv run python scripts/polymarket_digital_study.py                  # BTC，自动选最大那张阶梯
    uv run python scripts/polymarket_digital_study.py --list           # 只列阶梯清单
    uv run python scripts/polymarket_digital_study.py --offline        # 用最近落盘的链，不联网
    uv run python scripts/polymarket_digital_study.py --ladder dip-to/2025-11-24
    uv run python scripts/polymarket_digital_study.py --block-hours 168 --vol-target 0.335

输出：data/results/polymarket/deribit_chain_<CCY>.json（链快照）
      data/results/polymarket/deribit_vs_polymarket_<CCY>.csv（对照表）
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence

import pandas as pd

from src.config.paths import (
    POLYMARKET_MARKETS,
    POLYMARKET_RESULTS_ROOT,
    polymarket_dataset_dir,
)
from src.research.polymarket_digital import (
    ComparisonRow,
    comparison_table,
    smile_from_chain,
)
from src.research.polymarket_ladder import LadderRow, load_ladders, rule_warnings
from src.research.price_bootstrap import (
    BootstrapSpec,
    PathStats,
    hourly_returns,
    simulate_paths,
)
from src.utils.data_sources.base import create_session
from src.utils.data_sources.deribit import (
    DeribitChain,
    fetch_chain,
    forward_curve,
    load_chain,
    nearest_expiry,
    save_chain,
)

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
ASSETS = {"BTC": "bitcoin", "ETH": "ethereum"}
DEFAULT_CURRENCY = "BTC"
HOURS_PER_DAY = 24
CENTS = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阈值阶梯三口径对照")
    parser.add_argument("--currency", default=DEFAULT_CURRENCY, choices=sorted(SYMBOLS))
    parser.add_argument("--ladder", default=None, help="阶梯键，如 bitcoin/reach/2025-11-24")
    parser.add_argument("--list", action="store_true", help="只列阶梯清单后退出")
    parser.add_argument("--offline", action="store_true", help="用最近落盘的链，不联网")
    parser.add_argument("--lookback-days", type=int, default=365, help="bootstrap 回看天数")
    parser.add_argument("--block-hours", type=int, default=24, help="重抽样块长（小时）")
    parser.add_argument("--paths", type=int, default=20_000, help="模拟路径数")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vol-target", type=float, default=None, help="按目标波动率缩放")
    return parser.parse_args()


def read_snapshots() -> list[dict[str, object]]:
    """读全部 markets 快照行；同一市场在快照历史里有多行，这里保留最新一行。"""
    latest: dict[str, dict[str, object]] = {}
    for path in sorted(polymarket_dataset_dir(POLYMARKET_MARKETS).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record: dict[str, object] = json.loads(line)
                latest[str(record.get("conditionId"))] = record
    return list(latest.values())


def chain_for(args: argparse.Namespace) -> DeribitChain:
    """取链：默认联网（三个请求），--offline 用上次落盘的快照。"""
    path = POLYMARKET_RESULTS_ROOT / f"deribit_chain_{args.currency}.json"
    if args.offline:
        chain = load_chain(path)
        print(f"离线链：{path.name}（抓取于 {chain.fetched_at:%Y-%m-%d %H:%M:%S} UTC）")
        return chain
    chain = fetch_chain(create_session(), args.currency)
    save_chain(chain, path)
    return chain


def pick_ladder(
    ladders: dict[str, list[LadderRow]], wanted: str | None, asset: str
) -> list[LadderRow]:
    """选一张阶梯：显式给键就按键，否则取档位最多的那张（并打印清单供核对）。"""
    scoped = {key: rows for key, rows in ladders.items() if key.startswith(f"{asset}/")}
    if not scoped:
        raise SystemExit(f"没有 {asset} 的阈值阶梯")
    for key in sorted(scoped):
        print(f"  {key:32} {len(scoped[key]):2} 档")
    if wanted is not None:
        if wanted not in scoped:
            raise SystemExit(f"没有阶梯 {wanted}；清单见上")
        return scoped[wanted]
    return max(scoped.values(), key=len)


def bootstrap_pair(
    args: argparse.Namespace, *, hours: int
) -> dict[str, PathStats]:
    """两条参照：去漂移与含历史漂移——市场价常夹在这两者之间，必须同时给。"""
    returns = hourly_returns(SYMBOLS[args.currency], lookback_days=args.lookback_days)
    common = {
        "hours": hours,
        "paths": args.paths,
        "block_hours": args.block_hours,
        "seed": args.seed,
        "vol_target": args.vol_target,
    }
    print(f"bootstrap：{returns.size} 根 1h 收益、{args.paths} 条路径 × {hours} 步")
    return {
        "zero": simulate_paths(returns, BootstrapSpec(demean=True, **common)),
        "history": simulate_paths(returns, BootstrapSpec(demean=False, **common)),
    }


def to_frame(table: Sequence[ComparisonRow]) -> pd.DataFrame:
    """对照表转 DataFrame；附加一列与 Deribit 的差（正 = 市场偏贵）。"""
    frame = pd.DataFrame([dataclasses.asdict(row) for row in table])
    frame["diff_vs_deribit"] = [row.diff_vs_deribit for row in table]
    return frame


def print_table(table: Sequence[ComparisonRow]) -> None:
    """终端只印判读需要的列；完整列在 CSV 里。带 * 表示高出 Deribit 误差带上沿。"""
    print(
        f"\n{'档位':>9} {'向':>4} {'市场':>7} {'市场σ':>7} {'德σ':>6} {'德比':>7}"
        f" {'带下':>6} {'带上':>6} {'差':>7} {'历史0':>7} {'市宽%':>6} {'德宽%':>6}"
    )
    for row in table:
        mark = "*" if row.beyond_deribit_band else " "
        print(
            f"{row.strike:>9,.0f} {row.direction:>4}"
            f" {row.market_price * CENTS:>7.2f}"
            f" {_pct(row.market_iv)}"
            f" {_pct(row.deribit_iv)}"
            f" {_pct(row.deribit_touch)}"
            f" {_pct(row.deribit_touch_low)}"
            f" {_pct(row.deribit_touch_high)}{mark}"
            f" {_pct(row.diff_vs_deribit)}"
            f" {_pct(row.bootstrap_zero_drift)}"
            f" {_pct(row.market_relative_spread)}"
            f" {_pct(row.deribit_option_spread)}"
        )


def _pct(value: float | None) -> str:
    """单元格：None 留空，数字按百分数两位（价格与波动率都是同一量纲）。"""
    return "      -" if value is None else f"{value * CENTS:>7.2f}"


def main() -> int:
    args = parse_args()
    return run(args)


def run(args: argparse.Namespace) -> int:
    ladders = load_ladders(read_snapshots())
    if not ladders:
        raise SystemExit("没有可用的阈值阶梯——先跑 scripts/fetch_polymarket.py")
    asset = ASSETS[args.currency]
    print(f"{asset} 的阶梯清单")
    rows = pick_ladder(ladders, args.ladder, asset)
    if args.list:
        return 0
    for warning in rule_warnings(rows):
        print(f"⚠️ {warning}")

    end_at = max(row.end_at for row in rows)
    chain = chain_for(args)
    forward = forward_curve(chain, end_at)
    horizon = (end_at - chain.fetched_at).days / 365.0
    print(
        f"锚点：{args.currency} 指数 {chain.index_price:,.0f}、{forward.via} 推 carry"
        f" {forward.carry * 100:.2f}%/yr、目标远期 {forward.forward:,.0f}、"
        f"剩余 {horizon * 365:.1f} 天"
    )
    expiry = nearest_expiry(chain, end_at)
    smile = smile_from_chain(chain, expiry, forward.forward)
    print(f"微笑：{expiry} 共 {smile.strikes.size} 档（{smile.strikes.min():,.0f}~{smile.strikes.max():,.0f}）")

    stats = bootstrap_pair(args, hours=round(horizon * 365 * HOURS_PER_DAY))
    table = comparison_table(
        rows, spot=chain.index_price, horizon=horizon, smile=smile, bootstrap=stats
    )
    print_table(table)
    path = POLYMARKET_RESULTS_ROOT / f"deribit_vs_polymarket_{args.currency}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(table).to_csv(path, index=False, encoding="utf-8")
    print(f"\n已落盘：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
