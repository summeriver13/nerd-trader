"""scripts/tsmom_paper.py
趋势线前向纸面账本：每日跑一次，落「今天该持有什么」与当日纸面净值，不下单。

用法::

    uv run python scripts/tsmom_paper.py               # 追加今天的账（首次运行即起点）
    uv run python scripts/tsmom_paper.py --no-write    # 只看决策，不落账
    uv run python scripts/tsmom_paper.py --check       # 全历史回填，与研究读数对照
    uv run python scripts/tsmom_paper.py --verify      # 校验账本哈希链
    uv run python scripts/tsmom_paper.py --status      # 按预注册阈值裁决（0/1/2）

为什么需要它：趋势线的统计门槛已用尽（DSR/PBO/配对差），`docs/trend/README.md`
把「零前向验证」列为仍未解决的限制——前向记录是唯一还能增加真样本的动作。这一层
刻意与执行层解耦：它只产出目标权重（见 `src/research/trend/paper.py` 的口径说明）。

起点不回填：首次运行的起点取最新一根 K 线，账本第一行就是「从今天开始持仓」。
之后每次运行都从账本最后一行的次日续算，漏跑的日期会被补上（净值是确定性的），
但已写下的行绝不覆盖——同一日期重算结果不一致会作为「数据修订」报警。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.research.study.universe import liquidity_order, load_bars_dir
from src.research.trend.paper import (
    BTC_STEM,
    COST_RATE,
    LEDGER_DIR,
    LEDGER_NAME,
    PaperSpec,
    append_ledger,
    coverage_ok,
    forward_stats,
    ledger_panel,
    ledger_rows,
    nav_panel,
    positions,
    read_ledger,
    scale_signals,
    signal_panel,
    verify_ledger,
)
from src.research.trend.paper_status import VERDICT_KILL, retire_status, status_report

# 标的池：自持 OKX 现货日线（与研究侧同源）
SPOT_DIR = crypto_dataset_dir("okx", "spot", "1d")
DEFAULT_MIN_BARS = 365
DEFAULT_LIQUID_TOP = 30
DEFAULT_BOOK = 10_000.0
PRINT_TOP = 12


def load_pool(min_bars: int, pool: str, liquid_top: int) -> dict[str, pd.DataFrame]:
    """装载标的池：全池或流动性前 N 只（后者的口径见 universe.liquidity_order）。"""
    bars_map = load_bars_dir(SPOT_DIR, min_bars)
    if pool == "liquid":
        keep = liquidity_order(bars_map, liquid_top)
        bars_map = {stem: bars_map[stem] for stem in keep}
    return bars_map


def build(
    bars_map: dict[str, pd.DataFrame], spec: PaperSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """算出净值面板、持仓、当日目标与逐日收益；BTC 作为对照标的。"""
    returns = pd.DataFrame(
        {stem: bars["close"].pct_change() for stem, bars in bars_map.items()}
    )
    counts = returns.notna().sum(axis=1).astype(float)
    signals = signal_panel(bars_map, spec)
    held = positions(signals, counts)
    target = scale_signals(signals, counts)
    btc = cast(pd.Series, returns[BTC_STEM]) if BTC_STEM in returns.columns else None
    panel = nav_panel(held, returns, btc_returns=btc, start=spec.start)
    return panel, held, target, returns


def day(stamp: Any) -> str:
    """时间戳 → YYYY-MM-DD（面板索引的类型桩是泛型 Index，取不出 .date）。"""
    return str(cast(pd.Timestamp, pd.Timestamp(stamp)).date())


def resolve_start(ledger_path: Path, bars_map: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """续算起点：账本最后一行（补漏跑），没有账本就取最新一根 K 线。"""
    rows = read_ledger(ledger_path)
    if rows:
        return cast(pd.Timestamp, pd.Timestamp(str(rows[-1]["date"])))
    latest = [cast(pd.Timestamp, bars.index.max()) for bars in bars_map.values()]
    return cast(pd.Timestamp, pd.Series(latest).max())


def print_check(panel: pd.DataFrame, stats: dict[str, float]) -> None:
    """全历史回填读数，用来与研究侧 tsmom_ms 臂对照（应逐位一致）。"""
    print("=== 回填对照（全历史，非前向）===")
    print(
        f"期数 {int(stats.get('bars', 0))} ｜ 年化 {stats.get('annual_ret', 0.0):.1%}"
        f" ｜ 夏普 {stats.get('sharpe', 0.0):.2f}"
        f" ｜ 回撤 {stats.get('max_dd', 0.0):.1%}"
        f" ｜ 暴露 {stats.get('exposure', 0.0):.1%}"
    )
    btc_total = stats.get("btc_total_ret", 0.0)
    print(
        f"BTC 对照 总收益 {btc_total:.1%}"
        f" ｜ 年化 {stats.get('btc_annual_ret', 0.0):.1%}"
    )
    print(f"期末净值 {stats.get('nav', 1.0):.3f}（起 {day(panel.index[0])}）")
    print(
        "对照研究读数：tsmom_ms 现货费率口径 194.6%/13.1%/1.10/−17.3%（永续费率口径 198.6%/13.3%/1.11）"
    )


def print_decision(
    panel: pd.DataFrame,
    rows: list[dict[str, Any]],
    book: float,
    pool: str,
    *,
    first_row: bool = False,
) -> None:
    """打印当日决策：目标持仓、敞口、换手与需要下的单。"""
    latest = rows[-1] if rows else None
    if latest is None:
        print("账本为空，无决策")
        return
    target: dict[str, float] = dict(latest["target"])
    hold: dict[str, float] = dict(latest["hold"])
    stamp = panel.index[-1]
    print(f"=== 今日决策（K 线 {day(stamp)} 收盘后）===")
    print(
        f"池子 {pool} ｜ 标的数 {int(latest['n_symbols'])}"
        f" ｜ 目标持仓 {int(latest['n_target'])} 只"
        f" ｜ 目标敞口 {float(latest['exposure']):.2%}"
        f" ｜ 换手 {float(latest['turnover']):.4f}"
    )
    if not target:
        print("目标为空仓（信号全灭）——不需要下任何单")
        return
    print(
        f"账本 ${book:,.0f} ｜ 目标名义 ${book * float(latest['exposure']):,.0f} ｜ 前 {PRINT_TOP} 大持仓："
    )
    for stem, weight in list(target.items())[:PRINT_TOP]:
        print(f"  {stem:<14} {weight:>8.4%}   ${book * weight:>10,.0f}")
    if first_row:
        print(
            f"首行建仓：目标 {len(target)} 只全部新开，"
            f"建仓成本 ≈ ${book * float(latest['turnover']) * COST_RATE:,.2f}"
        )
        return
    changed = [
        stem
        for stem in set(target) | set(hold)
        if target.get(stem, 0.0) != hold.get(stem, 0.0)
    ]
    print(
        f"需要调整的标的 {len(changed)} 只（其中新开 {len(set(target) - set(hold))}、清仓 {len(set(hold) - set(target))}）"
    )


def print_forward(rows: list[dict[str, Any]]) -> None:
    """前向读数：只统计账本里真正走出来的那段。"""
    stats = forward_stats(ledger_panel(rows))
    if stats.get("bars", 0.0) < 1:
        print("=== 前向读数 ===\n期数为 0（今天刚建账，明天的行才会产生收益）")
        return
    print("=== 前向读数 ===")
    print(
        f"期数 {int(stats['bars'])} ｜ 净值 {stats['nav']:.4f}"
        f" ｜ 总收益 {stats['total_ret']:.2%}"
        f" ｜ 年化 {stats['annual_ret']:.1%}"
        f" ｜ 夏普 {stats['sharpe']:.2f}"
        f" ｜ 回撤 {stats['max_dd']:.1%}"
    )
    print(
        f"BTC 同期 总收益 {stats['btc_total_ret']:.2%}"
        f" ｜ 年化 {stats['btc_annual_ret']:.1%}"
        f" ｜ 账本平均暴露 {stats['exposure']:.1%}（前向记录满 180 天后才有意义）"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="趋势线前向纸面账本（不下单）")
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS)
    parser.add_argument("--pool", choices=("all", "liquid"), default="all")
    parser.add_argument("--liquid-top", type=int, default=DEFAULT_LIQUID_TOP)
    parser.add_argument(
        "--book", type=float, default=DEFAULT_BOOK, help="账本规模（仅用于打印名义额）"
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER_DIR / LEDGER_NAME)
    parser.add_argument(
        "--start", type=str, default=None, help="显式起点（仅供演示/对照）"
    )
    parser.add_argument("--no-write", action="store_true", help="只看决策，不落账")
    parser.add_argument(
        "--check", action="store_true", help="全历史回填，与研究读数对照"
    )
    parser.add_argument("--verify", action="store_true", help="校验账本哈希链后退出")
    parser.add_argument(
        "--status",
        action="store_true",
        help="按预注册判据裁决账本后退出（退出码 0/1/2 = 健康/观察/退役）",
    )
    args = parser.parse_args()

    if args.status:
        panel = ledger_panel(read_ledger(args.ledger))
        if panel.empty:
            print(f"账本 {args.ledger} 还没有行，先跑一次记账")
            return 1
        status = retire_status(panel)
        print(status_report(status))
        return status.exit_code

    if args.verify:
        ledger_rows_read = read_ledger(args.ledger)
        broken = verify_ledger(ledger_rows_read)
        print(
            f"账本 {args.ledger} ｜ 行数 {len(ledger_rows_read)} ｜ "
            + (f"断链于 {broken}" if broken else "哈希链完好")
        )
        return 1 if broken else 0

    bars_map = load_pool(args.min_bars, args.pool, args.liquid_top)
    if not bars_map:
        print(
            "标的池为空：先刷新日线缓存（scripts/scheduled_run.sh 不含日线，需单独跑）"
        )
        return 1

    if args.check:
        panel, _, _, returns = build(bars_map, PaperSpec())
        print_check(panel, forward_stats(panel))
        if not coverage_ok(returns):
            print(
                f"⚠ 末根 {day(panel.index[-1])} 只有 "
                f"{int(returns.notna().sum(axis=1).iloc[-1])} 只标的出数（缓存滞后），该根读数不可用"
            )
        return 0

    start = (
        pd.Timestamp(args.start) if args.start else resolve_start(args.ledger, bars_map)
    )
    spec = replace(PaperSpec(), start=start)
    panel, held, target, returns = build(bars_map, spec)
    if not coverage_ok(returns):
        counts = returns.notna().sum(axis=1)
        print(
            f"⚠ 数据滞后，拒绝记账：最新一根 {day(panel.index[-1])} 只有 "
            f"{int(counts.iloc[-1])} 只标的出数（近期中位 {int(counts.iloc[:-1].tail(30).median())}）——"
            f"分母 N_t 就是权重本身，此时记账会把持仓放大到远超常态。先刷新日线缓存。"
        )
        return 1
    rows = ledger_rows(panel, held.reindex(panel.index), target.reindex(panel.index))

    print_decision(
        panel, rows, args.book, args.pool, first_row=not read_ledger(args.ledger)
    )
    if not args.no_write:
        added, conflicts = append_ledger(args.ledger, rows)
        print(f"\n账本 {args.ledger} ｜ 起点 {day(panel.index[0])} ｜ 新增 {added} 行")
        if conflicts:
            print(
                f"⚠ 数据修订：{', '.join(conflicts)} 已在账本内，重算结果不一致（旧行保留）"
            )
    print()
    print_forward(read_ledger(args.ledger))
    # 判据随账打印：日志里只要出现 kill 就能被搜到，不需要额外跑一次 --status
    status = retire_status(ledger_panel(read_ledger(args.ledger)))
    print(status_report(status))
    # 只有退役才让任务失败：观察状态每天都会出现，不该把日志刷成红色
    return status.exit_code if status.verdict == VERDICT_KILL else 0


if __name__ == "__main__":
    raise SystemExit(main())
