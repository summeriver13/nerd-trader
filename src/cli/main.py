"""
src/cli/main.py
入口编排：解析参数 → 分派到单次回测 / 稳健性检验 / carry / 交互菜单。
"""

from __future__ import annotations

from typing import cast

from src.cli.args import parse_args
from src.cli.menu import interactive_menu
from src.research.carry import run_carry
from src.research.execution import MakerPolicy
from src.research.position_sizing import VolatilityTarget
from src.research.registry import (
    DEFAULT_BARS,
    ROBUST_DEFAULT_BARS,
    STRATEGY_ALIASES,
    STRATEGY_MAP,
    resolve_strategy,
    resolve_symbol,
    resolve_symbols,
)
from src.research.runners import RobustnessOptions, run_backtest, run_robustness
from src.utils.console import force_utf8_stdout, setup_logging
from src.utils.data_loader import TimeFrameStr


def main() -> None:
    force_utf8_stdout()  # 必须在任何 print 之前，否则 Windows 重定向遇到 emoji 会崩
    setup_logging()  # 日志出口（data_loader / crypto 的取数诊断走 logger）
    args = parse_args()
    # argparse 已用 TimeFrameStr 的取值做 choices 校验，这里只是把 str 收窄回 Literal
    timeframe = cast(TimeFrameStr, args.timeframe)
    sizing = (
        VolatilityTarget(
            target_vol=args.vol_target,
            lookback=args.vol_lookback,
            rebalance_every=args.vol_rebalance,
        )
        if args.vol_target
        else None
    )
    maker = MakerPolicy(patience=args.maker_patience) if args.maker else None

    if args.carry:
        if not run_carry(resolve_symbols(args.symbol), args.carry_days):
            raise SystemExit(1)
        return

    if args.robust:
        bars = args.bars if args.bars is not None else ROBUST_DEFAULT_BARS
        ok = run_robustness(
            RobustnessOptions(
                symbols=resolve_symbols(args.symbol),
                timeframe=timeframe,
                bars=bars,
                segments=args.segments,
                leverage=args.leverage,
                trials=args.trials,
                sizing=sizing,
                maker=maker,
            )
        )
        if not ok:
            raise SystemExit(1)
        return

    if args.symbol:
        strategy_conf = resolve_strategy(args.strategy)
        if strategy_conf is None:
            print(
                f"❌ 未知策略 {args.strategy}，可用: {'/'.join(STRATEGY_MAP)} "
                f"或 {'/'.join(STRATEGY_ALIASES)}"
            )
            raise SystemExit(2)  # 2 = 用法错误，与 argparse 一致
        bars = args.bars if args.bars is not None else DEFAULT_BARS
        ok = run_backtest(
            resolve_symbol(args.symbol),
            timeframe,
            strategy_conf,
            bars,
            args.leverage,
            sizing=sizing,
            maker=maker,
        )
        if not ok:
            raise SystemExit(1)
        return

    interactive_menu()
