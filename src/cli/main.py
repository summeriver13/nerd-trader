"""
src/cli/main.py
入口编排：解析参数 → 分派到单次回测 / 稳健性检验 / carry / 交互菜单。
"""

from __future__ import annotations

from typing import Any, cast

from src.cli.args import parse_args
from src.cli.menu import interactive_menu
from src.research.carry import (
    CAPITAL_MULTIPLE,
    ROLLING_DAYS,
    TURNOVER_SHARE,
    run_carry,
)
from src.research.execution import MakerPolicy
from src.research.position_sizing import VolatilityTarget
from src.research.provenance import record_run
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


def _common_config(
    args: Any, sizing: VolatilityTarget | None, maker: MakerPolicy | None
) -> dict[str, Any]:
    """进指纹的配置快照：只放会改变结果的参数。"""
    return {
        "timeframe": args.timeframe,
        "bars": args.bars,
        "leverage": args.leverage,
        "vol_target": sizing.target_vol if sizing else None,
        "vol_lookback": sizing.lookback if sizing else None,
        "vol_rebalance": sizing.rebalance_every if sizing else None,
        "maker_offset": maker.offset if maker else None,
        "maker_patience": maker.patience if maker else None,
    }


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
    symbols = resolve_symbols(args.symbol)

    if args.carry:
        ok = run_carry(symbols, args.carry_days)
        record_run(
            "carry",
            {
                "carry_days": args.carry_days,
                "capital_multiple": CAPITAL_MULTIPLE,
                "turnover_share": TURNOVER_SHARE,
                "rolling_days": ROLLING_DAYS,
            },
            symbols,
            timeframe,
        )
        if not ok:
            raise SystemExit(1)
        return

    if args.robust:
        bars = args.bars if args.bars is not None else ROBUST_DEFAULT_BARS
        config = _common_config(args, sizing, maker) | {
            "bars": bars,
            "segments": args.segments,
            "trials": args.trials,
            "strategies": list(STRATEGY_MAP),
        }
        ok = run_robustness(
            RobustnessOptions(
                symbols=symbols,
                timeframe=timeframe,
                bars=bars,
                segments=args.segments,
                leverage=args.leverage,
                trials=args.trials,
                sizing=sizing,
                maker=maker,
            )
        )
        # 即使部分标的取数失败也要留指纹：失败本身（数据为 null）已经进 hash
        record_run("robust", config, symbols, timeframe)
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
        config = _common_config(args, sizing, maker) | {
            "bars": bars,
            "strategy": strategy_conf["name"],
        }
        ok = run_backtest(
            resolve_symbol(args.symbol),
            timeframe,
            strategy_conf,
            bars,
            args.leverage,
            sizing=sizing,
            maker=maker,
        )
        record_run("backtest", config, symbols, timeframe)
        if not ok:
            raise SystemExit(1)
        return

    interactive_menu()
