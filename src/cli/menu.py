"""
src/cli/menu.py
交互式菜单：标的、周期、策略与参数的人工选择入口。
"""

from __future__ import annotations

import traceback
from typing import Any, cast

from src.research.registry import (
    DEFAULT_BARS,
    DEFAULT_LEVERAGE,
    DEFAULT_STRATEGY,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    STRATEGY_MAP,
    SYMBOL_PRESETS,
    resolve_strategy,
)
from src.research.runners import run_backtest
from src.utils.data_loader import TimeFrameStr


def customize_params(default_params: dict[str, Any]) -> dict[str, Any]:
    """交互式修改策略参数。"""
    print("\n⚙️ 参数设置 (直接回车保持默认):")
    new_params = default_params.copy()
    for key, default_val in default_params.items():
        while True:
            try:
                user_input = input(f"   - {key} [默认 {default_val}]: ").strip()
                if not user_input:
                    break
                target_type = type(default_val)
                new_params[key] = target_type(user_input)
                break
            except ValueError:
                print(f"   ❌ 格式错误，请输入有效的 {type(default_val).__name__}")
    return new_params


def select_symbol_ui() -> str:
    """标的选择界面。"""
    print("\n🎯 请选择回测标的:")
    quick_map = {}
    for category, items in SYMBOL_PRESETS.items():
        print(f"\n--- {category} ---")
        for key, (code, desc) in items.items():
            print(f"  [{key}] {desc} ({code})")
            quick_map[key] = code

    user_input = (
        input(f"\n请输入 [序号] 或 [代码] (默认 {DEFAULT_SYMBOL}): ").strip().lower()
    )
    if not user_input:
        return quick_map[DEFAULT_SYMBOL]
    return quick_map.get(user_input, user_input.upper())


def interactive_menu() -> None:
    print("🚀 Unified Quant Research Lab v2.0")
    print("-" * 40)

    while True:
        try:
            symbol = select_symbol_ui()
            timeframe = (
                input(f"\nK线周期 (1h, 4h, 1d) [默认 {DEFAULT_TIMEFRAME}]: ").strip()
                or DEFAULT_TIMEFRAME
            )
            bars_input = input(f"回测根数 [默认 {DEFAULT_BARS}]: ").strip()
            bars = int(bars_input) if bars_input else DEFAULT_BARS

            print("\n♟️ 选择策略:")
            for k, v in STRATEGY_MAP.items():
                print(f"  [{k}] {v['name']}")
            strat_choice = (
                input(f"请输入序号 [默认 {DEFAULT_STRATEGY}]: ").strip()
                or DEFAULT_STRATEGY
            )
            strategy_conf = resolve_strategy(strat_choice)

            if not strategy_conf:
                print("❌ 无效的策略序号")
                continue

            conf = strategy_conf.copy()
            if input("修改参数? (y/n) [默认 n]: ").strip().lower() == "y":
                conf["default_params"] = customize_params(conf["default_params"])

            lev_input = input(f"杠杆倍数 [默认 {DEFAULT_LEVERAGE:g}]: ").strip()
            leverage = float(lev_input) if lev_input else DEFAULT_LEVERAGE

            run_backtest(symbol, cast(TimeFrameStr, timeframe), conf, bars, leverage)

            if input("\n继续其他实验? (y/n) [默认 y]: ").strip().lower() == "n":
                print("👋 实验结束，再见！")
                break

        except KeyboardInterrupt:
            print("\n👋 用户中断实验")
            break
        except Exception:
            traceback.print_exc()
