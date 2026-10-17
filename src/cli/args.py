"""
src/cli/args.py
命令行参数：给了 --symbol 就走非交互模式，缺参数则进入交互菜单。
"""

from __future__ import annotations

import argparse
from typing import get_args

from src.research.backtest.position_sizing import (
    DEFAULT_REBALANCE_BARS,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_VOL_TARGET,
)
from src.research.validation.registry import (
    CARRY_DEFAULT_DAYS,
    DEFAULT_BARS,
    DEFAULT_LEVERAGE,
    DEFAULT_MAKER_PATIENCE,
    DEFAULT_SEGMENTS,
    DEFAULT_STRATEGY,
    DEFAULT_TIMEFRAME,
    ROBUST_DEFAULT_BARS,
    STRATEGY_ALIASES,
)
from src.utils.data_loader import TimeFrameStr


def parse_args() -> argparse.Namespace:
    """命令行参数：给了 --symbol 就走非交互模式，缺参数则进入交互菜单。"""
    parser = argparse.ArgumentParser(
        description="加密货币量化回测（省略参数则进入交互菜单）"
    )
    vol_target_help = (
        f"启用波动率目标仓位（默认关闭；只给开关则用 {DEFAULT_VOL_TARGET:.0%}）"
    )
    parser.add_argument(
        "--symbol", help="标的，如 BTC/USDT 或 c1，多个用逗号分隔（稳健性检验用）"
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        choices=get_args(TimeFrameStr),
        help="K 线周期",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help=(
            f"评估区间根数（默认单次回测 {DEFAULT_BARS}，稳健性检验 {ROBUST_DEFAULT_BARS}）"
        ),
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"策略序号或别名（{'/'.join(STRATEGY_ALIASES)}）",
    )
    parser.add_argument(
        "--leverage", type=float, default=DEFAULT_LEVERAGE, help="杠杆倍数"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--robust",
        action="store_true",
        help="稳健性检验：全样本 + 一年滚动窗口分布 + 分段稳定性 + 多重检验校正",
    )
    mode.add_argument(
        "--carry",
        action="store_true",
        help="资金费率 carry 研究：做多现货 + 做空等额永续的净收益",
    )
    parser.add_argument(
        "--carry-days",
        type=int,
        default=CARRY_DEFAULT_DAYS,
        help=f"carry 研究窗口（日历天，默认 {CARRY_DEFAULT_DAYS}）；窗口外的缓存数据不参与统计",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=DEFAULT_SEGMENTS,
        choices=range(2, 13),
        help="稳健性检验的分段数（至少 2）",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="多重检验校正的试验次数 N（默认取本次候选数；历史试过的参数/窗口越多，N 应越大）",
    )
    parser.add_argument(
        "--vol-target",
        nargs="?",
        const=DEFAULT_VOL_TARGET,
        type=float,
        default=None,
        # argparse 会对 help 做 % 插值，字面 % 必须写成 %%
        help=vol_target_help.replace("%", "%%"),
    )
    parser.add_argument(
        "--vol-lookback",
        type=int,
        default=DEFAULT_VOL_LOOKBACK,
        help="波动率目标的回看根数",
    )
    parser.add_argument(
        "--vol-rebalance",
        type=int,
        default=DEFAULT_REBALANCE_BARS,
        help="波动率目标的调仓间隔（根），控制换手成本",
    )
    parser.add_argument(
        "--maker",
        action="store_true",
        help="用 maker 挂单执行替代市价单：确定省手续费，但突破型策略可能受损",
    )
    parser.add_argument(
        "--maker-patience",
        type=int,
        default=DEFAULT_MAKER_PATIENCE,
        help="maker 挂单的耐心根数，超时回退市价",
    )
    args = parser.parse_args()

    # 静默忽略比报错危险：参数与模式不匹配时当场停下
    if args.maker and args.carry:
        parser.error("--maker 不适用于 --carry（carry 不产生挂单成交）")
    if args.maker_patience != DEFAULT_MAKER_PATIENCE and not args.maker:
        parser.error("--maker-patience 要配合 --maker 使用")
    return args
