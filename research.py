"""
research.py
科研控制台：集成动态预热缓冲与中文绘图修复版
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ccxt
import matplotlib.pyplot as plt
import pandas as pd

# 1. 绘图环境配置：解决中文乱码与负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 环境使用黑体
plt.rcParams["axes.unicode_minus"] = False

# 2. 路径补丁
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# noqa: E402
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.carry import run_carry  # noqa: E402
from src.research.domains.crypto import CryptoDomain  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402
from src.research.strategies.rsrs_trend_follower import RSRSTrendStrategy  # noqa: E402
from src.research.strategies.vector_bollinger_mr import (  # noqa: E402
    BollingerMRStrategy,  # noqa: E402
)
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.research.strategies.vector_trend_reversion import (
    TrendReversionStrategy,  # noqa: E402
)
from src.research.overfitting import print_overfitting  # noqa: E402
from src.research.position_sizing import (  # noqa: E402
    DEFAULT_REBALANCE_BARS,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_VOL_TARGET,
    VolatilityTarget,
    apply_volatility_target,
)
from src.research.robustness import (  # noqa: E402
    BuyAndHoldStrategy,
    build_report,
    print_reports,
)
from src.utils.data_loader import MarketDataHandler  # noqa: E402

# === 快捷预设菜单 ===
SYMBOL_PRESETS: dict[str, dict[str, tuple[str, str]]] = {
    "Crypto": {
        "c1": ("BTC/USDT", "比特币 (Bitcoin)"),
        "c2": ("ETH/USDT", "以太坊 (Ethereum)"),
    },
}

# === 菜单与 CLI 共用的默认值 ===
DEFAULT_SYMBOL = "c1"
DEFAULT_TIMEFRAME = "1d"
DEFAULT_DAYS = 730
DEFAULT_STRATEGY = "4"
DEFAULT_LEVERAGE = 1.0

# === 稳健性检验参数 ===
ROBUST_WARMUP = 700  # 覆盖 RSRS 最长预热，评估区间统一从第 700 根起算
ROBUST_DEFAULT_DAYS = 2900  # 默认取满可得历史（约 8 年）
DEFAULT_SEGMENTS = 4
DEFAULT_ROBUST_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
CARRY_DEFAULT_DAYS = 2900  # 资金费率取满可得历史（Binance BTC 自 2019-09 起）


def evaluation_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """截取最近 days 根作为评估区间。

    指标与信号已在全量帧上算好，预热根数留在窗口之外；
    历史不足 days 根时原样返回，由调用方提示区间缩短。
    """
    if days >= len(df):
        return df
    return cast(pd.DataFrame, df.iloc[-days:])


@dataclass(frozen=True)
class RobustnessOptions:
    """稳健性检验的一次运行配置。"""

    symbols: list[str]
    timeframe: str
    days: int
    segments: int
    leverage: float = DEFAULT_LEVERAGE
    trials: int | None = None
    sizing: VolatilityTarget | None = None

# 策略别名：CLI 里用可读名字，映射到菜单序号
STRATEGY_ALIASES: dict[str, str] = {
    "dual-ma": "1",
    "trend-reversion": "2",
    "bollinger": "3",
    "rsrs": "4",
}

# === 策略映射表 ===
STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "1": {
        "name": "自适应双均线 (Dual MA)",
        "class": DualMAStrategy,
        "default_params": {"trend_period": 120, "bull_fast": 10, "bull_slow": 30},
    },
    "2": {
        "name": "趋势+反转缝合怪 (Trend+Reversion)",
        "class": TrendReversionStrategy,
        "default_params": {"trend_period": 200, "bull_fast": 10},
    },
    "3": {
        "name": "布林带均值回归 (Bollinger MR)",
        "class": BollingerMRStrategy,
        "default_params": {
            "window": 20,
            "entry_std": 2.0,
            "exit_std": 0.5,
            "stop_loss_std": 4.0,
            "trend_ma_period": 200,
        },
    },
    "4": {
        "name": "RSRS 阻力支撑相对强度 (RSRS Trend)",
        "class": RSRSTrendStrategy,
        "default_params": {
            "rsrs_n": 18,
            "rsrs_m": 600,
            "buy_threshold": 0.7,
            "sell_threshold": -0.7,
            "max_leverage": 1.0,
            "stop_loss_mult": 2.5,
        },
    },
}


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


def run_backtest(
    symbol: str,
    timeframe: str,
    strategy_conf: dict[str, Any],
    days: int,
    leverage: float = 1.0,
    *,
    sizing: VolatilityTarget | None = None,
) -> None:
    """执行单次回测流程，支持动态数据预热。"""
    StrategyClass = strategy_conf["class"]
    params = strategy_conf["default_params"].copy()
    strategy_name = strategy_conf["name"]

    print(
        f"\n{'=' * 60}\n🧪 启动回测 | 策略: {strategy_name} | 标的: {symbol}\n{'=' * 60}"
    )

    # 1. 动态计算预热缓冲区
    required_buffer = 700 if "RSRS" in strategy_name else 200

    loader = MarketDataHandler(ccxt.okx(GLOBAL_CONFIG.to_ccxt_config()))  # type: ignore

    try:
        # 调用改进后的 prepare_data，传入 required_buffer
        df = loader.prepare_data(
            symbol, timeframe, days_back=days, required_buffer=required_buffer
        )
        if df is None or df.empty:
            print("❌ 数据为空，请检查 OKX 网络与代理配置。")
            return
        print(f"📊 数据准备完毕: 总行数 {len(df)}, 起始时间 {df.index[0]}")
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    # 2. 策略计算
    strategy = StrategyClass(**params)
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    if sizing is not None:
        df = apply_volatility_target(df, sizing)
        print(
            f"⚖️ 波动率目标 {sizing.target_vol:.0%}"
            f"（回看 {sizing.lookback} 根，每 {sizing.rebalance_every} 根调仓）"
        )

    # 3. 截取评估区间：days 表示回测窗口长度，预热根数留在窗口之外
    window = evaluation_window(df, days)
    print(
        f"📊 评估区间: {window.index[0]} → {window.index[-1]}"
        f"（{len(window)} 根，预热 {len(df) - len(window)} 根）"
    )

    # 4. 引擎适配
    domain = CryptoDomain(fee_rate=0.0005, slippage=0.0005)
    engine = BacktestEngine(domain=domain, initial_capital=10000.0, leverage=leverage)
    result = engine.run(window)

    if not result:
        print("⚠️ 回测未产生有效交易。")
        return

    # 5. 结果输出摘要（风险调整后收益优先看卡尔玛）
    print(f"\n📊 结果摘要 | {symbol}")
    print(
        f"💰 总收益: {result['total_ret']:.2%} | 年化: {result['annual_ret']:.2%} "
        f"| 基准: {result.get('bench_ret', 0):.2%}"
    )
    print(
        f"🛡️ 卡尔玛: {result['calmar_ratio']:.2f} | 夏普: {result['sharpe_ratio']:.2f} "
        f"| 索提诺: {result['sortino_ratio']:.2f} "
        f"| 最大回撤: {result['max_drawdown']:.2%}"
    )
    print(
        f"🔄 交易: {result['trade_count']} 回合 | 胜率: {result['win_rate']:.1%} "
        f"| 盈亏比: {result['profit_factor']:.2f} "
        f"| 平均持仓: {result['avg_hold_bars']:.1f} 根 "
        f"| 费用: {result['total_cost']:.2f} USDT"
    )

    # 5. 图表自动落盘，不再询问，方便批量与定时任务
    save_path = (
        PROJECT_ROOT
        / "data"
        / "results"
        / f"{symbol.replace('/', '_')}_{strategy_name}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    engine.plot(result, title=f"{symbol} - {strategy_name}", save_path=str(save_path))


def parse_args() -> argparse.Namespace:
    """命令行参数：给了 --symbol 就走非交互模式，缺参数则进入交互菜单。"""
    parser = argparse.ArgumentParser(
        description="加密货币量化回测（省略参数则进入交互菜单）"
    )
    parser.add_argument(
        "--symbol", help="标的，如 BTC/USDT 或 c1，多个用逗号分隔（稳健性检验用）"
    )
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="K 线周期")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"回测天数（默认单次回测 {DEFAULT_DAYS}，稳健性检验 {ROBUST_DEFAULT_DAYS}）",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"策略序号或别名（{'/'.join(STRATEGY_ALIASES)}）",
    )
    parser.add_argument(
        "--leverage", type=float, default=DEFAULT_LEVERAGE, help="杠杆倍数"
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="稳健性检验：全样本 + 一年滚动窗口分布 + 分段稳定性 + 多重检验校正",
    )
    parser.add_argument(
        "--carry",
        action="store_true",
        help="资金费率 carry 研究：做多现货 + 做空等额永续的净收益",
    )
    parser.add_argument(
        "--segments", type=int, default=DEFAULT_SEGMENTS, help="稳健性检验的分段数"
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
        help=f"启用波动率目标仓位（默认关闭；只给开关则用 {DEFAULT_VOL_TARGET:.0%}）",
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
    return parser.parse_args()


def resolve_strategy(choice: str) -> dict[str, Any] | None:
    """把菜单序号或可读别名解析成策略配置。"""
    return STRATEGY_MAP.get(STRATEGY_ALIASES.get(choice.lower(), choice))


def resolve_symbol(code: str) -> str:
    """把预设键（c1/c2）解析成交易对，非预设值原样返回。"""
    for items in SYMBOL_PRESETS.values():
        if code.lower() in items:
            return items[code.lower()][0]
    return code.upper()


def resolve_symbols(argument: str | None) -> list[str]:
    """--symbol 支持逗号分隔多个标的，缺省用默认标的集合。"""
    if not argument:
        return list(DEFAULT_ROBUST_SYMBOLS)
    return [resolve_symbol(code) for code in argument.split(",")]


def run_robustness(options: RobustnessOptions) -> None:
    """稳健性检验：多标的 × 全策略，输出窗口分布、分段稳定性与多重检验校正。"""
    loader = MarketDataHandler(ccxt.okx(GLOBAL_CONFIG.to_ccxt_config()))  # type: ignore
    domain = CryptoDomain(fee_rate=0.0005, slippage=0.0005)
    engine = BacktestEngine(
        domain=domain, initial_capital=10000.0, leverage=options.leverage
    )
    candidates: list[tuple[str, Any]] = [
        (conf["name"], conf["class"](**conf["default_params"]))
        for conf in STRATEGY_MAP.values()
    ]
    candidates.append(("买入持有 (基准)", BuyAndHoldStrategy()))

    for symbol in options.symbols:
        try:
            df = loader.prepare_data(
                symbol,
                options.timeframe,
                days_back=options.days + ROBUST_WARMUP,
                required_buffer=ROBUST_WARMUP,
            )
        except Exception as e:
            print(f"❌ {symbol} 数据获取失败: {e}")
            continue
        if df is None or df.empty:
            print(f"❌ {symbol} 无可用数据，跳过。")
            continue
        if len(df) < options.days + ROBUST_WARMUP:
            print(
                f"⚠️ {symbol} 仅 {len(df)} 根历史"
                f"（不足 {options.days + ROBUST_WARMUP}），评估区间与前段预热均缩短。"
            )

        eval_index = pd.DatetimeIndex(evaluation_window(df, options.days).index)
        reports = []
        series_map: dict[str, pd.Series] = {}
        for name, strategy in candidates:
            data = strategy.generate_signals(strategy.calculate_indicators(df.copy()))
            if options.sizing is not None:
                data = apply_volatility_target(data, options.sizing)
            result = engine.run(evaluation_window(data, options.days))
            if result:
                reports.append(build_report(name, result, options.segments))
                series_map[name] = result["df"]["net_ret"]

        print_reports(symbol, eval_index, reports)
        print_overfitting(symbol, series_map, options.trials)


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
            days_input = input(f"回测天数 [默认 {DEFAULT_DAYS}]: ").strip()
            days = int(days_input) if days_input else DEFAULT_DAYS

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

            run_backtest(symbol, timeframe, conf, days, leverage)

            if input("\n继续其他实验? (y/n) [默认 y]: ").strip().lower() == "n":
                print("👋 实验结束，再见！")
                break

        except KeyboardInterrupt:
            print("\n👋 用户中断实验")
            break
        except Exception:
            traceback.print_exc()


def main() -> None:
    args = parse_args()
    sizing = (
        VolatilityTarget(
            target_vol=args.vol_target,
            lookback=args.vol_lookback,
            rebalance_every=args.vol_rebalance,
        )
        if args.vol_target
        else None
    )
    if args.carry:
        days = args.days if args.days is not None else CARRY_DEFAULT_DAYS
        run_carry(resolve_symbols(args.symbol), days)
        return

    if args.robust:
        symbols = resolve_symbols(args.symbol)
        days = args.days if args.days is not None else ROBUST_DEFAULT_DAYS
        run_robustness(
            RobustnessOptions(
                symbols=symbols,
                timeframe=args.timeframe,
                days=days,
                segments=args.segments,
                leverage=args.leverage,
                trials=args.trials,
                sizing=sizing,
            )
        )
        return

    if args.symbol:
        strategy_conf = resolve_strategy(args.strategy)
        if strategy_conf is None:
            print(
                f"❌ 未知策略 {args.strategy}，可用: {'/'.join(STRATEGY_MAP)} "
                f"或 {'/'.join(STRATEGY_ALIASES)}"
            )
            return
        days = args.days if args.days is not None else DEFAULT_DAYS
        run_backtest(
            resolve_symbol(args.symbol),
            args.timeframe,
            strategy_conf,
            days,
            args.leverage,
            sizing=sizing,
        )
        return

    interactive_menu()


if __name__ == "__main__":
    main()
