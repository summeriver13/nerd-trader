"""
src/research/runners.py
回测编排：单次回测与稳健性检验的取数、执行、落盘。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.metrics import bars_per_year
from src.analytics.reporting import save_equity_series
from src.config.paths import RESULTS_ROOT
from src.research.data_window import fetch_window, load_window, open_loader
from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine
from src.research.execution import MakerPolicy
from src.research.overfitting import print_overfitting
from src.research.position_sizing import VolatilityTarget, apply_volatility_target
from src.research.registry import DEFAULT_LEVERAGE, STRATEGY_MAP
from src.research.robustness import build_report, print_reports
from src.research.strategies.buy_and_hold import BuyAndHoldStrategy
from src.utils.data_loader import MarketDataHandler, TimeFrameStr

# 绘图环境配置：解决中文乱码与负号显示（字体按平台回退）
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
]
plt.rcParams["axes.unicode_minus"] = False

# === 回测口径（单次回测与稳健性检验共用）===
INITIAL_CAPITAL = 10_000.0  # 初始资金 USDT
TAKER_FEE_RATE = 0.0005  # OKX 永续 VIP0 taker 费率
SLIPPAGE = 0.0005  # 单边滑点假设


def describe_maker(maker: MakerPolicy | None) -> str | None:
    """maker 挂单参数的一行描述；未启用返回 None"""
    if maker is None or not maker.enabled:
        return None
    where = "市价" if maker.offset == 0 else f"市价下方 {maker.offset * 10000:.0f}bp"
    return f"🎯 maker 挂单执行：挂{where}、耐心 {maker.patience} 根、超时回退市价"


def evaluation_window(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    """截取最近 bars 根作为评估区间。

    指标与信号已在全量帧上算好，预热根数留在窗口之外；
    历史不足 bars 根时原样返回，由调用方提示区间缩短。
    """
    if bars >= len(df):
        return df
    return cast(pd.DataFrame, df.iloc[-bars:])


@dataclass(frozen=True)
class RobustnessOptions:
    """稳健性检验的一次运行配置。"""

    symbols: list[str]
    timeframe: TimeFrameStr
    bars: int
    segments: int
    leverage: float = DEFAULT_LEVERAGE
    trials: int | None = None
    sizing: VolatilityTarget | None = None
    maker: MakerPolicy | None = None


def _print_result(result: dict[str, Any], symbol: str) -> None:
    """结果摘要：风险调整后收益优先看卡尔玛。"""
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
    fill_rate = result.get("maker_fill_rate")
    if fill_rate is not None:
        fills = result["fills"]
        n_maker = fills["fill"].eq("maker").sum()
        print(
            f"🎯 maker 成交率: {fill_rate:.1%}（挂单成交 {n_maker} 笔 / "
            f"市价回退 {len(fills) - n_maker} 笔）"
        )


def _persist_outputs(
    result: dict[str, Any], symbol: str, strategy_name: str, engine: BacktestEngine
) -> None:
    """图表与净值序列落盘：批量与定时任务不询问，直接写。"""
    save_path = RESULTS_ROOT / f"{symbol.replace('/', '_')}_{strategy_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    engine.plot(result, title=f"{symbol} - {strategy_name}", save_path=str(save_path))

    # 终端内字符曲线读 CSV（PNG 程序读不了）
    equity_path = save_path.with_name(f"{save_path.stem}_equity.csv")
    save_equity_series(result, equity_path)
    print(f"📈 净值序列已保存: {equity_path}")


def run_backtest(
    symbol: str,
    timeframe: TimeFrameStr,
    strategy_conf: dict[str, Any],
    bars: int,
    leverage: float,
    *,
    sizing: VolatilityTarget | None = None,
    maker: MakerPolicy | None = None,
) -> bool:
    """执行单次回测流程。返回是否跑出结果，由 CLI 决定退出码。"""
    strategy_class = strategy_conf["class"]
    strategy_name = strategy_conf["name"]
    print(
        f"\n{'=' * 60}\n🧪 启动回测 | 策略: {strategy_name} | 标的: {symbol}"
        f"\n{'=' * 60}"
    )

    # 1. 预热根数由策略自己声明；抓数深度把「评估 + 预热」的根数换算成日历天
    strategy = strategy_class(**strategy_conf["default_params"])
    df = fetch_window(symbol, timeframe, bars, strategy.warmup_bars)
    if df is None:
        print("❌ 取数失败：见上方 traceback（标的代码拼写、网络或代理）")
        return False
    print(f"📊 数据准备完毕: 总行数 {len(df)}, 起始时间 {df.index[0]}")

    # 2. 策略计算
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    if sizing is not None:
        df = apply_volatility_target(df, sizing)
        print(
            f"⚖️ 波动率目标 {sizing.target_vol:.0%}"
            f"（回看 {sizing.lookback} 根，每 {sizing.rebalance_every} 根调仓）"
        )

    # 3. 截取评估区间：bars 是评估窗根数，预热根数留在窗口之外
    window = evaluation_window(df, bars)
    print(
        f"📊 评估区间: {window.index[0]} → {window.index[-1]}"
        f"（{len(window)} 根，预热 {len(df) - len(window)} 根）"
    )

    # 4. 引擎适配
    engine = BacktestEngine(
        domain=CryptoDomain(fee_rate=TAKER_FEE_RATE, slippage=SLIPPAGE),
        initial_capital=INITIAL_CAPITAL,
        leverage=leverage,
        maker=maker,
    )
    maker_note = describe_maker(maker)
    if maker_note is not None:
        print(maker_note)
    result = engine.run(window)
    if not result:
        print("⚠️ 回测无结果（数据为空）。")
        return False

    _print_result(result, symbol)
    _persist_outputs(result, symbol, strategy_name, engine)
    return True


def run_robustness(options: RobustnessOptions) -> bool:
    """稳健性检验：多标的 × 全策略，输出窗口分布、分段稳定性与多重检验校正。"""
    domain = CryptoDomain(fee_rate=TAKER_FEE_RATE, slippage=SLIPPAGE)
    engine = BacktestEngine(
        domain=domain,
        initial_capital=INITIAL_CAPITAL,
        leverage=options.leverage,
        maker=options.maker,
    )
    maker_note = describe_maker(options.maker)
    if maker_note is not None:
        print(maker_note)
    candidates: list[tuple[str, Any]] = [
        (conf["name"], conf["class"](**conf["default_params"]))
        for conf in STRATEGY_MAP.values()
    ]
    candidates.append(("买入持有 (基准)", BuyAndHoldStrategy()))
    # 抓数深度按最长的策略预热算，免得某个策略的指标在窗口起点还没热透
    warmup = max(strategy.warmup_bars for _, strategy in candidates)

    with open_loader() as loader:
        return _run_symbols(loader, options, engine, candidates, warmup)


def _run_symbols(
    loader: MarketDataHandler,
    options: RobustnessOptions,
    engine: BacktestEngine,
    candidates: list[tuple[str, Any]],
    warmup: int,
) -> bool:
    """逐标的跑稳健性三张表，返回是否全部取到数据。"""
    ok = True
    for symbol in options.symbols:
        try:
            df = load_window(loader, symbol, options.timeframe, options.bars, warmup)
        except Exception:
            traceback.print_exc()
            print(f"❌ {symbol} 数据获取失败（见上方 traceback）")
            ok = False
            continue
        if df is None:
            print(f"❌ {symbol} 无可用数据，跳过。")
            ok = False
            continue

        eval_index = pd.DatetimeIndex(evaluation_window(df, options.bars).index)
        # 滚动窗口是「一年」：按评估区间的实际周期折算根数（1h 下 365 根只有 15 天）
        window = int(bars_per_year(eval_index))
        reports = []
        series_map: dict[str, pd.Series] = {}
        for name, strategy in candidates:
            data = strategy.generate_signals(strategy.calculate_indicators(df.copy()))
            if options.sizing is not None:
                data = apply_volatility_target(data, options.sizing)
            result = engine.run(evaluation_window(data, options.bars))
            if result:
                reports.append(build_report(name, result, options.segments, window))
                series_map[name] = result["df"]["net_ret"]

        print_reports(symbol, eval_index, reports, window)
        print_overfitting(symbol, series_map, options.trials)
    return ok
