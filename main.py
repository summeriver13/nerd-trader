from typing import Type

import ccxt
import pandas as pd

from config.settings import GLOBAL_CONFIG
from src.backtest.engine import BacktestEngine
from src.data_loader import MarketDataHandler
from src.strategies.base import BaseStrategy
from src.strategies.bollinger_mr import BollingerMRStrategy
from src.strategies.dual_ma import DualMAStrategy


def run_strategy_backtest(
    symbol: str,
    timeframe: str,
    strategy_cls: Type[BaseStrategy],
    strategy_params: dict,
    data_handler: MarketDataHandler,
):
    """
    通用策略运行函数
    """
    strategy_name = strategy_cls.__name__.replace("Strategy", "")
    print(f"\n{'=' * 60}")
    print(f"🧪 测试: {strategy_name} | 标的: {symbol} | 周期: {timeframe}")
    print(f"{'=' * 60}")

    # 1. 准备数据
    # 均值回归通常在较短周期(1h/15m)表现更好，趋势在长周期(4h/1d)表现更好
    # 这里根据传入的 timeframe 动态拉取
    days = 365 if timeframe == "1h" else 730
    df = data_handler.prepare_data(symbol=symbol, timeframe=timeframe, days_back=days)

    # 2. 初始化策略
    strategy = strategy_cls(**strategy_params)

    # 3. 计算信号
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    # 4. 运行回测 (启用滑点模拟)
    # initial_capital=10000, fee=0.0005, slippage=0.0005 (万5滑点)
    engine = BacktestEngine(initial_capital=10000, fee_rate=0.0005, slippage=0.0005)
    result = engine.run(df)

    # 5. 只有表现尚可的才画图，避免弹窗太多
    if result["sharpe_ratio"] > 1.0:
        engine.plot(result, title=f"{symbol} {timeframe} - {strategy_name}")

    return {
        "Strategy": strategy_name,
        "Symbol": symbol,
        "TF": timeframe,
        "Return": result["total_ret"],
        "Annual": result["annual_ret"],  # 新增指标
        "MaxDD": result["max_drawdown"],
        "Sharpe": result["sharpe_ratio"],
        "Calmar": result["calmar_ratio"],  # 新增指标
        "Trades": result["trade_count"],
    }


def main():
    print("🚀 量化实验室启动...")

    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)

    summary = []

    # === 实验配置 ===

    # 1. 趋势策略组 (Trend Following)
    # 适合: BTC, ETH 等大市值币种，4h/1d 级别
    trend_tasks = [
        {"symbol": "BTC/USDT", "tf": "4h"},
        {"symbol": "ETH/USDT", "tf": "4h"},
        {"symbol": "SOL/USDT", "tf": "4h"},
    ]

    # 2. 均值回归策略组 (Mean Reversion)
    # 适合: 震荡行情，或者山寨币，1h/15m 级别
    # 注意：DOGE 这种波动率大的币种，布林带策略往往有奇效
    mr_tasks = [
        {"symbol": "BTC/USDT", "tf": "1h"},  # 对比 BTC 在 1h 的震荡表现
        {"symbol": "DOGE/USDT", "tf": "1h"},
        {"symbol": "XRP/USDT", "tf": "1h"},
    ]

    # === 开始运行 ===

    # A. 跑双均线 (DualMA)
    print("\n>>> 正在执行：双均线趋势策略 (DualMA)...")
    for task in trend_tasks:
        res = run_strategy_backtest(
            symbol=task["symbol"],
            timeframe=task["tf"],
            strategy_cls=DualMAStrategy,
            strategy_params={
                "fast_period": 10,
                "slow_period": 60,
                "atr_multiplier": 5.0,
            },
            data_handler=data_handler,
        )
        summary.append(res)

    # B. 跑布林回归 (BollingerMR)
    print("\n>>> 正在执行：布林均值回归策略 (BollingerMR)...")
    for task in mr_tasks:
        res = run_strategy_backtest(
            symbol=task["symbol"],
            timeframe=task["tf"],
            strategy_cls=BollingerMRStrategy,
            # 参数：20周期，2倍标准差开仓，0.5倍平仓，4倍强制止损
            strategy_params={
                "window": 20,
                "entry_std": 2.5,
                "exit_std": 0.5,
                "stop_loss_std": 4.0,
            },
            data_handler=data_handler,
        )
        summary.append(res)

    # === 最终汇总报告 ===
    print("\n\n🏆 ================ 策略竞技场结果 ================")
    if not summary:
        print("没有生成回测结果。")
        return

    df_summary = pd.DataFrame(summary)

    # 格式化输出，使其易读
    format_dict = {
        "Return": "{:.2%}",
        "Annual": "{:.2%}",
        "MaxDD": "{:.2%}",
        "Sharpe": "{:.2f}",
        "Calmar": "{:.2f}",
    }

    # [修复] 显式应用格式化字典
    # 注意：这会将数值列转换为字符串，仅用于打印展示
    for col, fmt in format_dict.items():
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(lambda x: fmt.format(x))

    print(df_summary.to_markdown(index=False))

    print("\n💡 分析提示:")
    print("1. 如果[自适应]显著优于[通用]，说明该策略有效，但需要针对该币种单独调参。")
    print(
        "2. 如果[自适应]依然亏损或收益微薄，说明该策略逻辑(双均线)根本不适合该币种，建议放弃。"
    )


def main():
    print("🚀 初始化量化系统...")
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)

    # 运行研究实验
    run_research_experiment(data_handler)


if __name__ == "__main__":
    main()
