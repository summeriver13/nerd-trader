"""
src/research/run_experiments.py
量化策略实验室 - 批量回测入口
"""

import sys
from pathlib import Path
from typing import Type

import ccxt
import pandas as pd

# 1. 路径补丁 (必须在导入本地模块之前)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to cryptotrader/
sys.path.append(str(PROJECT_ROOT))

# 2. 本地模块导入 (使用 # noqa: E402 忽略 "导入位置不在文件头" 的警告)
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402
from src.research.strategies.base import BaseStrategy  # noqa: E402
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402

# noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402


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
    days = 365 if timeframe == "1h" else 730
    df = data_handler.prepare_data(symbol=symbol, timeframe=timeframe, days_back=days)

    # 2. 初始化策略
    strategy = strategy_cls(**strategy_params)

    # 3. 计算信号
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    # 4. 运行回测
    engine = BacktestEngine(initial_capital=10000, fee_rate=0.0005, slippage=0.0005)
    result = engine.run(df)

    # 5. 只有表现尚可的才画图
    if result["sharpe_ratio"] > 1.0:
        # 图表保存到 data/results
        save_path = (
            PROJECT_ROOT
            / f"data/results/{symbol.replace('/', '_')}_{strategy_name}.png"
        )
        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        engine.plot(result, title=f"{symbol} {timeframe}", save_path=str(save_path))

    return {
        "Strategy": strategy_name,
        "Symbol": symbol,
        "TF": timeframe,
        "Return": result["total_ret"],
        "Annual": result["annual_ret"],
        "MaxDD": result["max_drawdown"],
        "Sharpe": result["sharpe_ratio"],
        "Calmar": result["calmar_ratio"],
        "Trades": result["trade_count"],
    }


def main():
    print("🚀 量化实验室启动...")

    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)

    summary = []

    # === 实验配置 ===
    trend_tasks = [
        {"symbol": "BTC/USDT", "tf": "4h"},
        {"symbol": "ETH/USDT", "tf": "4h"},
    ]

    # === 开始运行 ===

    # A. 跑双均线 (DualMA)
    print("\n>>> 正在执行：双均线趋势策略 (DualMA)...")
    for task in trend_tasks:
        res = run_strategy_backtest(
            symbol=task["symbol"],
            timeframe=task["tf"],
            strategy_cls=DualMAStrategy,  # 使用向量版类
            strategy_params={
                "fast_period": 10,
                "slow_period": 60,
                "atr_multiplier": 5.0,
            },
            data_handler=data_handler,
        )
        summary.append(res)

    # === 最终汇总报告 ===
    print("\n\n🏆 ================ 策略竞技场结果 ================")
    if not summary:
        return

    df_summary = pd.DataFrame(summary)

    format_dict = {
        "Return": "{:.2%}",
        "Annual": "{:.2%}",
        "MaxDD": "{:.2%}",
        "Sharpe": "{:.2f}",
        "Calmar": "{:.2f}",
    }

    for col, fmt in format_dict.items():
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(lambda x: fmt.format(x))

    try:
        print(df_summary.to_markdown(index=False))
    except ImportError:
        print(df_summary)


if __name__ == "__main__":
    main()
