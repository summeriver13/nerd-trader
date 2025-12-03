import ccxt
import pandas as pd

from config.settings import GLOBAL_CONFIG
from src.backtest.engine import BacktestEngine
from src.data_loader import MarketDataHandler
from src.strategies.dual_ma import DualMAStrategy


def run_strategy_for_symbol(symbol: str, data_handler: MarketDataHandler):
    print(f"\n{'=' * 40}")
    print(f"🪙 正在分析: {symbol}")
    print(f"{'=' * 40}")

    # 1. 准备数据
    df = data_handler.prepare_data(symbol=symbol, timeframe="4h", days_back=730)

    # 2. 策略实例
    # 这里我们使用 BTC 跑出来的“黄金参数” (9, 30, 6.0) 去验证其他币
    # 这叫 "Out-of-sample Testing" (样本外测试)
    strategy = DualMAStrategy(fast_period=9, slow_period=30, atr_multiplier=6.0)

    # 3. 计算信号
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    # 4. 运行回测
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(df)

    # 5. 画图
    engine.plot(result, title=f"{symbol} Dual MA (9,30)")

    return {
        "symbol": symbol,
        "return": result["total_ret"],
        "max_drawdown": result["max_drawdown"],
        "sharpe": result["sharpe_ratio"],
        "trades": result["trade_count"],
    }


def main():
    print("🚀 初始化量化系统 (多品种验证模式)...")

    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)

    # 定义要测试的币种列表
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]

    summary = []

    for symbol in symbols:
        try:
            res = run_strategy_for_symbol(symbol, data_handler)
            summary.append(res)
        except Exception as e:
            print(f"❌ {symbol} 回测失败: {e}")

    # 打印最终汇总
    print("\n\n🏆 ======== 全市场验证报告 ========")
    df_summary = pd.DataFrame(summary)
    # 格式化显示
    df_summary["return"] = df_summary["return"].apply(lambda x: f"{x * 100:.2f}%")
    df_summary["max_drawdown"] = df_summary["max_drawdown"].apply(
        lambda x: f"{x * 100:.2f}%"
    )
    df_summary["sharpe"] = df_summary["sharpe"].apply(lambda x: f"{x:.2f}")

    print(df_summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
