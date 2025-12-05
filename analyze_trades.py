import ccxt
import pandas as pd

from config.settings import GLOBAL_CONFIG
from src.data_loader import MarketDataHandler
from src.strategies.dual_ma import DualMAStrategy


def analyze_symbol_details(symbol, params):
    print(f"\n🔬 正在深度分析: {symbol} ...")

    # 1. 获取数据
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)
    df = data_handler.prepare_data(symbol, "4h", days_back=730)

    # 2. 运行策略并获取流水
    strategy = DualMAStrategy(
        fast_period=params["fast"],
        slow_period=params["slow"],
        atr_multiplier=params["atr_mult"],
    )

    trades = strategy.get_trade_history(df)

    if not trades:
        print("❌ 无交易记录")
        return

    # 3. 转换为 DataFrame
    df_trades = pd.DataFrame(trades)

    # 4. 计算核心指标
    total_trades = len(df_trades)
    win_trades = df_trades[df_trades["pnl"] > 0]
    loss_trades = df_trades[df_trades["pnl"] <= 0]

    win_rate = len(win_trades) / total_trades
    avg_win = win_trades["pnl"].mean()
    avg_loss = loss_trades["pnl"].mean()
    pnl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 扣除手续费后的净盈亏
    df_trades["net_pnl"] = df_trades["pnl"] - 0.001  # 假设买卖各万5

    # 5. 打印分析报告
    print(f"\n📊 {symbol} 交易行为分析报告")
    print("-----------------------------")
    print(f"总交易次数: {total_trades}")
    print(f"胜率 (Win Rate): {win_rate * 100:.2f}%")
    print(f"盈亏比 (P/L Ratio): {pnl_ratio:.2f}")
    print("-----------------------------")
    print(f"平均盈利: {avg_win * 100:.2f}%")
    print(f"平均亏损: {avg_loss * 100:.2f}%")
    print(f"最大单笔盈利: {df_trades['pnl'].max() * 100:.2f}%")
    print(f"最大单笔亏损: {df_trades['pnl'].min() * 100:.2f}%")
    print(
        f"平均持仓K线数: {df_trades['bars_held'].mean():.1f} 根 (约 {df_trades['bars_held'].mean() * 4:.1f} 小时)"
    )
    print(
        f"止损触发比例: {len(df_trades[df_trades['reason'] == 'StopLoss']) / total_trades * 100:.1f}%"
    )

    # 6. 保存 CSV
    safe_sym = symbol.replace("/", "_")
    csv_path = f"data/results/{safe_sym}_trades.csv"
    df_trades.to_csv(csv_path, index=False)
    print(f"\n💾 详细流水已保存至: {csv_path}")
    print("👉 建议用 Excel 打开 CSV，观察那些亏损单是怎么发生的。")


if __name__ == "__main__":
    # 使用之前跑出来的 DOGE 最佳参数
    doge_params = {"fast": 8, "slow": 25, "atr_mult": 7.0}
    analyze_symbol_details("DOGE/USDT", doge_params)

    # 也可以对比看看 ETH 的惨状
    # eth_params = {'fast': 8, 'slow': 20, 'atr_mult': 7.0}
    # analyze_symbol_details("ETH/USDT", eth_params)
