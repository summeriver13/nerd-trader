"""
src/research/analysis/deep_dive.py
单币种深度分析脚本
"""

import sys
from pathlib import Path

# [修改 1] 标准库和第三方库尽量放在最前面
import ccxt
import pandas as pd

# 1. 确保能导入项目根目录
# 解释：__file__ 是当前文件，.parent 是 analysis，.parents[1] 是 research，.parents[2] 是 src，.parents[3] 是 cryptotrader
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# [修改 2] 项目内部模块必须在 sys.path 修改后导入，所以加上 # noqa: E402 忽略警告
from config.settings import GLOBAL_CONFIG  # noqa: E402

# 引用位于 research 下的向量化策略
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402


def analyze_symbol_details(symbol, params):
    print(f"\n🔬 正在深度分析: {symbol} ...")

    # 1. 获取数据
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    data_handler = MarketDataHandler(exchange)

    # 获取两年的数据
    df = data_handler.prepare_data(symbol, "4h", days_back=730)

    # 2. 运行策略 (使用向量化版本)
    strategy = DualMAStrategy(
        fast_period=params["fast"],
        slow_period=params["slow"],
        atr_multiplier=params["atr_mult"],
    )

    # 这里现在可以正常调用了，因为 vector_dual_ma.py 里有这个方法
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

    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    avg_win = win_trades["pnl"].mean() if not win_trades.empty else 0
    avg_loss = loss_trades["pnl"].mean() if not loss_trades.empty else 0
    pnl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 扣除手续费后的净盈亏 (近似计算)
    df_trades["net_pnl"] = df_trades["pnl"] - 0.001

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

    if "bars_held" in df_trades.columns:
        print(f"平均持仓K线数: {df_trades['bars_held'].mean():.1f} 根")

    if "reason" in df_trades.columns:
        stop_loss_count = len(df_trades[df_trades["reason"] == "StopLoss"])
        print(f"止损触发比例: {stop_loss_count / total_trades * 100:.1f}%")

    # 6. 保存 CSV
    # 保存到 data/research_results
    output_dir = PROJECT_ROOT / "data" / "research_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_sym = symbol.replace("/", "_")
    csv_path = output_dir / f"{safe_sym}_trades.csv"

    df_trades.to_csv(csv_path, index=False)
    print(f"\n💾 详细流水已保存至: {csv_path}")
    print("👉 建议用 Excel 打开 CSV，观察那些亏损单是怎么发生的。")


if __name__ == "__main__":
    # 使用之前跑出来的 DOGE 最佳参数
    doge_params = {"fast": 8, "slow": 25, "atr_mult": 7.0}

    try:
        analyze_symbol_details("DOGE/USDT", doge_params)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print(
            "提示: 请确保你已经创建了 src/research/strategies/vector_dual_ma.py 并粘贴了旧版代码。"
        )
