import matplotlib

matplotlib.use("Agg")
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.0005):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.results_dir = os.path.join("data", "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self, df: pd.DataFrame) -> dict:
        """
        运行回测
        :param df: 必须包含 'close', 'signal' 列
        """
        print(f"\n⚡ [回测引擎] 开始回测 ({len(df)} 行)...")
        self._check_memory(df)

        df = df.copy()

        # 1. 计算持仓 (Signal -> Position)
        # Shift(1) 代表信号产生的下一根K线开盘成交
        df["position"] = df["signal"].shift(1).fillna(0)

        # 2. 计算基准收益 (Market)
        df["market_ret"] = df["close"].pct_change()

        # 3. 计算策略收益 (Strategy)
        df["strategy_raw_ret"] = df["market_ret"] * df["position"]

        # 4. 计算交易成本
        df["trades"] = df["position"].diff().abs().fillna(0)
        df["cost"] = df["trades"] * self.fee_rate

        # 5. 净收益
        df["strategy_net_ret"] = df["strategy_raw_ret"] - df["cost"]

        # 6. 资金曲线
        df["equity_curve"] = (
            1 + df["strategy_net_ret"]
        ).cumprod() * self.initial_capital
        df["benchmark_curve"] = (1 + df["market_ret"]).cumprod() * self.initial_capital

        # === [新增] 高级指标计算 ===

        # 1. 策略最大回撤 (Max Drawdown)
        # 计算历史最高净值
        df["peak"] = df["equity_curve"].cummax()
        # 计算当前回撤百分比: (当前净值 - 历史最高) / 历史最高
        df["drawdown"] = (df["equity_curve"] - df["peak"]) / df["peak"]
        # 取最小值即为最大回撤
        max_drawdown = df["drawdown"].min()

        # 2. [新增] 基准最大回撤 (用于对比风险)
        bench_peak = df["benchmark_curve"].cummax()
        bench_drawdown = (df["benchmark_curve"] - bench_peak) / bench_peak
        bench_max_drawdown = bench_drawdown.min()

        # 3. 收益率
        total_ret = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1
        bench_ret = (df["benchmark_curve"].iloc[-1] / self.initial_capital) - 1

        # 4. 交易统计
        trade_count = df["trades"].sum()

        # 5. 夏普比率 (简化版，假设无风险利率为0)
        # 需将年化因子根据周期调整
        # 假设是 4h 数据 (一天6根K线，一年365天 = 2190根)
        annual_factor = np.sqrt(365 * 6)

        volatility = df["strategy_net_ret"].std()
        if volatility == 0:
            sharpe_ratio = 0
        else:
            # 年化夏普
            sharpe_ratio = (df["strategy_net_ret"].mean() / volatility) * annual_factor

        print("======== 回测深度报告 ========")
        print(f"💰 初始本金:   {self.initial_capital}")
        print(f"📈 最终净值:   {df['equity_curve'].iloc[-1]:.2f}")
        print(f"🚀 策略收益:   {total_ret * 100:.2f}%")
        print(f"🐢 基准收益:   {bench_ret * 100:.2f}%")
        print(
            f"📉 最大回撤:   {max_drawdown * 100:.2f}% (vs 基准 {bench_max_drawdown * 100:.2f}%)"
        )
        print(f"📊 夏普比率:   {sharpe_ratio:.2f} (越高越好, >1优秀, >2顶尖)")
        print(f"🔄 交易次数:   {int(trade_count)}")
        print("==============================")

        return {
            "df": df,
            "total_ret": total_ret,
            "max_drawdown": max_drawdown,
            "trade_count": trade_count,
            "sharpe_ratio": sharpe_ratio,
        }

    def plot(self, result: dict, title: str = "Backtest Result"):
        df = result["df"]
        max_dd = result.get("max_drawdown", 0)

        plt.figure(figsize=(12, 10))  # 加高一点图表

        # 子图1: 资金曲线
        ax1 = plt.subplot(3, 1, 1)  # 改为3个子图
        ax1.plot(df.index, df["equity_curve"], label="Strategy", color="red")
        ax1.plot(
            df.index,
            df["benchmark_curve"],
            label="Benchmark",
            color="gray",
            linestyle="--",
        )
        ax1.set_title(
            f"{title} (Return: {result['total_ret'] * 100:.1f}%, MDD: {max_dd * 100:.1f}%)"
        )
        ax1.legend()
        ax1.grid(True)

        # 子图2: 价格与均线
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, df["close"], label="Price", alpha=0.5)
        if "MA_Fast" in df.columns:
            ax2.plot(df.index, df["MA_Fast"], color="orange", lw=1)
        if "MA_Slow" in df.columns:
            ax2.plot(df.index, df["MA_Slow"], color="blue", lw=1)
        ax2.legend(["Price", "Fast", "Slow"])
        ax2.grid(True)

        # 子图3: 回撤曲线 (新增)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.fill_between(df.index, df["drawdown"], 0, color="red", alpha=0.3)
        ax3.set_title("Drawdown")
        ax3.grid(True)

        filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        print(f"📊 图表已保存: {path}")
        plt.close()

    def _check_memory(self, df):
        if psutil:
            mem = psutil.virtual_memory()
            print(f"   系统可用内存: {mem.available / (1024**3):.1f} GB")
