import matplotlib

matplotlib.use("Agg")  # 保持后台绘图模式，适合服务器环境
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
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,  # 手续费万5
        slippage: float = 0.0005,  # [优化] 新增滑点万5 (双边交易成本 = fee + slippage)
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.results_dir = os.path.join("data", "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self, df: pd.DataFrame) -> dict:
        """
        运行向量化回测
        :param df: 必须包含 'close', 'signal' 列。
                   signal: 1(做多), -1(做空), 0(空仓)
        """
        if df.empty:
            print("⚠️ [回测引擎] 数据为空，跳过回测。")
            return {}

        print(f"\n⚡ [回测引擎] 开始回测 ({len(df)} bars)...")
        self._check_memory(df)

        df = df.copy()

        # -----------------------------------------------------------
        # 1. 核心交易逻辑 (Vectorized)
        # -----------------------------------------------------------

        # 计算持仓: 信号产生的下一根K线生效 (避免未来函数)
        # fillna(0) 假设初始状态为空仓
        df["position"] = df["signal"].shift(1).fillna(0)

        # 计算基准收益 (Buy & Hold)
        # pct_change 默认计算的是 Close-to-Close 收益
        df["market_ret"] = df["close"].pct_change().fillna(0)

        # 计算策略原始收益
        # 支持做空 (position=-1) 和 杠杆 (position=2.0)
        df["strategy_raw_ret"] = df["market_ret"] * df["position"]

        # [优化] 计算交易成本 (手续费 + 滑点)
        # diff() 计算仓位变化量，abs() 取绝对值
        # 例如: 0 -> 1 (开多) diff=1;  1 -> -1 (反手) diff=2
        df["trades"] = df["position"].diff().abs().fillna(0)

        # 总费率 = 手续费 + 滑点
        total_cost_rate = self.fee_rate + self.slippage
        df["cost"] = df["trades"] * total_cost_rate

        # 净收益 = 原始收益 - 成本
        df["strategy_net_ret"] = df["strategy_raw_ret"] - df["cost"]

        # -----------------------------------------------------------
        # 2. 资金曲线与回撤
        # -----------------------------------------------------------

        # 策略资金曲线
        df["equity_curve"] = (
            1 + df["strategy_net_ret"]
        ).cumprod() * self.initial_capital

        # 基准资金曲线
        df["benchmark_curve"] = (1 + df["market_ret"]).cumprod() * self.initial_capital

        # 策略最大回撤计算
        df["peak"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak"]) / df["peak"]
        max_drawdown = df["drawdown"].min()  # 这是一个负数，例如 -0.25

        # 基准最大回撤
        bench_peak = df["benchmark_curve"].cummax()
        bench_drawdown = (df["benchmark_curve"] - bench_peak) / bench_peak
        bench_max_drawdown = bench_drawdown.min()

        # -----------------------------------------------------------
        # 3. [优化] 高级指标统计
        # -----------------------------------------------------------

        # 总收益率
        total_ret = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1
        bench_ret = (df["benchmark_curve"].iloc[-1] / self.initial_capital) - 1

        # [新增] 年化收益率 (CAGR)
        # 假设 df.index 是 datetime 类型
        time_span_days = (df.index[-1] - df.index[0]).days
        if time_span_days > 0:
            # 公式: (1 + total_ret) ^ (365 / days) - 1
            annual_ret = (1 + total_ret) ** (365 / time_span_days) - 1
        else:
            annual_ret = 0.0

        # [新增] 卡尔马比率 (Calmar Ratio) = 年化收益 / 最大回撤
        # 衡量承受每单位最大回撤能换来的年化收益
        if max_drawdown != 0:
            calmar_ratio = annual_ret / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # 夏普比率 (Sharpe Ratio)
        # 假设是 4h 数据 (一天6根 * 365天 = 2190) -> 年化因子 sqrt(2190)
        # 如果是 1d 数据 -> sqrt(365)
        # 这里做个简单的自适应
        # 简单推断周期 (取前两个时间戳差值)
        try:
            delta_seconds = (df.index[1] - df.index[0]).total_seconds()
            annual_factor = np.sqrt(365 * 24 * 3600 / delta_seconds)
        except Exception:
            annual_factor = np.sqrt(365)  # 默认日线

        volatility = df["strategy_net_ret"].std()
        if volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (df["strategy_net_ret"].mean() / volatility) * annual_factor

        # [新增] 盈亏比 (Profit Factor)
        # 简单算法：总盈利 / 总亏损绝对值
        gross_profit = df[df["strategy_net_ret"] > 0]["strategy_net_ret"].sum()
        gross_loss = abs(df[df["strategy_net_ret"] < 0]["strategy_net_ret"].sum())

        if gross_loss == 0:
            profit_factor = float("inf")
        else:
            profit_factor = gross_profit / gross_loss

        # [新增] 胜率 (Win Rate) - 基于K线周期 (Bar-by-Bar)
        # 注意：向量化回测很难精确统计“每笔交易”的胜率，除非加循环。
        # 这里统计的是“持仓期间盈利的K线比例”，供参考。
        active_bars = df[df["position"] != 0]
        if len(active_bars) > 0:
            win_rate = len(active_bars[active_bars["strategy_net_ret"] > 0]) / len(
                active_bars
            )
        else:
            win_rate = 0.0

        trade_count = df["trades"].sum()

        # -----------------------------------------------------------
        # 4. 打印报告
        # -----------------------------------------------------------
        print("======== 📊 深度回测报告 ========")
        print(
            f"💰 初始/最终:   {self.initial_capital:.0f} -> {df['equity_curve'].iloc[-1]:.2f}"
        )
        print(f"🚀 总收益率:    {total_ret * 100:.2f}% (基准: {bench_ret * 100:.2f}%)")
        print(f"📅 年化收益:    {annual_ret * 100:.2f}%")
        print(
            f"📉 最大回撤:    {max_drawdown * 100:.2f}% (基准: {bench_max_drawdown * 100:.2f}%)"
        )
        print(f"🛡️ 卡尔马比:    {calmar_ratio:.2f} (收益回撤比，>2 优秀)")
        print(f"📊 夏普比率:    {sharpe_ratio:.2f} (>1 良好)")
        print(f"⚖️ 盈亏比(PF):  {profit_factor:.2f} (>1.5 优秀)")
        print(f"🎲 周期胜率:    {win_rate * 100:.1f}%")
        print(f"🔄 交易次数:    {int(trade_count)} (换手次数)")
        print("=================================")

        return {
            "df": df,
            "total_ret": total_ret,
            "annual_ret": annual_ret,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "trade_count": trade_count,
        }

    def plot(self, result: dict, title: str = "Backtest Result"):
        df = result["df"]
        max_dd = result.get("max_drawdown", 0)

        # 使用 matplotlib 的样式，让图表更专业
        plt.style.use("bmh")

        fig = plt.figure(figsize=(14, 12))
        # 调整子图布局: 3行1列，高度比例 3:1:1
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])

        # 子图1: 资金曲线
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            df.index,
            df["equity_curve"],
            label="Strategy Equity",
            color="#d62728",
            linewidth=1.5,
        )
        ax1.plot(
            df.index,
            df["benchmark_curve"],
            label="Benchmark (Buy&Hold)",
            color="gray",
            linestyle="--",
            alpha=0.6,
        )
        ax1.set_title(
            f"{title}\nTotal Ret: {result['total_ret'] * 100:.1f}% | MaxDD: {max_dd * 100:.1f}% | Sharpe: {result['sharpe_ratio']:.2f}"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # 子图2: 价格与均线 (如果有)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df["close"], label="Price", color="black", alpha=0.3)
        if "MA_Fast" in df.columns:
            ax2.plot(
                df.index, df["MA_Fast"], color="#1f77b4", linewidth=1, label="MA Fast"
            )
        if "MA_Slow" in df.columns:
            ax2.plot(
                df.index, df["MA_Slow"], color="#ff7f0e", linewidth=1, label="MA Slow"
            )
        ax2.set_ylabel("Price")
        ax2.legend(loc="upper left", fontsize="small")
        ax2.grid(True)

        # 子图3: 水下回撤图 (Underwater Plot)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.fill_between(df.index, df["drawdown"], 0, color="red", alpha=0.3)
        ax3.plot(df.index, df["drawdown"], color="red", linewidth=0.5)
        ax3.set_ylabel("Drawdown")
        ax3.set_ylim(bottom=df["drawdown"].min() * 1.1, top=0.01)  # 动态调整Y轴
        ax3.grid(True)

        # 自动生成文件名
        filename = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(self.results_dir, filename)

        plt.tight_layout()
        plt.savefig(path, dpi=150)  # 提高分辨率
        print(f"📊 策略图表已保存: {path}")
        plt.close()

    def _check_memory(self, df):
        if psutil:
            mem = psutil.virtual_memory()
            # 简单的内存警告
            if mem.percent > 85:
                print(f"⚠️ 警告: 系统内存占用过高 ({mem.percent}%)")
