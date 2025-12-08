"""
src/research/engine.py
向量化回测引擎 (Vectorized Backtest Engine)
职责：基于 Pandas dataframe 进行快速的资金曲线回测、绩效统计和绘图。
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # [新增]
from plotly.subplots import make_subplots  # [新增]

# 年化系数 (加密货币通常按 365 天计算)
ANNUAL_DAYS = 365


class BacktestEngine:
    """
    轻量级向量化回测引擎
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005,  # 万5手续费
        slippage: float = 0.0005,  # 万5滑点
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        执行回测
        :param df: 必须包含 'close' 和 'signal' 列的 DataFrame
                   signal: 1(持多), 0(空仓), -1(持空)
        """
        if df.empty:
            print("⚠️ [回测引擎] 数据为空，跳过回测。")
            return {}

        # 1. 检查必要列
        if "close" not in df.columns or "signal" not in df.columns:
            raise ValueError("Dataframe must contain 'close' and 'signal' columns")

        df = df.copy()

        # 2. 计算持仓 (Position)
        # 信号产生的下一根K线生效 (避免未来函数)
        # fillna(0) 假设初始状态为空仓
        df["pos"] = df["signal"].shift(1).fillna(0)

        # 3. 计算持仓变化 (Trades)
        # diff() 计算仓位变化量，abs() 取绝对值
        # 例如: 0 -> 1 (开多) diff=1;  1 -> -1 (反手) diff=2
        df["pos_change"] = df["pos"].diff().abs().fillna(0)

        # 4. 计算基准收益 (Market Return)
        # pct_change 默认计算的是 Close-to-Close 收益
        df["pct_change"] = df["close"].pct_change().fillna(0)

        # 5. 计算策略收益 (Strategy Return)
        # 策略收益 = 持仓方向 * 市场涨跌幅
        df["strategy_raw_ret"] = df["pos"] * df["pct_change"]

        # 6. 计算交易成本 (Cost)
        # 总费率 = 手续费 + 滑点
        total_cost_rate = self.fee_rate + self.slippage
        df["cost"] = df["pos_change"] * total_cost_rate

        # 7. 计算净收益 (Net Return)
        df["net_ret"] = df["strategy_raw_ret"] - df["cost"]

        # 8. 计算资金曲线 (Equity Curve)
        # 使用累乘计算复利
        df["equity_curve"] = (1 + df["net_ret"]).cumprod() * self.initial_capital
        df["benchmark_curve"] = (1 + df["pct_change"]).cumprod() * self.initial_capital

        # 9. 计算回撤 (Drawdown)
        df["peak"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak"]) / df["peak"]

        return self._calculate_metrics(df)

    def _calculate_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """计算核心绩效指标"""
        # 总收益率
        total_ret = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1

        # [新增] 基准收益率
        bench_ret = (df["benchmark_curve"].iloc[-1] / self.initial_capital) - 1

        # 时间跨度 (天)
        days = (df.index[-1] - df.index[0]).days
        if days <= 0:
            days = 1

        # 年化收益率 (CAGR)
        annual_ret = (1 + total_ret) ** (ANNUAL_DAYS / days) - 1

        # 最大回撤
        max_drawdown = df["drawdown"].min()

        # 夏普比率 (Sharpe Ratio)
        # 简单估算：Sharpe = Mean / Std * sqrt(N)
        # 这里的 N 取决于数据频率，为了通用性，我们基于数据长度倒推每年的Bar数
        bars_per_year = len(df) / (days / ANNUAL_DAYS)
        volatility = df["net_ret"].std()

        if volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (df["net_ret"].mean() / volatility) * np.sqrt(bars_per_year)

        # 卡尔玛比率 (Calmar Ratio)
        calmar_ratio = 0.0
        if max_drawdown != 0:
            calmar_ratio = annual_ret / abs(max_drawdown)

        # 交易次数
        trade_count = df["pos_change"].sum()

        return {
            "total_ret": total_ret,
            "bench_ret": bench_ret,
            "annual_ret": annual_ret,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "trade_count": trade_count,
            "equity_curve": df["equity_curve"],  # 返回 Series 供画图
            "benchmark_curve": df["benchmark_curve"],
            "drawdown_curve": df["drawdown"],
            "df": df,  # 返回完整数据的引用
        }

    def plot(
        self,
        result: dict,
        title: str = "Backtest Result",
        save_path: Optional[str] = None,
    ):
        """
        绘制回测图表 (Matplotlib 静态图)
        :param save_path: 如果提供路径，则保存图片而不显示
        """
        equity = result["equity_curve"]
        benchmark = result["benchmark_curve"]
        drawdown = result["drawdown_curve"]

        # 使用 matplotlib 的样式
        plt.style.use("bmh")

        fig = plt.figure(figsize=(14, 10))
        # 3行1列，高度比例 3:1
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

        # 子图1：资金曲线对比
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity.index, equity, label="Strategy", color="#d62728", linewidth=1.5)
        ax1.plot(
            benchmark.index,
            benchmark,
            label="Benchmark",
            color="gray",
            linestyle="--",
            alpha=0.6,
        )

        ax1.set_title(
            f"{title}\n"
            f"Total Return: {result['total_ret']:.2%} | "
            f"Sharpe: {result['sharpe_ratio']:.2f} | "
            f"MaxDD: {result['max_drawdown']:.2%}"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_ylabel("Equity")

        # 子图2：水下回撤图 (Underwater Plot)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color="red", linewidth=0.5)
        ax2.set_ylabel("Drawdown")
        ax2.set_ylim(bottom=min(drawdown.min() * 1.1, -0.01), top=0.01)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"📊 策略图表已保存: {save_path}")
            plt.close()  # 关闭画布，释放内存
        else:
            plt.show()

    def plot_interactive(
        self,
        result: dict,
        title: str = "Backtest Result",
        save_path: Optional[str] = None,
    ):
        """
        生成交互式 HTML 研报 (Plotly)
        """
        df = result["df"]
        equity = result["equity_curve"]
        benchmark = result["benchmark_curve"]
        drawdown = result["drawdown_curve"]

        # 创建子图结构: 3行1列 (资金、价格、回撤)
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("资金曲线 (Equity)", "标的价格 (Price)", "回撤 (Drawdown)"),
            row_heights=[0.4, 0.4, 0.2],
        )

        # 1. 资金曲线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=equity,
                name="Strategy Equity",
                line=dict(color="#ffa726", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=benchmark,
                name="Benchmark",
                line=dict(color="gray", dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 2. 价格 & 均线 (如果有)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                name="Price",
                line=dict(color="#29b6f6", width=1),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )
        if "MA_Fast" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA_Fast"],
                    name="MA Fast",
                    line=dict(color="#ef5350", width=1),
                ),
                row=2,
                col=1,
            )
        if "MA_Slow" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA_Slow"],
                    name="MA Slow",
                    line=dict(color="#66bb6a", width=1),
                ),
                row=2,
                col=1,
            )

        # 3. 回撤区域
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#ef5350", width=1),
            ),
            row=3,
            col=1,
        )

        # 布局美化
        fig.update_layout(
            title=f"{title} | Sharpe: {result['sharpe_ratio']:.2f} | MaxDD: {result['max_drawdown']:.2%}",
            template="plotly_dark",  # 酷炫的暗色主题
            hovermode="x unified",  # 鼠标悬停显示所有数据
            height=1000,
        )

        if save_path:
            # 自动补全 .html 后缀
            if not save_path.endswith(".html"):
                save_path += ".html"
            fig.write_html(save_path)
            print(f"📊 交互式研报已生成: {save_path}")
        else:
            fig.show()
