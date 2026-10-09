from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 引入我们刚才写的 Domain
from src.research.domains.base import MarketDomain

# 年化系数
ANNUAL_DAYS = 365


class BacktestEngine:
    """
    轻量级向量化回测引擎 (支持多市场 Domain + 杠杆)
    """

    def __init__(
        self,
        domain: MarketDomain,  # 注入市场规则
        initial_capital: float = 10000.0,
        leverage: float = 1.0,
    ):
        self.domain = domain
        self.initial_capital = initial_capital
        self.leverage = leverage

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        执行回测
        :param df: 必须包含 'close' 和 'signal' 列的 DataFrame
        """
        if df.empty:
            print("⚠️ [回测引擎] 数据为空，跳过回测。")
            return {}

        if "close" not in df.columns or "signal" not in df.columns:
            raise ValueError("Dataframe must contain 'close' and 'signal' columns")

        df = df.copy()

        # 1. 信号预处理 (由 Domain 处理)
        # 使用 cast 强制告诉检查器：df["signal"] 取出来的一定是 Series
        raw_signal = cast(pd.Series, df["signal"])
        df["signal"] = self.domain.adjust_signal(raw_signal)

        # 2. 计算持仓 (Position)
        # 信号产生的下一根K线生效
        df["pos"] = df["signal"].shift(1).fillna(0)

        # 3. 计算持仓变化 (Trades)
        # 原始变化量 (用于判断方向)
        raw_pos_change = df["pos"].diff().fillna(0)
        df["pos_change"] = raw_pos_change.abs()

        # 4. 计算基准收益 (Market Return)
        df["pct_change"] = df["close"].pct_change().fillna(0)

        # 5. 计算策略收益 (Strategy Return)
        # 策略收益 = 持仓 * 市场涨跌幅 * 杠杆倍数
        df["strategy_raw_ret"] = df["pos"] * df["pct_change"] * self.leverage

        # 6. 计算交易成本 (Cost) - 委托 Domain 计算
        # ------------------------------------------------------------------
        # 6.1 计算成交名义金额 (Notional Value)
        # 简化模型：基于初始本金估算
        trade_notional = df["pos_change"] * self.initial_capital * self.leverage

        # 6.2 识别卖出行为 (用于印花税)
        is_sell_mask = raw_pos_change < 0

        # 6.3 调用 Domain 获取费用详情 (绝对金额)
        cost_obj = self.domain.calculate_cost(trade_notional, is_sell_mask)

        # 6.4 将绝对费用转换为收益率扣减
        # Cost Rate = 总费用 / 初始本金
        df["cost"] = cost_obj.total / self.initial_capital
        # ------------------------------------------------------------------

        # 7. 计算净收益 (Net Return)
        df["net_ret"] = df["strategy_raw_ret"] - df["cost"]

        # 8. 计算资金曲线 (Equity Curve)
        df["equity_curve"] = (1 + df["net_ret"]).cumprod() * self.initial_capital

        # === 爆仓检测 ===
        if (df["equity_curve"] <= 0).any():
            bust_idx = df[df["equity_curve"] <= 0].index[0]
            print(f"💀 [严重警告] 策略在 {bust_idx} 发生爆仓(归零)！")
            df.loc[bust_idx:, "equity_curve"] = 0
            df.loc[bust_idx:, "net_ret"] = 0

        df["benchmark_curve"] = (1 + df["pct_change"]).cumprod() * self.initial_capital

        # 9. 计算回撤 (Drawdown)
        df["peak"] = df["equity_curve"].cummax()
        df["drawdown"] = np.where(
            df["peak"] > 0,
            (df["equity_curve"] - df["peak"]) / df["peak"],
            -1.0,
        )
        df["drawdown"] = pd.Series(df["drawdown"], index=df.index)

        return self._calculate_metrics(df)

    def _calculate_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """计算核心绩效指标"""
        total_ret = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1
        bench_ret = (df["benchmark_curve"].iloc[-1] / self.initial_capital) - 1

        days = (df.index[-1] - df.index[0]).days
        if days <= 0:
            days = 1

        if total_ret <= -1:
            annual_ret = -1.0
        else:
            annual_ret = (1 + total_ret) ** (ANNUAL_DAYS / days) - 1

        max_drawdown = df["drawdown"].min()

        bars_per_year = len(df) / (days / ANNUAL_DAYS)
        volatility = df["net_ret"].std()

        if volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (df["net_ret"].mean() / volatility) * np.sqrt(bars_per_year)

        calmar_ratio = 0.0
        if max_drawdown != 0:
            calmar_ratio = annual_ret / abs(max_drawdown)

        # 索提诺：下行偏差用 MAR=0 的标准定义（全样本取 min(r,0) 的均方根）
        downside_gap = np.minimum(df["net_ret"].to_numpy(), 0.0)
        downside_dev = float(np.sqrt((downside_gap**2).mean()))
        sortino_ratio = 0.0
        if downside_dev:
            sortino_ratio = (df["net_ret"].mean() / downside_dev) * np.sqrt(
                bars_per_year
            )

        # 交易质量：一个回合 = 一段连续的非零持仓
        pos = df["pos"]
        segment = (pos != pos.shift()).cumsum()
        active = pos != 0
        trade_ret = (
            df[active]
            .groupby(segment[active])["net_ret"]
            .apply(lambda r: float((1 + r).prod() - 1))
        )
        hold_bars = df[active].groupby(segment[active]).size()

        gross_gain = float(trade_ret[trade_ret > 0].sum())
        gross_loss = abs(float(trade_ret[trade_ret < 0].sum()))

        return {
            "total_ret": total_ret,
            "bench_ret": bench_ret,
            "annual_ret": annual_ret,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "trade_count": int(len(trade_ret)),
            "order_count": float(df["pos_change"].sum()),
            "win_rate": float((trade_ret > 0).mean()) if len(trade_ret) else 0.0,
            "profit_factor": gross_gain / gross_loss if gross_loss else 0.0,
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "total_cost": float(df["cost"].sum() * self.initial_capital),
            "equity_curve": df["equity_curve"],
            "benchmark_curve": df["benchmark_curve"],
            "drawdown_curve": df["drawdown"],
            "df": df,
        }

    def plot(
        self,
        result: dict,
        title: str = "Backtest Result",
        save_path: Optional[str] = None,
    ):
        """绘制 Matplotlib 静态图"""
        equity = result["equity_curve"]
        benchmark = result["benchmark_curve"]
        drawdown = result["drawdown_curve"]

        plt.style.use("bmh")
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            equity.index,
            equity,
            label=f"Strategy ({self.leverage}x)",
            color="#d62728",
            linewidth=1.5,
        )
        ax1.plot(
            benchmark.index,
            benchmark,
            label="Benchmark (1x)",
            color="gray",
            linestyle="--",
            alpha=0.6,
        )

        ax1.set_title(
            f"{title}\n"
            f"Total Return: {result['total_ret']:.2%} | "
            f"Calmar: {result['calmar_ratio']:.2f} | "
            f"Sharpe: {result['sharpe_ratio']:.2f} | "
            f"Sortino: {result['sortino_ratio']:.2f} | "
            f"MaxDD: {result['max_drawdown']:.2%} | "
            f"WinRate: {result['win_rate']:.1%}"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_ylabel("Equity")

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
            plt.close()
        else:
            plt.show()

    def plot_interactive(
        self,
        result: dict,
        title: str = "Backtest Result",
        save_path: Optional[str] = None,
    ):
        """生成 Plotly 交互式图表"""
        df = result["df"]
        equity = result["equity_curve"]
        benchmark = result["benchmark_curve"]
        drawdown = result["drawdown_curve"]

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
                name=f"Strategy Equity ({self.leverage}x)",
                line=dict(color="#ffa726", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=benchmark,
                name="Benchmark (1x)",
                line=dict(color="gray", dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 2. 价格 & 均线
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

        fig.update_layout(
            title=f"{title} | Sharpe: {result['sharpe_ratio']:.2f} | MaxDD: {result['max_drawdown']:.2%}",
            template="plotly_dark",
            hovermode="x unified",
            height=1000,
        )

        if save_path:
            if not save_path.endswith(".html"):
                save_path += ".html"
            fig.write_html(save_path)
            print(f"📊 交互式研报已生成: {save_path}")
        else:
            fig.show()
