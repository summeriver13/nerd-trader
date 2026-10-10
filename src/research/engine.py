from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 年化系数
from src.analytics import metrics
from src.analytics.metrics import (
    annualize,
    annualized_sharpe,
    bars_per_year,
    drawdown_series,
)

# 引入我们刚才写的 Domain
from src.research.domains.base import MarketDomain
from src.research.execution import (
    FILL_COLUMNS,
    ExecutionResult,
    MakerPolicy,
    execute_maker,
)


class BacktestEngine:
    """
    轻量级向量化回测引擎 (支持多市场 Domain + 杠杆)
    """

    def __init__(
        self,
        domain: MarketDomain,  # 注入市场规则
        initial_capital: float = 10000.0,
        leverage: float = 1.0,
        *,
        maker: MakerPolicy | None = None,  # 传入则改用 maker 挂单执行
        funding: pd.Series | None = None,  # 按日资金费率（正 = 多头付钱）
    ):
        self.domain = domain
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maker = maker
        self.funding = funding

    def _funding_cost(self, index: pd.Index, pos: pd.Series) -> pd.Series:
        """永续资金费率：多头付正费率、空头收正费率，按持仓名义×杠杆计。

        逐次结算已按日汇总，日费率按当天实际有几根 K 线分摊（1d 一根正好一天，
        4h 六根各占 1/6）：不摊的话日内周期会把同一天的费率重复计，4h 放大 6 倍。
        缓存没覆盖到的日子按 0 计，覆盖情况由取数侧打印，不在这里静默。
        """
        if self.funding is None:
            return pd.Series(0.0, index=index)
        days = pd.DatetimeIndex(pd.DatetimeIndex(index).to_series().dt.normalize())
        rate = np.nan_to_num(self.funding.reindex(days).to_numpy(dtype=float), nan=0.0)
        # 按当天实际根数分摊（当天缺根时不会少算：那一天合计仍是完整日费率）
        _, inverse, counts = np.unique(
            days.to_numpy(), return_inverse=True, return_counts=True
        )
        share = 1.0 / counts[inverse]
        return pd.Series(
            pos.to_numpy(dtype=float) * rate * share * self.leverage, index=index
        )

    def _resolve_fills(
        self, df: pd.DataFrame, target_pos: pd.Series
    ) -> ExecutionResult:
        """把目标仓位解析成实际成交仓位、逐根成本与成交明细。

        默认路径按 Domain 规则收取市价单成本；传入 ``maker`` 时改走
        挂单执行（可能延迟成交或回退市价）。
        """
        if self.maker is None or not self.maker.enabled:
            change = target_pos.diff().fillna(0)
            notional = change.abs() * self.initial_capital * self.leverage
            cost_obj = self.domain.calculate_cost(notional, change < 0)
            return ExecutionResult(
                pos=target_pos,
                cost=cost_obj.total / self.initial_capital,
                fills=pd.DataFrame(columns=pd.Index(FILL_COLUMNS)),
            )

        executed = execute_maker(df, target_pos, self.maker)
        return ExecutionResult(
            pos=executed.pos, cost=executed.cost * self.leverage, fills=executed.fills
        )

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
        target_pos = cast(pd.Series, df["signal"].shift(1).fillna(0))

        # 3. 计算持仓变化 (Trades)
        # 原始变化量 (用于判断方向)
        executed = self._resolve_fills(df, target_pos)
        df["pos"] = executed.pos
        df["cost"] = executed.cost
        raw_pos_change = df["pos"].diff().fillna(0)
        df["pos_change"] = raw_pos_change.abs()

        # 4. 计算基准收益 (Market Return)
        df["pct_change"] = df["close"].pct_change().fillna(0)

        # 5. 计算策略收益 (Strategy Return)
        # 策略收益 = 持仓 * 市场涨跌幅 * 杠杆倍数
        df["strategy_raw_ret"] = df["pos"] * df["pct_change"] * self.leverage

        # 6. 计算交易成本 (Cost)
        # ------------------------------------------------------------------
        # 6.1 计算成交名义金额 (Notional Value)
        # 简化模型：基于初始本金估算
        trade_notional = df["pos_change"] * self.initial_capital * self.leverage

        # 6.2 识别卖出行为 (用于印花税)
        is_sell_mask = raw_pos_change < 0

        # 6.3 调用 Domain 获取费用详情 (绝对金额)
        # maker 路径已在 _resolve_fills 中算好逐根成本，此处仅保留名义金额
        if self.maker is None or not self.maker.enabled:
            cost_obj = self.domain.calculate_cost(trade_notional, is_sell_mask)

            # 6.4 将绝对费用转换为收益率扣减
            # Cost Rate = 总费用 / 初始本金
            df["cost"] = cost_obj.total / self.initial_capital
        # ------------------------------------------------------------------

        # 7. 计算净收益 (Net Return) = 策略毛收益 − 交易成本 − 资金费率
        df["funding_cost"] = self._funding_cost(df.index, cast(pd.Series, df["pos"]))
        df["net_ret"] = df["strategy_raw_ret"] - df["cost"] - df["funding_cost"]

        # 8. 计算资金曲线 (Equity Curve)
        df["equity_curve"] = (1 + df["net_ret"]).cumprod() * self.initial_capital

        # === 爆仓检测 ===
        if (df["equity_curve"] <= 0).any():
            bust_idx = df[df["equity_curve"] <= 0].index[0]
            print(f"💀 [严重警告] 策略在 {bust_idx} 发生爆仓(归零)！")
            df.loc[bust_idx:, "equity_curve"] = 0
            df.loc[bust_idx:, "net_ret"] = 0

        df["benchmark_curve"] = (1 + df["pct_change"]).cumprod() * self.initial_capital

        # 9. 计算回撤（口径来自 src/analytics/metrics）
        df["drawdown"] = drawdown_series(cast(pd.Series, df["equity_curve"]))

        return self._calculate_metrics(df, executed.fills)

    def _calculate_metrics(
        self, df: pd.DataFrame, fills: pd.DataFrame
    ) -> dict[str, Any]:
        """计算核心绩效指标（口径统一来自 src/analytics/metrics）"""
        net_ret = cast(pd.Series, df["net_ret"])
        total_ret = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1
        bench_ret = (df["benchmark_curve"].iloc[-1] / self.initial_capital) - 1

        # 区间跨度用日历天，7×24 与有缺口的品种都按实际跨度折算
        periods = (df.index[-1] - df.index[0]).days
        if periods <= 0:
            periods = 1
        per_year = bars_per_year(df.index)

        annual_ret = annualize(total_ret, periods)
        max_drawdown = float(df["drawdown"].min())
        sharpe_ratio = annualized_sharpe(net_ret, per_year)
        sortino_ratio = metrics.sortino_ratio(net_ret, per_year)
        calmar_ratio = metrics.calmar_ratio(annual_ret, max_drawdown)

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

        # 持仓时间占比：策略只在部分时间有仓位，拿它跟满仓基准比收益率本来就不公平
        exposure = float(active.mean())
        long_share = float((pos > 0).mean())
        short_share = float((pos < 0).mean())

        # 多空归因：毛收益按方向拆开，用对数收益求和（几何口径）。
        # 简单求和会被波动率放大——高波动标的上「求和 +1882% 而实际净值 −27%」，读起来会带偏。
        gross = cast(pd.Series, df["strategy_raw_ret"])
        long_pnl = float(np.expm1(np.log1p(gross.loc[pos > 0]).sum()))
        short_pnl = float(np.expm1(np.log1p(gross.loc[pos < 0]).sum()))

        gross_gain = float(trade_ret[trade_ret > 0].sum())
        gross_loss = abs(float(trade_ret[trade_ret < 0].sum()))

        # 资金费率：未注入时不报这个数（而不是报 0，免得被读成「已计入」）
        funding_total = (
            float(df["funding_cost"].sum()) if self.funding is not None else None
        )

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
            "exposure": exposure,
            "long_share": long_share,
            "short_share": short_share,
            "avg_abs_pos": float(pos.abs().mean()),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "win_rate": float((trade_ret > 0).mean()) if len(trade_ret) else 0.0,
            "profit_factor": gross_gain / gross_loss if gross_loss else 0.0,
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "total_cost": float(df["cost"].sum() * self.initial_capital),
            "total_funding": funding_total,
            "funding_annual": (
                annualize(funding_total, periods) if funding_total is not None else None
            ),
            "fills": fills,
            "maker_fill_rate": (
                float((fills["fill"] == "maker").mean()) if len(fills) else None
            ),
            "equity_curve": df["equity_curve"],
            "benchmark_curve": df["benchmark_curve"],
            "drawdown_curve": df["drawdown"],
            "df": df,
        }

    def plot(
        self,
        result: dict[str, Any],
        title: str = "Backtest Result",
        save_path: str | None = None,
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
        result: dict[str, Any],
        title: str = "Backtest Result",
        save_path: str | None = None,
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
                line={"color": "#ffa726", "width": 2},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=benchmark,
                name="Benchmark (1x)",
                line={"color": "gray", "dash": "dot"},
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
                line={"color": "#29b6f6", "width": 1},
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
                    line={"color": "#ef5350", "width": 1},
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
                    line={"color": "#66bb6a", "width": 1},
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
                line={"color": "#ef5350", "width": 1},
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
