import os
from datetime import datetime
from typing import Any

import ccxt
import matplotlib

# 强制 Matplotlib 使用 Agg 后端（Agg 只渲染到文件，不需要屏幕）
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 引入 psutil (可选)
try:
    import psutil
except ImportError:
    psutil = None

# 引入数据处理模块
from market_data import MarketDataHandler

# 加载当前目录下的 .env 文件
load_dotenv()


class StrategyRunner:
    """
    策略运行器：专注于策略逻辑计算和回测
    """

    def __init__(self):
        # 1. 初始化交易所 (仅用于下载数据，如果不下载可为空)
        api_key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET")
        passphrase = os.getenv("OKX_PASSPHRASE")

        # 代理设置
        use_proxy = os.getenv("OKX_USE_PROXY", "false").lower() == "true"
        proxy_port = os.getenv("OKX_PROXY_PORT", "7890")

        exchange_config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }

        if use_proxy:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            exchange_config["proxies"] = {"http": proxy_url, "https": proxy_url}

        # type: ignore 忽略 basedpyright 报错
        self.exchange = ccxt.okx(exchange_config)  # type: ignore

        # 初始化数据处理器
        self.data_handler = MarketDataHandler(self.exchange)

    def prepare_data(self, symbol: str, timeframe: str = "1h", days_back: int = 30):
        """
        准备数据：优先读取本地，没有则下载
        """
        # 生成一个标准化的文件名，例如 "BTC-USDT_1h.parquet"
        safe_symbol = symbol.replace("/", "-")
        filename = f"{safe_symbol}_{timeframe}.parquet"

        # 1. 尝试从本地加载
        df = self.data_handler.load_from_parquet(filename)

        # 2. 如果本地没有数据，或者强制更新 (这里简化逻辑，没有就下载)
        if df is None or df.empty:
            print("\n🌐 本地无数据，准备从交易所下载...")

            # 计算开始时间
            start_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
            # 添加 type: ignore 忽略 basedpyright 认为 start_dt 可能是 NaTType 的误报
            since_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore

            # 下载
            df = self.data_handler.fetch_history_candles(symbol, timeframe, since_str)

            # 清洗
            df = self.data_handler.clean_data(df, timeframe)

            # 保存
            self.data_handler.save_to_parquet(df, filename)

        return df

    def calculate_indicators(self, df):
        """
        计算策略指标
        """
        if df is None or df.empty:
            return None

        # 确保按时间排序
        df = df.sort_index()

        # 计算均线 (MA20)
        df["MA20"] = df["close"].rolling(20).mean()
        return df

    def run_backtest(self, df, fee_rate=0.0005):
        """
        执行向量化回测
        """
        if df is None:
            return

        print("\n⚡ 正在执行策略回测...")

        # 硬件能力分析
        self.analyze_memory_capacity(df)

        df = df.dropna().copy()

        # 策略逻辑
        df["market_return"] = df["close"].pct_change()
        # 信号: 收盘 > MA20 持仓
        df["signal"] = np.where(df["close"] > df["MA20"], 1, 0)

        # 仓位变化
        df["position"] = df["signal"].shift(1)
        # 使用新版写法避免 FutureWarning
        df["position"] = df["position"].fillna(0)

        # 收益计算
        df["strategy_gross_return"] = df["market_return"] * df["position"]

        # 成本计算
        df["trades"] = df["position"].diff().abs()
        df["transaction_costs"] = df["trades"] * fee_rate
        df["strategy_net_return"] = (
            df["strategy_gross_return"] - df["transaction_costs"]
        )

        # 净值曲线
        df["equity_curve"] = (1 + df["strategy_net_return"]).cumprod()
        df["benchmark_curve"] = (1 + df["market_return"]).cumprod()

        # 输出结果
        total_ret = df["equity_curve"].iloc[-1] - 1
        bench_ret = df["benchmark_curve"].iloc[-1] - 1
        trade_count = df["trades"].sum()

        print("======== 回测结果 ========")
        print(f"数据周期: {df.index[0]} -> {df.index[-1]}")
        print(f"交易次数: {int(trade_count)}")
        print(f"策略收益: {total_ret * 100:.2f}%")
        print(f"基准收益: {bench_ret * 100:.2f}%")

        self.plot_results(df, trade_count)

    def plot_results(self, df, trade_count):
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df.index, df["equity_curve"], label="Strategy", color="red")
        ax1.plot(
            df.index,
            df["benchmark_curve"],
            label="Benchmark",
            color="gray",
            linestyle="--",
        )
        ax1.set_title(f"Backtest Result (Trades: {int(trade_count)})")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(df.index, df["close"], label="Price", alpha=0.5)
        ax2.plot(df.index, df["MA20"], label="MA20", color="orange")

        filename = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.tight_layout()
        plt.savefig(filename)
        print(f"📊 图表已保存: {filename}")
        plt.close()

    @staticmethod
    def analyze_memory_capacity(df):
        # 简单的内存分析打印
        rows = len(df)
        print(f"[硬件分析] 当前加载数据: {rows} 行")
        if psutil:
            mem = psutil.virtual_memory()
            print(f"[硬件分析] 系统可用内存: {mem.available / (1024**3):.1f} GB")


if __name__ == "__main__":
    runner = StrategyRunner()

    # 这里我们设置回测最近 90 天的数据
    # 程序会自动下载 -> 清洗 -> 保存 -> 读取
    # 下次运行如果文件存在，会直接读取，秒开！
    df = runner.prepare_data("BTC/USDT", timeframe="1h", days_back=90)

    if df is not None:
        df = runner.calculate_indicators(df)
        runner.run_backtest(df)
