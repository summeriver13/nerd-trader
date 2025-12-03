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

# 加载当前目录下的 .env 文件
load_dotenv()


class OkxTrader:
    """
    OKX 交易接口封装类
    """

    def __init__(self):
        """
        初始化连接 (自动从环境变量读取配置)
        """
        # 1. 从环境变量获取敏感信息
        api_key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET")
        passphrase = os.getenv("OKX_PASSPHRASE")

        # 2. 从环境变量获取代理配置
        use_proxy = os.getenv("OKX_USE_PROXY", "false").lower() == "true"
        proxy_port = os.getenv("OKX_PROXY_PORT", "7890")

        if not all([api_key, secret, passphrase]):
            # 为了让代码在没有 env 的情况下也能演示回测逻辑，这里做个非阻塞提示
            print(
                "⚠️ 警告: 未找到完整 API 配置，实盘功能将不可用，仅可进行回测逻辑演示。"
            )
            self.exchange = None
            return

        # 3. 基础配置
        # 使用 dict[str, Any] 显式标注类型，解决类型检查器推断为 dict[str, Unknown] 的问题
        exchange_config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # 默认使用合约，现货则改为 'spot'
        }

        # 4. 代理配置
        if use_proxy:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            exchange_config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }

        # 5. 实例化
        # ccxt 的类型定义比较严格，直接传入 dict[str, Any] 会导致 basedpyright 报错
        # 这里使用 # type: ignore 忽略该行的类型检查，因为运行时这是完全合法的
        self.exchange = ccxt.okx(exchange_config)  # type: ignore

    def fetch_data_mock_or_real(self, symbol: str, limit: int = 500):
        """
        如果连接成功则从交易所获取数据，否则生成模拟数据用于测试
        """
        if self.exchange:
            try:
                print(f"[网络] 正在获取 {symbol} 数据...")
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", limit=limit)
                # pandas 的类型检查有时会误报 columns 类型不匹配，这里忽略该错误
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
                )
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("datetime", inplace=True)
                return df
            except Exception as e:
                print(f"❌ API 获取失败: {e}")
                return None
        else:
            print("[模拟] 生成随机漫步数据用于演示...")
            dates = pd.date_range(end=datetime.now(), periods=limit, freq="1h")
            # 随机漫步生成价格
            prices = 10000 + np.cumsum(np.random.randn(limit)) * 100
            df = pd.DataFrame(index=dates)
            df["close"] = prices
            # 简单模拟其他列
            df["open"] = df["close"].shift(1).fillna(prices[0])
            df["high"] = df[["open", "close"]].max(axis=1) + 50
            df["low"] = df[["open", "close"]].min(axis=1) - 50
            df["volume"] = np.random.randint(100, 1000, size=limit)
            return df

    def calculate_indicators(self, df):
        """
        计算技术指标
        """
        if df is None:
            return None

        # 1. 转换时区
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Shanghai")

        # 2. 计算均线
        df["MA20"] = df["close"].rolling(20).mean()

        return df


def run_robust_backtest(df, fee_rate=0.0005):
    """
    执行带手续费的向量化回测
    fee_rate: 单边手续费率，默认万分之五
    """
    print("\n⚡ 正在执行增强版回测 (含手续费)...")

    # 清洗数据，去除 MA 计算产生的 NaN
    df = df.dropna().copy()

    # 1. 计算基准收益 (对数收益率通常计算更准确，这里保持简单百分比)
    df["market_return"] = df["close"].pct_change()

    # 2. 生成信号
    # 1 = 持仓, 0 = 空仓
    df["signal"] = np.where(df["close"] > df["MA20"], 1, 0)

    # 3. 计算仓位变化 (用于计算手续费)
    # diff() != 0 的地方就是发生了买入或卖出
    # shift(1) 是因为今天的信号决定明天的持仓，或者说通过昨天的信号决定今天的持仓
    # 这里我们假设：K线结束时产生信号，下一根K线开盘立即成交

    # 将信号下移一格，代表"实际持仓状态"
    df["position"] = df["signal"].shift(1)

    # 策略毛收益 (不含手续费)
    df["strategy_gross_return"] = df["market_return"] * df["position"]

    # 4. 计算手续费成本
    # position 发生变化时 (0->1 买入, 1->0 卖出)，产生手续费
    # 填充 NaN 为 0
    df["position"].fillna(0, inplace=True)
    df["trades"] = df["position"].diff().abs()

    # 手续费 = 交易动作 * 费率
    # 注意：这里简化处理，直接从收益率中减去费率。
    df["transaction_costs"] = df["trades"] * fee_rate

    # 5. 策略净收益
    df["strategy_net_return"] = df["strategy_gross_return"] - df["transaction_costs"]

    # 6. 资金曲线
    df["equity_curve"] = (1 + df["strategy_net_return"]).cumprod()
    df["benchmark_curve"] = (1 + df["market_return"]).cumprod()

    # --- 输出报告 ---
    total_return = df["equity_curve"].iloc[-1] - 1
    benchmark_return = df["benchmark_curve"].iloc[-1] - 1
    trade_count = df["trades"].sum()

    print("======== 增强回测报告 ========")
    print(f"交易次数: {int(trade_count)}")
    print(f"费率设置: {fee_rate * 100}%")
    print(f"策略净收益: {total_return * 100:.2f}%")
    print(f"基准收益:   {benchmark_return * 100:.2f}%")

    # --- 绘图 ---
    plt.figure(figsize=(12, 8))

    # 上图：资金曲线
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df["equity_curve"], label="Strategy (Net)", color="red")
    ax1.plot(
        df.index, df["benchmark_curve"], label="Benchmark", color="gray", linestyle="--"
    )
    ax1.set_title(f"Backtest: MA20 Strategy (Trades: {int(trade_count)})")
    ax1.legend()
    ax1.grid(True)

    # 下图：价格与均线
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df.index, df["close"], label="Price", alpha=0.5)
    ax2.plot(df.index, df["MA20"], label="MA20", color="orange", linestyle="--")
    # 标记买卖点
    buys = df[df["trades"] > 0]  # 简化显示，所有交易点
    ax2.scatter(
        buys.index, buys["close"], marker="^", color="green", alpha=1, label="Trade"
    )
    ax2.legend()
    ax2.grid(True)

    filename = f"backtest_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 详细图表已保存: {filename}")
    plt.close()


if __name__ == "__main__":
    trader = OkxTrader()
    # 使用模拟数据或实盘数据
    df = trader.fetch_data_mock_or_real("BTC/USDT", limit=1000)

    if df is not None:
        df = trader.calculate_indicators(df)
        # 传入手续费率，例如万分之五
        run_robust_backtest(df, fee_rate=0.0005)
