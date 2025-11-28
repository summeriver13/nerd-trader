import os
import time
from datetime import datetime

import ccxt
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
        # 考虑到你在东京VPS，通常不需要代理，但保留该逻辑作为兼容性
        use_proxy = os.getenv("OKX_USE_PROXY", "false").lower() == "true"
        proxy_port = os.getenv("OKX_PROXY_PORT", "7890")

        # 检查是否缺少必要的 Key
        if not all([api_key, secret, passphrase]):
            raise ValueError("❌ 错误: 未在 .env 文件中找到 API 配置，请检查环境变量。")

        # 3. 基础配置
        exchange_config = {
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
        }

        # 4. 代理配置
        if use_proxy:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            exchange_config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            print(f"[系统] 代理已启用: {proxy_url}")
        else:
            print("[系统] 未启用代理")

        # 5. 实例化
        self.exchange = ccxt.okx(exchange_config)

    def check_connection(self):
        """测试连接"""
        try:
            server_time = self.exchange.fetch_time()
            readable_time = datetime.fromtimestamp(server_time / 1000)
            print(f"✅ [连接成功] 服务器时间: {readable_time}")
            return True
        except Exception as e:
            print(f"❌ [连接失败] {e}")
            return False

    def get_ticker_price(self, symbol: str):
        """获取最新价"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            print(f"[{symbol}] 当前价格: {ticker['last']}")
            return ticker["last"]
        except Exception as e:
            print(f"❌ [获取行情失败] {e}")
            return 0.0

    def get_account_balance(self, currency: str = "USDT"):
        """获取余额"""
        try:
            balance = self.exchange.fetch_balance()
            target = balance.get(currency, {})
            print(f"--- {currency} 资产 ---")
            print(f"可用: {target.get('free', 0)}")
            print(f"总计: {target.get('total', 0)}")
        except Exception as e:
            print(f"❌ [获取余额失败] {e}")

    def get_kline_data(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """
        获取 K 线数据并转换为 DataFrame
        """
        try:
            print(f"\n[数据] 正在获取 {symbol} {timeframe} K线 (Limit={limit})...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            # 转换为 Pandas 表格
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)

            print(f"✅ 获取成功: {len(df)} 条数据")
            return df
        except Exception as e:
            print(f"❌ [获取K线失败] {e}")
            return None

    def get_order_book(self, symbol: str):
        """获取盘口深度"""
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=5)
            ask1 = ob["asks"][0]  # 卖一
            bid1 = ob["bids"][0]  # 买一
            print(f"\n[盘口] {symbol}")
            print(f"卖一: {ask1[0]} (量: {ask1[1]})")
            print(f"买一: {bid1[0]} (量: {bid1[1]})")
            print(f"价差: {ask1[0] - bid1[0]:.2f}")
        except Exception as e:
            print(f"❌ [获取深度失败] {e}")

    def calculate_indicators(self, df):
        """
        计算技术指标 (MA, Bollinger Bands)
        """
        if df is None or len(df) < 20:
            print("❌ 数据量太少，无法计算指标 (至少需要20条)")
            return df

        # 1. 转换时区 (UTC -> 北京时间)
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

        # 2. 计算均线 (MA)
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA20"] = df["close"].rolling(20).mean()

        # 3. 计算布林带
        std_dev = df["close"].rolling(20).std()
        df["Upper_Band"] = df["MA20"] + (std_dev * 2)
        df["Lower_Band"] = df["MA20"] - (std_dev * 2)

        return df


def run_vectorized_backtest(df):
    """
    执行向量化回测
    """
    print("\n⚡ 正在执行回测...")

    # 1. 计算市场基准收益 (Market Return)
    df["market_return"] = df["close"].pct_change()

    # 2. 生成交易信号 (Signal)
    # 逻辑: 收盘价 > 20小时均线时持有 (1)，否则空仓 (0)
    df["signal"] = (df["close"] > df["MA20"]).astype(int)

    # 3. 计算策略收益 (Strategy Return)
    df["strategy_return"] = df["market_return"] * df["signal"].shift(1)

    # 4. 计算资金曲线 (Equity Curve)
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["benchmark_curve"] = (1 + df["market_return"]).cumprod()

    # --- 输出回测结果 ---
    total_return = df["equity_curve"].iloc[-1] - 1
    benchmark_return = df["benchmark_curve"].iloc[-1] - 1

    print(f"======== 回测报告 ========")
    print(f"回测周期: {len(df)} 小时")
    print(f"策略总收益: {total_return * 100:.2f}%")
    print(f"持币不动收益: {benchmark_return * 100:.2f}%")

    if total_return > benchmark_return:
        print("✅ 策略跑赢了市场！(厉害)")
    else:
        print("❌ 策略跑输了市场。(正常，震荡市均线策略通常会磨损)")

    # --- 画图 (保存到文件) ---
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity_curve"], label="Strategy (MA20)", color="red")
    plt.plot(
        df.index,
        df["benchmark_curve"],
        label="Benchmark (Hold BTC)",
        color="gray",
        linestyle="--",
    )
    plt.title("Vectorized Backtest: MA20 Strategy")
    plt.legend()
    plt.grid(True)

    # 替换 plt.show() 为保存文件逻辑
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_result_{current_time_str}.png"
    filepath = os.path.join(os.getcwd(), filename)

    plt.savefig(filepath)
    print(f"📊 回测图表已保存到: {filepath}")
    plt.close() # 关闭图形，释放内存


# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        trader = OkxTrader()

        if trader.check_connection():
            symbol = "BTC/USDT"

            # 1. 运行基础信息检查
            trader.get_ticker_price(symbol)
            trader.get_account_balance("USDT")
            trader.get_order_book(symbol)

            # 2. 拉取数据 (这里使用 300 条 K 线进行回测)
            df = trader.get_kline_data(symbol, "1h", limit=300)

            # 3. 计算指标 & 运行回测
            if df is not None:
                df = trader.calculate_indicators(df)

                # 运行回测函数 (会生成 backtest_result_....png 文件)
                run_vectorized_backtest(df)

                # 打印最新的信号供参考
                print("\n[实时信号参考]")
                last_close = df["close"].iloc[-1]
                last_ma20 = df["MA20"].iloc[-1]
                print(f"当前价格: {last_close} | MA20: {last_ma20:.2f}")

    except ValueError as e:
        print(e)
        time.sleep(5) # 休息一下，防止循环崩溃
    except Exception as e:
        print(f"程序异常退出: {e}")
        time.sleep(5)
