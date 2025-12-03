# 文件路径: core/gateway/okx_gateway.py
# 核心功能: OKX 交易所的 REST API 连接、认证和低频查询。

import os
import time
from datetime import datetime

import ccxt
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件
load_dotenv()


class OkxGateway:
    """
    OKX 交易与数据网关接口封装类。

    该类负责将系统核心（事件引擎、执行引擎）与 OKX 交易所 API 连接，
    是实现实盘交易功能的核心组件。

    注意：在完整的事件驱动系统中，该类应继承自 BaseGateway，并应包含 WebSocket 客户端。
    """

    def __init__(self):
        """
        初始化连接 (自动从环境变量读取配置)。
        """
        # 1. 从环境变量获取敏感信息
        api_key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET")
        passphrase = os.getenv("OKX_PASSPHRASE")

        # 2. 从环境变量获取代理配置
        use_proxy = os.getenv("OKX_USE_PROXY", "false").lower() == "true"
        proxy_port = os.getenv("OKX_PROXY_PORT", "7890")

        # 检查是否缺少必要的 Key
        if not all([api_key, secret, passphrase]):
            raise ValueError(
                "❌ 错误: 未在 .env 文件中找到 OKX API 配置，请检查环境变量。"
            )

        # 3. 基础配置
        exchange_config = {
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            # 启用 ccxt 内置限速器，这是加密货币网关的关键
            "enableRateLimit": True,
        }

        # 4. 代理配置 (逻辑保持不变，确保兼容性)
        if use_proxy:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            exchange_config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            print(f"[系统] 代理已启用: {proxy_url}")
        else:
            print("[系统] 未启用代理")

        # 5. 实例化 ccxt 交易所对象
        self.exchange = ccxt.okx(exchange_config)

    def check_connection(self):
        """
        测试连接并获取服务器时间。
        """
        try:
            server_time = self.exchange.fetch_time()
            readable_time = datetime.fromtimestamp(server_time / 1000)
            print(f"✅ [连接成功] OKX 服务器时间: {readable_time}")
            return True
        except Exception as e:
            print(f"❌ [连接失败] {e}")
            return False

    def get_ticker_price(self, symbol: str):
        """
        通过 REST API 获取最新价。
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            print(f"[{symbol}] 当前价格: {ticker['last']}")
            return ticker["last"]
        except Exception as e:
            print(f"❌ [获取行情失败] {e}")
            return None

    def get_account_balance(self, currency: str = "USDT"):
        """
        获取账户余额。
        """
        try:
            balance = self.exchange.fetch_balance()
            target = balance.get(currency, {})
            print(f"--- {currency} 资产 ---")
            print(f"可用: {target.get('free', 0)}")
            print(f"总计: {target.get('total', 0)}")
            return target
        except Exception as e:
            print(f"❌ [获取余额失败] {e}")
            return None

    def get_kline_data(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """
        获取 K 线数据 (OHLCV 列表)。

        注意：Gateway 负责获取原始数据，数据处理（如转换为 DataFrame）应由 DataHandler 完成。
        """
        try:
            print(f"\n[数据] 正在获取 {symbol} {timeframe} K线 (Limit={limit})...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                print("❌ 获取 K 线数据为空。")
                return None

            print(f"✅ 获取成功: {len(ohlcv)} 条数据")
            # 原始 ccxt 格式：[[timestamp, open, high, low, close, volume], ...]
            return ohlcv
        except Exception as e:
            print(f"❌ [获取K线失败] {e}")
            return None

    def get_order_book(self, symbol: str):
        """
        获取盘口深度数据。
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=5)
            ask1 = ob["asks"][0]
            bid1 = ob["bids"][0]
            print(f"\n[盘口] {symbol}")
            print(f"卖一: {ask1[0]} (量: {ask1[1]})")
            print(f"买一: {bid1[0]} (量: {bid1[1]})")
            print(f"价差: {ask1[0] - bid1[0]:.2f}")
            return ob
        except Exception as e:
            print(f"❌ [获取深度失败] {e}")
            return None

    # TODO: 作为完整的交易网关，您还需要增加以下核心功能：
    # 1. 交易功能：send_order, cancel_order, query_open_orders
    # 2. 实时数据：subscribe_market_data, subscribe_account_updates (通常通过 WebSocket 实现)
    pass
