import json
import os
import time
import traceback
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd

from config.settings import GLOBAL_CONFIG
from src.data_loader import MarketDataHandler
from src.strategies.dual_ma import DualMAStrategy


class TraderBot:
    def __init__(self, strategy_name="DualMAStrategy"):
        print(f"🤖 [Trader] 初始化机器人 (模拟盘: {GLOBAL_CONFIG.sandbox_mode})...")

        # 1. 初始化交易所
        self.exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
        if GLOBAL_CONFIG.sandbox_mode:
            self.exchange.set_sandbox_mode(True)
            print("✅ 已切换至 OKX 模拟盘模式")

        # [新增] 加载市场数据 (获取合约面值等信息)
        print("⏳ 正在加载市场详情...")
        self.exchange.load_markets()
        print("✅ 市场详情加载完毕")

        # 2. 加载数据处理器
        self.data_handler = MarketDataHandler(self.exchange)

        # 3. 加载策略配置
        self.strategy_config = self._load_strategy_config(strategy_name)
        print(f"📖 已加载 {len(self.strategy_config)} 个交易对配置")

        # 4. 启动前连接检查 & 余额显示
        self._check_connection_and_balance()

    def _check_connection_and_balance(self):
        """
        检查交易所连接，并打印当前 USDT 余额
        """
        print("🔍 正在检查账户状态...")
        try:
            balance = self.exchange.fetch_balance()
            # OKX V5 统一账户通常关注 'USDT' 的 'free' (可用余额)
            # 安全地处理 None 类型，避免 float(None) 报错
            usdt_free = balance.get("USDT", {}).get("free")
            usdt_balance = float(usdt_free) if usdt_free is not None else 0.0

            if usdt_balance == 0 and "total" in balance:
                usdt_total = balance["total"].get("USDT")
                usdt_balance = float(usdt_total) if usdt_total is not None else 0.0

            print(f"✅ 连接成功! 当前可用余额: {usdt_balance:.2f} USDT")

            if usdt_balance < 100:
                print(
                    "⚠️ 警告: 余额过低！请去 OKX 模拟盘页面点击 '重置资产' 领取虚拟资金。"
                )

        except Exception as e:
            err_msg = str(e)
            print(f"❌ 连接失败: {err_msg}")
            if "50101" in err_msg:
                print(f"💡 致命错误: API Key 与环境不匹配！(Code: 50101)")
                print(
                    f"   当前程序设定为: {'模拟盘 (Sandbox)' if GLOBAL_CONFIG.sandbox_mode else '实盘 (Production)'}"
                )
                print(
                    f"   请检查 .env 文件: 如果使用的是模拟盘 Key，请确保添加 OKX_SANDBOX=true"
                )

    def _load_strategy_config(self, strategy_name):
        path = os.path.join("config", "strategies.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件未找到: {path}")

        with open(path, "r") as f:
            data = json.load(f)
            return data.get(strategy_name, {})

    def get_current_position(self, symbol):
        """
        获取当前交易对的持仓数量 (多单)
        返回: 持仓张数 (float)
        """
        try:
            # 针对 Swap (永续合约)
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos["contracts"]) if pos["contracts"] else 0.0
                if contracts > 0 and pos["side"] == "long":
                    return contracts
            return 0.0
        except Exception as e:
            print(f"❌ [API] 获取持仓失败: {e}")
            return 0.0

    def _calc_contract_amount(self, symbol, amount_usdt, leverage, price):
        """
        [新增] 计算需要购买的合约张数
        """
        try:
            market = self.exchange.market(symbol)
            contract_size = market.get("contractSize")  # 比如 BTC 可能是 0.01 或 100

            if not contract_size:
                print(f"⚠️ 无法获取 {symbol} 合约面值，尝试直接按币数计算...")
                return (amount_usdt * leverage) / price

            # 计算逻辑：
            # 总名义价值 (币) = (USDT金额 * 杠杆) / 价格
            # 张数 = 总名义价值 / 单张面值
            # 注意：OKX 的 contractSize 通常是币的数量 (如 0.01 BTC)

            coin_amount = (amount_usdt * leverage) / price
            contracts = int(coin_amount / contract_size)

            if contracts < 1:
                print(
                    f"⚠️ 资金过小，不足以买入 1 张合约 (需 {contract_size * price / leverage:.2f} USDT)"
                )
                return 0

            return contracts

        except Exception as e:
            print(f"❌ 计算张数出错: {e}")
            return 0

    def execute_trade(self, symbol, action, params):
        """
        执行下单逻辑 (修复 tdMode 和张数问题)
        """
        amount_usdt = params.get("amount_usdt", 100)
        leverage = params.get("leverage", 1)

        # [关键修复] OKX 下单必须指定 tdMode (trade mode)
        # cross: 全仓, isolated: 逐仓, cash: 现货
        # 我们默认使用全仓模式 'cross'
        # 显式注明类型 dict[str, Any] 以支持后续加入 bool 类型参数 (reduceOnly)
        trade_params: dict[str, Any] = {"tdMode": "cross"}

        try:
            # 1. 确保杠杆倍数正确
            try:
                self.exchange.set_leverage(
                    leverage, symbol, params={"mgnMode": "cross"}
                )
            except Exception:
                pass

            # 2. 获取当前市价
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # 3. 执行交易
            if action == "buy":
                # 计算合约张数
                amount_contracts = self._calc_contract_amount(
                    symbol, amount_usdt, leverage, current_price
                )

                if amount_contracts > 0:
                    print(
                        f"🚀 [下单] 买入 {symbol} (做多)... 张数: {amount_contracts} (杠杆 {leverage}x)"
                    )
                    # create_market_buy_order 的 amount 参数在合约模式下通常指“张数”
                    order = self.exchange.create_market_buy_order(
                        symbol, amount_contracts, params=trade_params
                    )
                    print(f"✅ 买入成功! ID: {order['id']}")
                else:
                    print(f"⚠️ 跳过下单: 计算张数为 0")

            elif action == "sell":
                pos_size = self.get_current_position(symbol)
                if pos_size > 0:
                    print(f"📉 [下单] 平仓 {symbol}... 持仓张数: {pos_size}")
                    # 平仓参数
                    close_params = trade_params.copy()
                    close_params["reduceOnly"] = True

                    order = self.exchange.create_market_sell_order(
                        symbol, pos_size, params=close_params
                    )
                    print(f"✅ 平仓成功! ID: {order['id']}")
                else:
                    print("⚠️ 无持仓可平")

        except ccxt.InsufficientFunds as e:
            print(f"❌ [下单失败] 余额不足！请去 OKX 模拟盘充值。")
        except Exception as e:
            print(f"❌ [下单失败] {e}")
            # traceback.print_exc()

    def run_strategy_logic(self, symbol, params):
        """
        运行单个币种的策略逻辑
        """
        timeframe = params["timeframe"]

        try:
            # OKX API 限制，fetch_ohlcv 默认返回 100 条
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=300)
            # type: ignore 忽略 columns 类型误报
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)

            # 💡 丢弃最后一根未收盘 K 线
            df_closed = df.iloc[:-1].copy()
            last_close_time = df_closed.index[-1]

        except Exception as e:
            err_msg = str(e)
            print(f"❌ [数据] 获取 {symbol} K线失败: {err_msg}")
            if "50101" in err_msg:
                print("💡 提示: 请检查 .env 文件中的 OKX_SANDBOX 是否设置为 true")
            return

        # 2. 计算指标 & 信号
        strategy = DualMAStrategy(
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            atr_multiplier=params["atr_multiplier"],
        )

        df_res = strategy.generate_signals(df_closed)

        # 获取最新信号
        current_signal = df_res["signal"].iloc[-1]
        current_close = df_res["close"].iloc[-1]
        ma_fast = df_res["MA_Fast"].iloc[-1]
        ma_slow = df_res["MA_Slow"].iloc[-1]

        print(
            f"🔍 [{symbol}] {timeframe} | 收盘: {last_close_time} | 价格: {current_close} | 信号: {current_signal}"
        )

        # 3. 执行同步
        current_position = self.get_current_position(symbol)
        is_holding = current_position > 0

        if current_signal == 1 and not is_holding:
            print(f"⚡ 信号触发: 金叉买入")
            self.execute_trade(symbol, "buy", params)

        elif current_signal == 0 and is_holding:
            print(f"⚡ 信号触发: 死叉/止损平仓")
            self.execute_trade(symbol, "sell", params)

        else:
            status = "持仓中 🟢" if is_holding else "空仓观望 ⚪"
            # 为了减少日志刷屏，可以注释掉下面这行，或者只打印持仓的
            if is_holding:
                print(f"💤 {status} (持仓张数: {current_position})")

    def start(self):
        """
        主循环
        """
        print(f"\n🚀 交易机器人已启动... (按 Ctrl+C 停止)")

        while True:
            print(f"\n⏰ 巡检时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            for symbol, params in self.strategy_config.items():
                try:
                    self.run_strategy_logic(symbol, params)
                except Exception as e:
                    print(f"❌ 处理 {symbol} 时发生未知错误: {e}")

                # 稍微休息，避免 API 限频
                time.sleep(1)

            print("⏳ 等待 60 秒...")
            time.sleep(60)


if __name__ == "__main__":
    bot = TraderBot()
    bot.start()
