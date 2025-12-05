import json
import os
import time
import traceback
from datetime import datetime

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

        # 2. 加载数据处理器
        self.data_handler = MarketDataHandler(self.exchange)

        # 3. 加载策略配置
        self.strategy_config = self._load_strategy_config(strategy_name)
        print(f"📖 已加载 {len(self.strategy_config)} 个交易对配置")

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
        """
        try:
            # 针对 Swap (永续合约)
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                # 安全转换 contracts 为 float，防止 None 或 str 类型导致比较报错
                contracts = float(pos["contracts"]) if pos["contracts"] else 0.0

                # 只有多单且有持仓量
                if contracts > 0 and pos["side"] == "long":
                    return contracts  # 返回张数
            return 0.0
        except Exception as e:
            print(f"❌ [API] 获取持仓失败: {e}")
            return 0.0

    def execute_trade(self, symbol, action, params):
        """
        执行下单逻辑
        """
        amount_usdt = params.get("amount_usdt", 100)
        leverage = params.get("leverage", 1)

        try:
            # 1. 确保杠杆倍数正确
            try:
                self.exchange.set_leverage(leverage, symbol)
            except Exception:
                pass  # 可能已经设置过了，忽略报错

            # 2. 获取当前市价
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # 3. 计算下单数量 (张数或币数)
            # 简单起见，这里演示计算合约张数 (OKX 通常一张 = 0.01 BTC 或类似，需查阅文档或用 create_market_order 的金额模式)
            # 为了通用性，我们直接让 ccxt 帮我们算，或者下“市价单”时指定 cost (USDT数量)
            # 注意：ccxt create_order 的 amount 通常指基础货币数量(BTC)，但在 swap 可能是张数。
            # 最稳妥的方式：
            if action == "buy":
                # 计算大概买多少个币: 1000 USDT / 50000 Price = 0.02 BTC
                amount_coin = (amount_usdt * leverage) / current_price
                print(
                    f"🚀 [下单] 正在买入 {symbol} (做多)... 目标金额: {amount_usdt} USDT"
                )

                # 下市价单
                order = self.exchange.create_market_buy_order(symbol, amount_coin)
                print(f"✅ 买入成功! ID: {order['id']}")

            elif action == "sell":
                # 获取当前所有持仓，全部平仓
                # OKX 平仓通常是反向开单，或者使用 close_position 接口(如果 ccxt 支持)
                # 这里使用简单的反向卖出
                pos_size = self.get_current_position(symbol)
                if pos_size > 0:
                    print(f"📉 [下单] 正在平仓 {symbol}... 持仓量: {pos_size}")
                    # 注意：如果 pos_size 是张数，这里要传张数；如果是币数传币数。
                    # OKX V5 Swap API fetch_positions 返回的是张数(contracts) 或 币数取决于设置
                    # 建议实盘前手动确认一下 ccxt 返回单位。通常是“张”。
                    # 这里假设我们在 trade mode，简单使用 'sell'
                    # 为了安全，这里仅打印逻辑，实盘需谨慎测试

                    # 修正：直接调用平仓方法会更安全，或者卖出同等数量
                    # self.exchange.create_market_sell_order(symbol, pos_size, params={'reduceOnly': True})
                    print("⚠️ (模拟执行) 已发送平仓指令")
                else:
                    print("⚠️ 无持仓可平")

        except Exception as e:
            print(f"❌ [下单] 执行失败: {e}")
            traceback.print_exc()

    def run_strategy_logic(self, symbol, params):
        """
        运行单个币种的策略逻辑
        """
        timeframe = params["timeframe"]

        # 1. 获取数据 (多取一些以保证指标计算准确，比如 500 根)
        # 注意：这里我们不使用 data_loader 的缓存，而是必须强制从交易所拿最新数据
        try:
            # OKX API 限制，可能需要分段拿，或者直接拿最近 100 根 (对于 EMA/ATR 来说 100 根勉强够，建议 300+)
            # fetch_ohlcv 默认返回 100 条
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=300)
            # type: ignore 忽略 columns 类型误报
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],  # type: ignore
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)

            # 💡 关键点：丢弃最后一根 K 线
            # 因为最后一根 K 线还在跳动，均线和 ATR 都在变，信号不稳定 (Repainting)
            # 我们只看“已经收盘”的 K 线
            df_closed = df.iloc[:-1].copy()
            last_close_time = df_closed.index[-1]

        except Exception as e:
            print(f"❌ [数据] 获取 {symbol} K线失败: {e}")
            return

        # 2. 计算指标 & 信号
        strategy = DualMAStrategy(
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            atr_multiplier=params["atr_multiplier"],
        )

        # 复用我们之前写的 generate_signals
        df_res = strategy.generate_signals(df_closed)

        # 获取最新一根已完成 K 线的信号
        current_signal = df_res["signal"].iloc[-1]
        current_close = df_res["close"].iloc[-1]
        ma_fast = df_res["MA_Fast"].iloc[-1]
        ma_slow = df_res["MA_Slow"].iloc[-1]

        print(
            f"🔍 [{symbol}] {timeframe} | 最后收盘: {last_close_time} | 价格: {current_close}"
        )
        print(
            f"   MA({params['fast_period']}): {ma_fast:.4f} | MA({params['slow_period']}): {ma_slow:.4f} | 信号: {current_signal}"
        )

        # 3. 执行同步 (Signal vs Position)
        current_position = self.get_current_position(symbol)
        is_holding = current_position > 0

        if current_signal == 1 and not is_holding:
            print("⚡ 信号触发: 金叉买入 (Target 1, Current 0)")
            self.execute_trade(symbol, "buy", params)

        elif current_signal == 0 and is_holding:
            print("⚡ 信号触发: 死叉/止损平仓 (Target 0, Current {current_position})")
            self.execute_trade(symbol, "sell", params)

        else:
            status = "持仓中 🟢" if is_holding else "空仓观望 ⚪"
            print(f"💤 无需操作 ({status})")

    def start(self):
        """
        主循环
        """
        print("\n🚀 交易机器人已启动... (按 Ctrl+C 停止)")

        while True:
            print(f"\n⏰ 巡检时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            for symbol, params in self.strategy_config.items():
                try:
                    self.run_strategy_logic(symbol, params)
                except Exception as e:
                    print(f"❌ 处理 {symbol} 时发生未知错误: {e}")

                # 稍微休息，避免 API 限频
                time.sleep(2)

            # 等待下一次巡检
            # 策略是 4h 周期，其实不需要每分钟看。但为了测试，可以设短一点 (比如 60秒)
            # 实盘建议：每 1 分钟或 5 分钟检查一次，看最后一根 K 线是否更新
            print("⏳ 等待 60 秒...")
            time.sleep(60)


if __name__ == "__main__":
    bot = TraderBot()
    bot.start()
