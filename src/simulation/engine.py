"""
src/simulation/engine.py
模拟引擎：负责加载历史数据，并以回放的形式驱动 MockGateway 和 策略
Refactored: Fully integrated ReplayLoader, Synchronization, and Trade Recording
"""

import os
from datetime import date, datetime
from time import sleep
from typing import Type

import pandas as pd

from src.core.event import Event, EventEngine, EventType
from src.core.interfaces import StrategyBase
from src.core.types import OrderRequest, TradeData
from src.simulation.mock_gateway import MockGateway

# [新增] 引入回放加载器
from src.simulation.replay_loader import ReplayLoader


class SimulationEngine:
    """
    模拟盘引擎
    实现了 StrategyContext 协议
    """

    def __init__(self, start_date: str, data_path: str, symbol: str = "UNKNOWN"):
        self.start_date = start_date
        self.data_path = data_path
        self.symbol = symbol

        # 1. 初始化核心组件
        self.event_engine = EventEngine()
        self.gateway = MockGateway(self.event_engine, gateway_name="BACKTEST")

        # 2. 策略容器
        self.strategy: StrategyBase | None = None

        # 3. 数据记录
        self.trades: list[TradeData] = []
        self.daily_results: dict[date, float] = {}

        # 4. 注册日志监听
        self.event_engine.register(EventType.LOG, self.process_log)

    def init_engine(self):
        """初始化引擎"""
        self.event_engine.start()
        self.gateway.connect()
        print(">>> 模拟引擎初始化完成")

    def load_strategy(self, strategy_class: Type[StrategyBase], settings: dict):
        """加载策略"""
        self.strategy = strategy_class(
            context=self,
            strategy_name="SimulationStrategy",
            settings=settings,
        )

        self.event_engine.register(EventType.TICK, self.process_tick_event)
        self.event_engine.register(EventType.ORDER, self.process_order_event)
        self.event_engine.register(EventType.TRADE, self.process_trade_event)

        self.strategy.on_init()
        print(f">>> 策略 {self.strategy.strategy_name} 已加载")

    def run(self):
        """
        [核心循环] 使用 ReplayLoader 流式回放数据
        """
        if not self.strategy:
            print("错误：未加载策略")
            return

        if not self.event_engine._active:
            self.init_engine()

        # [修改] 使用 ReplayLoader 代替手写 Pandas 逻辑
        print(f">>> 正在初始化加载器: {self.data_path}")
        loader = ReplayLoader(self.data_path, self.symbol)

        # 获取生成器 (Generator)
        tick_stream = loader.load_parquet()

        print(">>> 开始回放数据...")
        self.strategy.on_start()

        # [修改] 极简循环：直接从加载器拿 Tick
        for tick in tick_stream:
            # 1. 喂给网关
            self.gateway.process_tick(tick)

            # 2. [关键同步] 等待事件处理完成
            while not self.event_engine._queue.empty():
                sleep(0.0001)

        print(">>> 数据推送完毕，等待事件队列清空...")
        while not self.event_engine._queue.empty():
            sleep(0.1)

        print(">>> 回放结束")
        self.strategy.on_stop()
        self.event_engine.stop()

        self.show_summary()

    # --- 事件处理函数 ---

    def process_tick_event(self, event: Event):
        if self.strategy:
            self.strategy.on_tick(event.data)

    def process_order_event(self, event: Event):
        if self.strategy:
            self.strategy.on_order(event.data)

    def process_trade_event(self, event: Event):
        trade = event.data
        if trade:
            self.trades.append(trade)
            if self.strategy:
                self.strategy.on_trade(trade)

    # --- 结果统计与导出 ---

    def show_summary(self):
        if not self.trades:
            print("\n⚠️ 本次回测无成交记录。")
            return

        print("\n" + "=" * 40)
        print("📊 回测结果摘要 (Backtest Summary)")
        print("=" * 40)
        print(f"交易笔数 (Total Trades): {len(self.trades)}")

        first_trade = self.trades[0]
        last_trade = self.trades[-1]
        print(f"首笔成交: {first_trade.datetime}")
        print(f"末笔成交: {last_trade.datetime}")

        # 导出 CSV
        trade_dicts = []
        for t in self.trades:
            d = {
                "symbol": t.symbol,
                "direction": t.direction.value,
                "offset": t.offset.value,
                "price": t.price,
                "volume": t.volume,
                "datetime": t.datetime,
            }
            trade_dicts.append(d)

        df_trades = pd.DataFrame(trade_dicts)

        os.makedirs("data/results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/results/trades_{timestamp}.csv"

        df_trades.to_csv(csv_path, index=False)
        print(f"📝 详细成交记录已保存至: {csv_path}")
        print("=" * 40 + "\n")

    # --- 实现 StrategyContext 协议 ---

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        return self.gateway.send_order(req)

    def cancel_order(self, orderid: str, gateway_name: str):
        self.gateway.cancel_order(orderid, symbol="UNKNOWN")

    def write_log(self, msg: str, strategy_name: str):
        print(f"[{strategy_name}] {msg}")

    def query_account(self):
        pass

    def process_log(self, event: Event):
        print(f"[ENGINE-LOG] {event.data}")
