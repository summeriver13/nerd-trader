"""
src/simulation/engine.py
模拟引擎：负责加载历史数据，并以回放的形式驱动 MockGateway 和 策略
Refactored: 集成多标的加载器 (MultiSymbolLoader)，适配 StrategyContext 协议
"""

import os
from datetime import date, datetime
from time import sleep
from typing import Any, Dict, List, Type, Union


from src.analytics.equity import EquityTracker
from src.analytics.reporting import save_trades
from src.config.paths import RESULTS_ROOT
from src.core.event import Event, EventEngine, EventType
from src.core.interfaces import StrategyBase
from src.core.types import (
    BarData,
    OrderRequest,
    PositionData,
    TickData,
    TradeData,
)
from src.simulation.mock_gateway import DEFAULT_INITIAL_CAPITAL, MockGateway
from src.simulation.replay_loader import MultiSymbolLoader, ReplayLoader


class SimulationEngine:
    """
    模拟盘引擎
    功能：
    1. 加载历史数据 (Tick/Bar)
    2. 驱动 EventEngine
    3. 撮合订单 (通过 MockGateway)
    4. 统计结果
    """

    def __init__(
        self,
        data_path: str,
        symbols: Union[str, List[str]],
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        name: str = "BacktestEngine",
    ):
        """
        :param data_path: 数据文件路径 (单标的) 或 数据目录 (多标的)
        :param symbols: 标的代码 (字符串) 或 标的列表 (List)
        :param initial_capital: 初始资金，透传给仿真网关
        """
        self.data_path = data_path
        self.engine_name = name

        # 统一转为列表处理
        if isinstance(symbols, str):
            self.symbols = [symbols]
            self.is_multi_symbol = False
        else:
            self.symbols = symbols
            self.is_multi_symbol = True

        # 1. 初始化核心组件
        self.event_engine = EventEngine()
        self.gateway = MockGateway(
            self.event_engine, gateway_name="BACKTEST", initial_capital=initial_capital
        )

        # 2. 策略容器
        self.strategy: StrategyBase | None = None

        # 3. 数据记录
        self.trades: List[TradeData] = []
        self.daily_results: Dict[date, float] = {}
        self.initial_capital = self.gateway.balance
        # 逐根按收盘价 mark-to-market 的权益序列（口径见 src/analytics）
        self.equity = EquityTracker(
            initial_capital=self.initial_capital, single_symbol=not self.is_multi_symbol
        )

        # 4. 注册日志监听
        self.event_engine.register(EventType.LOG, self.process_log)

    def init_engine(self):
        """初始化引擎"""
        self.event_engine.start()
        self.gateway.connect()
        print(f">>> [{self.engine_name}] 模拟引擎初始化完成")

    def load_strategy(
        self, strategy_class: Type[StrategyBase], settings: dict[str, Any]
    ) -> None:
        """加载策略"""
        self.strategy = strategy_class(
            context=self,  # 注入 context (自身)
            strategy_name=settings.get("strategy_name", "SimStrategy"),
            settings=settings,
        )

        # 注册事件监听
        self.event_engine.register(EventType.TICK, self.process_tick_event)
        self.event_engine.register(
            EventType.BAR, self.process_bar_event
        )  # 新增 Bar 支持
        self.event_engine.register(EventType.ORDER, self.process_order_event)
        self.event_engine.register(EventType.TRADE, self.process_trade_event)

        self.strategy.on_init()
        print(f">>> 策略 {self.strategy.strategy_name} 已加载")

    def _get_loader(self) -> Union[ReplayLoader, MultiSymbolLoader]:
        """工厂方法：构建数据加载器"""
        if not self.is_multi_symbol:
            # 单标的模式：data_path 被视为完整文件路径
            return ReplayLoader(self.data_path, self.symbols[0], is_bar=True)
        else:
            # 多标的模式：data_path 被视为目录
            # 假设文件名为 {symbol}.parquet
            loaders = []
            for sym in self.symbols:
                # 这里假设文件名规则，如果不匹配需要根据实际情况修改
                file_path = os.path.join(self.data_path, f"{sym}.parquet")
                loaders.append(ReplayLoader(file_path, sym, is_bar=True))
            return MultiSymbolLoader(loaders)

    def run(self) -> None:
        """
        [核心循环] 流式回放数据
        """
        if not self.strategy:
            print("错误：未加载策略")
            return

        # 行情回调至少实现一个，否则回放跑完什么都不会发生（静默空跑最难查）
        if not any(
            getattr(type(self.strategy), name) is not getattr(StrategyBase, name)
            for name in ("on_bar", "on_bars", "on_tick")
        ):
            raise ValueError(
                f"策略 {self.strategy.strategy_name} 未实现任何行情回调"
                "（on_bar / on_bars / on_tick 至少一个）"
            )

        if not self.event_engine._active:
            self.init_engine()

        # 订阅行情：回测里网关默认全订阅，但真实网关需要这个接入点
        for symbol in self.symbols:
            self.gateway.subscribe(symbol)

        # 1. 准备数据流
        print(
            f">>> 正在初始化加载器, 模式: {'多标的' if self.is_multi_symbol else '单标的'}"
        )
        loader_instance = self._get_loader()

        if self.is_multi_symbol:
            assert isinstance(loader_instance, MultiSymbolLoader)
            data_stream = loader_instance.merge_streams()
        else:
            assert isinstance(loader_instance, ReplayLoader)
            data_stream = loader_instance.load_data()

        print(">>> 开始回放数据...")
        self.strategy.on_start()

        # 2. 消费数据流
        for data in data_stream:
            # 推送给网关进行撮合 (更新行情快照)
            if isinstance(data, TickData):
                self.gateway.process_tick(data)
            elif isinstance(data, BarData):
                # 需确保 MockGateway 支持 process_bar，或在此处转换为 EventType.BAR
                self.gateway.process_bar(data)

            # [关键同步] 等待事件处理完成 (确保策略处理完当前行情，再进下一个时间点)
            # 在回测中，必须必须串行执行
            while not self.event_engine._queue.empty():
                sleep(0.00001)

        # 3. 结束处理
        print(">>> 数据推送完毕，等待事件队列清空...")
        while not self.event_engine._queue.empty():
            sleep(0.1)

        print(">>> 回放结束")
        self.strategy.on_stop()
        self.event_engine.stop()

        self.show_summary()
        self._save_equity_curve()
        self.gateway.close()

    # --- 事件处理函数 ---

    def process_tick_event(self, event: Event) -> None:
        if self.strategy:
            self.strategy.on_tick(event.data)

    def process_bar_event(self, event: Event) -> None:
        if self.strategy:
            self.strategy.on_bar(event.data)
            # TODO: 未来支持 on_bars (切片) 时在此处聚合
        self._record_equity(event.data)

    def process_order_event(self, event: Event) -> None:
        if self.strategy:
            self.strategy.on_order(event.data)

    def process_trade_event(self, event: Event) -> None:
        trade = event.data
        if trade:
            self.trades.append(trade)
            if self.strategy:
                self.strategy.on_trade(trade)

    # --- 结果统计与导出 ---

    def show_summary(self) -> None:
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

        # 导出成交明细
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parquet_path = RESULTS_ROOT / f"trades_{timestamp}.parquet"
        save_trades(self.trades, parquet_path)
        print(f"📝 详细成交记录已保存至: {parquet_path}")
        print("=" * 40 + "\n")

    # --- 权益曲线 ---

    def _record_equity(self, bar: BarData) -> None:
        """记录本根收盘价下的账户权益与同期买入持有基准"""
        self.equity.record(bar.datetime, self.gateway.equity_value(), bar.close_price)

    def _save_equity_curve(self) -> None:
        """落盘净值/基准/回撤三条序列"""
        strategy_name = self.strategy.strategy_name if self.strategy else "UNKNOWN"
        path = RESULTS_ROOT / (
            f"{self.symbols[0].replace('/', '_')}_{strategy_name}_equity.csv"
        )
        self.equity.save(path)

    # --- [关键] 实现 StrategyContext 协议 ---

    def send_order(self, req: OrderRequest) -> str:
        """策略发送订单接口"""
        return self.gateway.send_order(req)

    def cancel_order(self, orderid: str) -> None:
        """策略撤单接口"""
        self.gateway.cancel_order(orderid)

    def write_log(self, msg: str, strategy_name: str) -> None:
        """策略日志接口"""
        print(f"[{strategy_name}] {msg}")

    def query_account(self) -> float:
        """查询资金"""
        return self.gateway.query_account()

    def query_position(self) -> dict[str, PositionData]:
        """查询持仓"""
        return self.gateway.query_position()

    def process_log(self, event: Event) -> None:
        print(f"[ENGINE-LOG] {event.data}")
