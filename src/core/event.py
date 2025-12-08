"""
src/core/event.py
轻量级事件引擎 (Production Ready & Modernized)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Any, Callable


# --- [变更点 1] 使用 StrEnum 定义事件类型 ---
class EventType(StrEnum):
    """
    系统内部通用的事件类型定义
    使用 '类别.名称' 的命名空间风格，提高可读性
    """

    # 系统级事件
    LOG = "system.log"
    TIMER = "system.timer"  # 1秒触发一次
    ERROR = "system.error"

    # 行情级事件
    TICK = "market.tick"
    BAR = "market.bar"

    # 交易级事件
    ORDER = "trade.order"  # 订单状态更新
    TRADE = "trade.trade"  # 成交回报
    POSITION = "trade.position"  # 持仓更新
    ACCOUNT = "trade.account"  # 资金更新


@dataclass(slots=True)
class Event:
    """
    事件对象
    slots=True 减少内存占用
    """

    type: str | EventType  # 既可以是枚举，也可以是字符串(兼容自定义事件)
    data: Any = None


class EventEngine:
    """
    事件引擎：负责接收事件并分发给处理函数
    """

    def __init__(self):
        self._queue: Queue = Queue()
        self._active: bool = False
        self._thread: Thread | None = None
        self._timer_thread: Thread | None = None
        self._handlers: defaultdict = defaultdict(list)

        # 获取专属 logger
        self.logger = logging.getLogger("EventEngine")

    def start(self):
        """启动引擎"""
        if self._active:
            return

        self._active = True

        # 1. 启动事件处理线程
        self._thread = Thread(target=self._run, name="EventEngine_Worker")
        self._thread.daemon = True
        self._thread.start()

        # 2. 启动定时器线程
        self._timer_thread = Thread(target=self._run_timer, name="EventEngine_Timer")
        self._timer_thread.daemon = True
        self._timer_thread.start()

        self.logger.info("事件引擎已启动")

    def stop(self):
        """停止引擎"""
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join()

        self.logger.info("事件引擎已停止")

    def put(self, event: Event):
        """向队列压入事件"""
        self._queue.put(event)

    def register(self, type: str | EventType, handler: Callable):
        """注册事件处理函数"""
        if handler not in self._handlers[type]:
            self._handlers[type].append(handler)

    def unregister(self, type: str | EventType, handler: Callable):
        """注销事件处理函数"""
        if handler in self._handlers[type]:
            self._handlers[type].remove(handler)

    def _run(self):
        """后台消费者线程"""
        while self._active:
            try:
                event = self._queue.get(timeout=1.0)
                self._process(event)
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"事件引擎循环异常: {e}", exc_info=True)

    def _process(self, event: Event):
        """分发事件"""
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(
                        f"处理事件 {event.type} 发生错误: {e}", exc_info=True
                    )

    def _run_timer(self):
        """后台定时器线程 (每秒触发)"""
        while self._active:
            sleep(1)
            # [变更点 2] 使用枚举创建事件
            event = Event(type=EventType.TIMER)
            self.put(event)
