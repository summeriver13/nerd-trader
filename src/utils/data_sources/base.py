from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSourceError(RuntimeError):
    """取数失败（网络/代理/接口异常），与「该标的确实没有数据」区分开"""


class BaseDataSource(ABC):
    @property
    def label(self) -> str:
        """数据源标签，用于缓存目录名与报错信息"""
        return self.__class__.__name__

    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """核心抓取接口"""
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """符号标准化"""
        return symbol
