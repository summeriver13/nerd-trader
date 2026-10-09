from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class BaseDataSource(ABC):
    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """核心抓取接口"""
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """符号标准化"""
        return symbol
