"""
src/utils/data_sources/stock_cn.py
A股数据源实现：支持 Tushare, Baostock 和 AkShare
代码位置：src/utils/data_sources/stock_cn.py
遵循规范：Ruff + Python 3.12 + Pyright Strict
"""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime
from typing import Final, Generator, cast

import numpy as np
import pandas as pd

# 内部模块导入
from config.settings import GLOBAL_CONFIG

from .base import BaseDataSource

# 动态导入处理，配合类型检查
try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import baostock as bs
except ImportError:
    bs = None

# 配置日志
logger = logging.getLogger(__name__)

# 定义常量
DEFAULT_FREQ: Final[str] = "1d"
COLUMN_PRICE: Final[pd.Index] = pd.Index(["open", "high", "low", "close", "volume"])


class TushareSource(BaseDataSource):
    """Tushare 数据源实现。"""

    def __init__(self) -> None:
        token = getattr(GLOBAL_CONFIG, "TUSHARE_TOKEN", "")
        # 避免硬编码字符串检查
        if ts and token and token != "YOUR_TOKEN_HERE":
            ts.set_token(token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
            logger.warning("Tushare Token 未配置或库未安装。")

    def normalize_symbol(self, symbol: str) -> str:
        """归一化代码格式 (如 600519.SH)。"""
        s = symbol.strip().upper()
        if "." in s:
            return s
        if s.startswith(("5", "6")):
            return f"{s}.SH"
        if s.startswith(("0", "1", "3")):
            return f"{s}.SZ"
        return f"{s}.BJ"

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if ts is None or self.pro is None or timeframe != DEFAULT_FREQ:
            return pd.DataFrame()

        ts_code = self.normalize_symbol(symbol)
        asset_type = "FD" if self.is_etf(symbol) else "E"

        logger.info(f"正在从 Tushare 获取 {ts_code} K线数据...")

        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                adj="qfq",
                asset=asset_type,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 链式处理
            df_processed = (
                df.rename(columns={"trade_date": "datetime", "vol": "volume"})
                .assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
                .set_index("datetime")
                .sort_index()
            )

            # 类型对齐与时区处理
            df_processed.index = cast(pd.DatetimeIndex, df_processed.index).tz_localize(
                "UTC"
            )
            # 显式转换数值列
            df_processed[COLUMN_PRICE] = df_processed[COLUMN_PRICE].apply(
                pd.to_numeric, errors="coerce"
            )

            return cast(pd.DataFrame, df_processed[COLUMN_PRICE])

        except Exception as e:
            logger.error(f"Tushare 异常 [{ts_code}]: {e}")
            return pd.DataFrame()


class BaostockSource(BaseDataSource):
    """Baostock 数据源实现：提供估值指标。"""

    def normalize_symbol(self, symbol: str) -> str:
        """归一化代码格式 (如 sh.600519)。"""
        code = symbol.lower().split(".")[0]
        prefix = "sh" if code.startswith(("5", "6")) else "sz"
        return f"{prefix}.{code}"

    @contextlib.contextmanager
    def _bs_session(self) -> Generator[None, None, None]:
        """上下文管理器自动处理登录注销。"""
        if bs is None:
            yield
            return

        login_res = bs.login()
        if login_res.error_code != "0":
            logger.error(f"Baostock 登录失败: {login_res.error_msg}")
            yield
            return

        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                bs.logout()

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if bs is None:
            return pd.DataFrame()

        bs_code = self.normalize_symbol(symbol)
        freq_map: dict[str, str] = {"1d": "d", "5m": "5", "1h": "60"}

        if timeframe not in freq_map:
            logger.warning(f"Baostock 不支持周期: {timeframe}")
            return pd.DataFrame()

        with self._bs_session():
            fields = (
                "date,open,high,low,close,volume,peTTM,pbMRQ"
                if timeframe == DEFAULT_FREQ
                else "date,time,open,high,low,close,volume"
            )

            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                frequency=freq_map[timeframe],
                adjustflag="2",
            )

            # 严谨校验结果集
            if rs is None or rs.error_code != "0":
                err_msg = getattr(rs, "error_msg", "Unknown Error")
                logger.error(f"Baostock 查询失败 [{bs_code}]: {err_msg}")
                return pd.DataFrame()

            data_list: list[list[str]] = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            # 使用 pd.Index 包装 fields 解决 Pyright 警告
            df = pd.DataFrame(data_list, columns=pd.Index(rs.fields))
            return self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """数值与时间解析逻辑。"""
        df = df.copy()

        # 核心价格列转换
        df[COLUMN_PRICE] = df[COLUMN_PRICE].apply(pd.to_numeric, errors="coerce")

        # 估值指标转换
        if "peTTM" in df.columns:
            df["pe"] = pd.to_numeric(df["peTTM"], errors="coerce")
        if "pbMRQ" in df.columns:
            df["pb"] = pd.to_numeric(df["pbMRQ"], errors="coerce")

        # 时间序列解析
        if "time" in df.columns and not df.empty and str(df["time"].iloc[0]):
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S000")
        else:
            df["datetime"] = pd.to_datetime(df["date"])

        df = df.set_index("datetime").sort_index()

        # 时区处理
        dti = cast(pd.DatetimeIndex, df.index)
        if dti.tz is None:
            df.index = dti.tz_localize("UTC")

        return cast(pd.DataFrame, df)


class AkShareSource(BaseDataSource):
    """AkShare 数据源实现。"""

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.split(".")[0]

    def fetch_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        if ak is None:
            return pd.DataFrame()

        code = self.normalize_symbol(symbol)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        period_map: dict[str, str] = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(timeframe)

        if not period:
            logger.warning(f"AkShare 周期不支持: {timeframe}")
            return pd.DataFrame()

        logger.info(f"正在从 AkShare 获取 {code} 数据...")

        try:
            # 接口选择
            if self.is_etf(symbol):
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period=period,
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period=period,
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )

            if df is None or df.empty:
                return pd.DataFrame()

            # 中文字段对齐映射
            rename_map: dict[str, str] = {
                "日期": "datetime",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }

            df_final = (
                df.rename(columns=rename_map)
                .assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
                .set_index("datetime")
                .sort_index()
            )

            # 时区与数值清洗
            dti = cast(pd.DatetimeIndex, df_final.index)
            df_final.index = dti.tz_localize("UTC")
            df_final[COLUMN_PRICE] = df_final[COLUMN_PRICE].apply(
                pd.to_numeric, errors="coerce"
            )

            return cast(pd.DataFrame, df_final[COLUMN_PRICE])

        except Exception as e:
            logger.error(f"AkShare 异常 [{code}]: {e}")
            return pd.DataFrame()
