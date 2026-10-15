"""数据源公共部分：抽象基类与共享 HTTP 传输（代理 + 重试）。

本机对境外交易所（Polymarket、Deribit、OKX）**直连不通**（实测），必须走代理；代理切节点
会让 SSL 直接断，所以取数统一走 `get_json()` 的重试循环，不再各自手写 requests 调用。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, cast

import pandas as pd
import requests

from src.config.settings import load_config

REQUEST_TIMEOUT = 60.0
RETRY_ATTEMPTS = 5
RETRY_BACKOFF = 1.5
HTTP_OK = 200
HTTP_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class DataSourceError(RuntimeError):
    """取数失败（网络/代理/接口异常），与「该标的确实没有数据」区分开"""


class SourceHTTPError(RuntimeError):
    """非 200 响应，携带状态码供调用方降级（如 400 = 请求窗口过长）。"""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} {url}: {body[:200]}")
        self.status = status


def parse_instant(raw: Any, label: str) -> pd.Timestamp:
    """上游时间字段 → Timestamp；NaT/非法值当场报错，不把坏时间往下游传。

    类型上的 `Timestamp | NaTType` 是 pandas 的声明事实，不 cast 会让每一处调用点
    都得自己处理 NaT；这里把它收成一处，换来下游「拿到就是好时间」的不变量。
    """
    stamp = pd.Timestamp(raw)
    if stamp is pd.NaT:
        raise ValueError(f"{label} 不是有效时间：{raw!r}")
    return cast(pd.Timestamp, stamp)


def create_session() -> requests.Session:
    """沿用 OKX 的代理开关：这些端点本机直连不通（实测），必须走代理。"""
    session = requests.Session()
    config = load_config()
    if config.use_proxy and config.proxy_url:
        session.proxies.update({"http": config.proxy_url, "https": config.proxy_url})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = RETRY_ATTEMPTS,
) -> Any:
    """带重试的 GET；代理切节点会让 SSL 直接断（实测频繁），故只对可重试状态码重来。"""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as error:
            last_error = error
        else:
            if response.status_code == HTTP_OK:
                return response.json()
            error = SourceHTTPError(response.status_code, url, response.text[:200])
            if response.status_code not in HTTP_RETRY_STATUSES:
                raise error
            last_error = error
        time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"{url} 连续 {attempts} 次请求失败") from last_error


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
