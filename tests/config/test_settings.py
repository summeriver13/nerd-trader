"""settings 的客户端构造契约：行情取数绝不能带沙盒头或凭据。

OKX 的模拟环境（x-simulated-trading 头）返回的是另一套历史 K 线，实测沙盒 feed
是整体微扰（1d BTC 与实盘逐根一致仅 4/20）且只回溯到 2022-05-27（实盘到 2019-01），
用它取行情会静默污染缓存、作废全部回测结论。
"""

import ccxt
import pytest

from src.config.settings import GLOBAL_CONFIG, create_data_exchange, create_exchange

SANDBOX_HEADER = "x-simulated-trading"


def _headers(exchange: ccxt.Exchange) -> dict[str, str]:
    """ccxt 把 headers 标成 dict | None，取值前先归一化。"""
    return exchange.headers or {}


@pytest.fixture
def sandbox_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GLOBAL_CONFIG, "sandbox_mode", True)


def test_data_exchange_never_uses_sandbox(sandbox_on: None) -> None:
    """行情取数走实盘：即使配置开了模拟盘，也不许打沙盒头。"""
    exchange: ccxt.Exchange = create_data_exchange("okx")
    try:
        assert SANDBOX_HEADER not in _headers(exchange)
        assert not exchange.apiKey
        assert not exchange.secret
    finally:
        exchange.close()


def test_trading_exchange_honors_sandbox(sandbox_on: None) -> None:
    """下单客户端仍按配置走模拟盘（reconcile_execution / measure_maker_fill 依赖它）。"""
    exchange: ccxt.Exchange = create_exchange("okx")
    try:
        assert _headers(exchange).get(SANDBOX_HEADER) == "1"
    finally:
        exchange.close()
