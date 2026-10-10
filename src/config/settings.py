import os
from dataclasses import dataclass
from typing import Any

import ccxt
from dotenv import load_dotenv

# 加载 .env
load_dotenv()


@dataclass
class ExchangeConfig:
    api_key: str | None
    secret: str | None
    passphrase: str | None
    use_proxy: bool
    proxy_url: str | None
    sandbox_mode: bool  # 新增：是否开启模拟盘

    def to_ccxt_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.passphrase,
            "enableRateLimit": True,
            # 单次请求超时（毫秒）：没有超时会把抓取循环挂死
            "timeout": 20000,
            "options": {"defaultType": "swap"},  # 默认交易永续合约
        }

        # 代理配置
        if self.use_proxy and self.proxy_url:
            config["proxies"] = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }

        return config


def load_config() -> ExchangeConfig:
    return ExchangeConfig(
        api_key=os.getenv("OKX_API_KEY"),
        secret=os.getenv("OKX_SECRET"),
        passphrase=os.getenv("OKX_PASSPHRASE"),
        use_proxy=os.getenv("OKX_USE_PROXY", "false").lower() == "true",
        proxy_url=f"http://127.0.0.1:{os.getenv('OKX_PROXY_PORT', '7890')}",
        # 读取 SANDBOX 环境变量，默认为 False (实盘)
        sandbox_mode=os.getenv("OKX_SANDBOX", "false").lower() == "true",
    )


# 全局配置单例
GLOBAL_CONFIG = load_config()


def create_exchange(exchange_id: str = "okx") -> Any:
    """按配置构造 ccxt 客户端。

    OKX 的模拟盘不换域名，靠 set_sandbox_mode 打 x-simulated-trading 头，
    所以开关必须在这里生效——否则 OKX_SANDBOX=true 会被静默忽略、直接打实盘。

    ccxt 基类把 fetch_* 声明成抛 NotImplementedError（返回 Never），
    静态分析会把调用之后的代码全判成不可达，故返回 Any。
    """
    exchange: Any = getattr(ccxt, exchange_id)(GLOBAL_CONFIG.to_ccxt_config())
    if GLOBAL_CONFIG.sandbox_mode:
        exchange.set_sandbox_mode(True)
    return exchange


def create_data_exchange(exchange_id: str = "okx") -> Any:
    """构造行情取数客户端：不带凭据、不打沙盒头。

    OKX 的模拟环境（x-simulated-trading 头）返回的是另一套历史 K 线：实测
    2023-07-02 沙盒给 H=36162247.1/C=133653.3、实盘给 H=30800.4/C=30615.6；
    而且沙盒 feed 是整体微扰（1d BTC 与实盘逐根一致仅 4/20，中位偏差 0.71bp）
    且只回溯到 2022-05-27（实盘可到 2019-01）。行情取数带沙盒头会静默污染缓存。

    ccxt 基类把 fetch_* 声明成抛 NotImplementedError（返回 Never），
    静态分析会把调用之后的代码全判成不可达，故返回 Any。
    """
    config = GLOBAL_CONFIG.to_ccxt_config()
    # 公开行情不需要凭据；带 key 走实盘会触发 AuthenticationError
    config.update({"apiKey": None, "secret": None, "password": None})
    return getattr(ccxt, exchange_id)(config)
