import os
from dataclasses import dataclass
from typing import Any

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
