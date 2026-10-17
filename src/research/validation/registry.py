"""
src/research/validation/registry.py
标的预设与策略注册表：菜单、CLI、编排都从这里取，避免各处各写一份映射。
"""

from __future__ import annotations

from typing import Any

from src.research.strategies.donchian_trend import DonchianTrendStrategy
from src.research.strategies.rsrs_trend_follower import RSRSTrendStrategy
from src.research.strategies.vector_bollinger_mr import BollingerMRStrategy
from src.research.strategies.vector_dual_ma import DualMAStrategy
from src.research.strategies.vector_trend_reversion import TrendReversionStrategy

# === 菜单与 CLI 共用的默认值 ===
DEFAULT_SYMBOL = "c1"
DEFAULT_TIMEFRAME = "1d"
DEFAULT_BARS = 730  # 评估区间根数（1d 下约两年）
DEFAULT_STRATEGY = "4"
DEFAULT_LEVERAGE = 1.0
# maker 挂单耐心：实测（139 标的 × 4 策略）耐心 1 根的净效应最好、方差最小，
# 拉长耐心不增加成本节省却放大择时偏差，故取 1。
DEFAULT_MAKER_PATIENCE = 1

# === 稳健性检验参数 ===
ROBUST_DEFAULT_BARS = 2900  # 默认取满可得历史（1d 下约 8 年）
DEFAULT_SEGMENTS = 4
DEFAULT_ROBUST_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
CARRY_DEFAULT_DAYS = 2900  # 资金费率取满可得历史（Binance BTC 自 2019-09 起）

# === 快捷预设菜单 ===
SYMBOL_PRESETS: dict[str, dict[str, tuple[str, str]]] = {
    "Crypto": {
        "c1": ("BTC/USDT", "比特币 (Bitcoin)"),
        "c2": ("ETH/USDT", "以太坊 (Ethereum)"),
    },
}

# 策略别名：CLI 里用可读名字，映射到菜单序号
STRATEGY_ALIASES: dict[str, str] = {
    "dual-ma": "1",
    "trend-reversion": "2",
    "bollinger": "3",
    "rsrs": "4",
    "donchian": "5",
}

# 策略短名：控制台表格的列宽需要，长中文名会破坏对齐；含基准，故不只列可选策略
STRATEGY_SHORT_NAMES: dict[str, str] = {
    "buy-and-hold": "BuyHold",
    "1": "DualMA",
    "2": "TrendRev",
    "3": "Bollinger",
    "4": "RSRS",
    "5": "Donchian",
}

# === 策略映射表 ===
STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "1": {
        "name": "自适应双均线 (Dual MA)",
        "class": DualMAStrategy,
        "default_params": {"trend_period": 120, "bull_fast": 10, "bull_slow": 30},
    },
    "2": {
        "name": "趋势+反转缝合怪 (Trend+Reversion)",
        "class": TrendReversionStrategy,
        "default_params": {"trend_period": 200, "bull_fast": 10},
    },
    "3": {
        "name": "布林带均值回归 (Bollinger MR)",
        "class": BollingerMRStrategy,
        "default_params": {
            "window": 20,
            "entry_std": 2.0,
            "exit_std": 0.5,
            "stop_loss_std": 4.0,
            "trend_ma_period": 200,
        },
    },
    "4": {
        "name": "RSRS 阻力支撑相对强度 (RSRS Trend)",
        "class": RSRSTrendStrategy,
        "default_params": {
            "rsrs_n": 18,
            "rsrs_m": 600,
            "buy_threshold": 0.7,
            "sell_threshold": -0.7,
            "max_leverage": 1.0,
            "stop_loss_mult": 2.5,
        },
    },
    "5": {
        "name": "唐奇安突破 (Donchian)",
        "class": DonchianTrendStrategy,
        "default_params": {"entry_period": 55, "exit_period": 20},
    },
}


def resolve_strategy(choice: str) -> dict[str, Any] | None:
    """把菜单序号或可读别名解析成策略配置。"""
    return STRATEGY_MAP.get(STRATEGY_ALIASES.get(choice.lower(), choice))


def resolve_symbol(code: str) -> str:
    """把预设键（c1/c2）解析成交易对，非预设值原样返回。"""
    for items in SYMBOL_PRESETS.values():
        if code.lower() in items:
            return items[code.lower()][0]
    return code.upper()


def resolve_symbols(argument: str | None) -> list[str]:
    """--symbol 支持逗号分隔多个标的，缺省用默认标的集合。"""
    if not argument:
        return list(DEFAULT_ROBUST_SYMBOLS)
    return [resolve_symbol(code.strip()) for code in argument.split(",")]
