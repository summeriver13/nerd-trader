"""
src/research/factors/panel.py
截面面板：从自持的 crypto 缓存构建宽表（时间 × 标的）与逐日可交易掩码。

为什么单独一层：
1. 单标的引擎吃长表，截面研究要的是对齐后的宽表；
2. 因子结论的第一杀手是池子口径——池子规则只能有一处定义，研究脚本不得就地过滤；
3. 掩码必须点内时间：用「今天的成交额」筛「三年前的标的」就是未来信息，这类错误
   不会报错，只会安静地把结论变好看。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analytics.metrics import bars_per_year
from src.config.paths import CRYPTO_ROOT, MARKET_SPOT

logger = logging.getLogger(__name__)

# === 主池口径：规模与统计效力的平衡点（实测 289 个缓存标的中留下约 100 个）===
MIN_NOTIONAL_USD: float = 1_000_000.0
MIN_LISTED_DAYS: int = 180
LIQUIDITY_WINDOW: int = 30

# === 非加密仪器 ===
# OKX 的代币化股票/ETF/商品以 X<标的代码>-USDT 上线（实测 2026-07-16 起共 42 个），
# 它们的共同因子是股指/商品而不是加密市场，混入截面会直接污染波动率与动量类因子。
TOKENIZED_PREFIX: str = "X"
# X 前缀但确实是加密资产；新增这类标的必须显式加进来，否则会被前缀规则误剔
CRYPTO_X_PREFIX: frozenset[str] = frozenset({"XRP-USDT", "XTZ-USDT", "XPL-USDT"})
# 外汇与测试仪器：零成交，任何口径下都不可交易
NON_MARKET: frozenset[str] = frozenset({"AUDF-USDT", "AUDM-USDT", "XTESTA-USDT"})

# 缓存目录里的交易所名（自持数据按 <venue>/<market>/<timeframe>/<标的>.parquet 组织）
SOURCE_VENUE: str = "okx"
# 退市标的补集所在的数据集：OKX 对退市标的**不保留任何历史**（instruments 只列
# live/preopen、history-candles 返回 51001），实测缓存标的与 OKX 官方退市名单零交集。
# 该数据集由 scripts/fetch_delisted_bars.py 从 Binance 回填（那里退市标的的日线仍可用）。
DELISTED_VENUE: str = "binance"


@dataclass(frozen=True)
class UniverseRule:
    """可交易池口径：全项目唯一出处。"""

    min_notional_usd: float = MIN_NOTIONAL_USD
    min_listed_days: int = MIN_LISTED_DAYS
    liquidity_window: int = LIQUIDITY_WINDOW
    keep_non_crypto: bool = False

    def describe(self) -> str:
        """报告表头用的一行口径说明。"""
        scope = "含非加密仪器" if self.keep_non_crypto else "剔除非加密仪器"
        return (
            f"≥${self.min_notional_usd / 1e6:.1f}M/日（{self.liquidity_window} 日中位）"
            f" | 上市 ≥{self.min_listed_days} 天 | {scope}"
        )


@dataclass(frozen=True)
class Panel:
    """宽表面板：行 = 时间，列 = 标的。"""

    timeframe: str
    close: pd.DataFrame
    volume: pd.DataFrame
    notional: pd.DataFrame
    # 池子规则静态剔除的标的（时间无关的口径），报告里必须打印
    excluded: tuple[str, ...] = ()

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns

    @property
    def bars_per_year(self) -> float:
        """每年根数：复用 analytics 口径，避免与回测各算一套。"""
        return bars_per_year(self.index)

    def returns(self) -> pd.DataFrame:
        """逐根收益（crypto 7×24 无缺口，直接 pct_change）。

        显式关掉 pad：面板故意用 NaN 表示「未上市/未开盘」，默认的向前填充会
        把缺口变成 0% 收益，从而系统性低估波动率——而波动率类因子正是本库的主力。
        """
        return self.close.pct_change(fill_method=None)


def load_panel(
    timeframe: str,
    rule: UniverseRule | None = None,
    root: Path = CRYPTO_ROOT,
    *,
    market: str = MARKET_SPOT,
    include_delisted: bool = False,
) -> Panel:
    """读本地缓存构建面板，并按池子规则做静态剔除（只读，不触发采集）。

    静态剔除是与时间无关的口径（代币化股票、外汇、测试仪器）；与时间有关的条件
    （流动性、上市天数）在 tradable_mask 里逐日判定。默认读现货：池子过滤规则
    （X 前缀代币化股票等）与现货缓存的口径一致。

    `include_delisted` 打开时并入退市标的补集（见 DELISTED_VENUE）：只用 okx 缓存
    做截面研究，等于默认「买过的小币都不会死」，小市值类结论会被系统性抬高。
    补集与主池按构造互斥（同一批标的不会既在 OKX 交易又已从 OKX 退市）；若某标的
    被 OKX 重新上架而两边都有，以主池行情为准。
    """
    rule = rule or UniverseRule()
    venue_dir = Path(root) / SOURCE_VENUE / market / timeframe
    if not venue_dir.is_dir():
        raise FileNotFoundError(
            f"{venue_dir} 不存在：先跑一次回测把 {timeframe} 行情抓到本地"
        )

    frames = _read_venue(venue_dir)
    if include_delisted:
        delisted_dir = Path(root) / DELISTED_VENUE / market / timeframe
        frames = {**_read_venue(delisted_dir), **frames}
    if not frames:
        raise FileNotFoundError(f"{venue_dir} 下没有非空 parquet")

    excluded: tuple[str, ...] = ()
    if not rule.keep_non_crypto:
        kept = {sym: frame for sym, frame in frames.items() if not _is_non_crypto(sym)}
        excluded = tuple(sorted(set(frames) - set(kept)))
        frames = kept

    return Panel(
        timeframe=timeframe,
        close=_wide(frames, "close"),
        volume=_wide(frames, "volume"),
        notional=_wide(frames, "close") * _wide(frames, "volume"),
        excluded=excluded,
    )


def listing_age_days(panel: Panel) -> pd.DataFrame:
    """每个标的距其首根 K 线的天数（上市天数代理，逐日展开）。

    首根 K 线之前留 NaN（「还没上市」没有年龄），不做负数填充：负数会被读成
    「上市了 N 天」，而 NaN 在任何比较里都是 False，方向安全。
    """
    first = panel.close.apply(pd.Series.first_valid_index)
    age = pd.DataFrame(
        {sym: (panel.index - first[sym]).days for sym in panel.symbols},
        index=panel.close.index,
    )
    return age.where(panel.close.notna())


def tradable_mask(panel: Panel, rule: UniverseRule | None = None) -> pd.DataFrame:
    """逐日可交易掩码：只用当日及以前的信息（点内时间）。

    流动性用**满窗口**的中位名义额：窗口不满即视为不可交易，宁可让池子在标的
    新上市时更窄，也不要用半截窗口的名义额把它放进来。
    """
    rule = rule or UniverseRule()
    liquidity = panel.notional.rolling(
        rule.liquidity_window, min_periods=rule.liquidity_window
    ).median()
    age = listing_age_days(panel)
    return (
        (age >= rule.min_listed_days)
        & (liquidity >= rule.min_notional_usd)
        & panel.close.notna()
    )


def tradable_counts(mask: pd.DataFrame) -> pd.Series:
    """逐日可交易标的数（面板宽度）：截面 N 有多宽，读结论时就要看这条曲线。"""
    return mask.sum(axis=1)


def _read_venue(directory: Path) -> dict[str, pd.DataFrame]:
    """读一个数据集目录下的全部非空 parquet；目录缺失返回空（补集可能还没抓）。"""
    if not directory.is_dir():
        return {}
    return {
        path.stem: frame
        for path in sorted(directory.glob("*.parquet"))
        if not (frame := pd.read_parquet(path)).empty
    }


def _is_non_crypto(symbol: str) -> bool:
    """代币化股票/外汇/测试仪器的判定（前缀 + 白名单，见模块常量）。"""
    if symbol in NON_MARKET:
        return True
    return symbol.startswith(TOKENIZED_PREFIX) and symbol not in CRYPTO_X_PREFIX


def _wide(frames: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """长表字典 → 宽表；各标的起止日期不同，缺失点留 NaN。"""
    return pd.DataFrame({sym: frame[field] for sym, frame in frames.items()})
