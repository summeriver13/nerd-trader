"""
src/research/meme_bars.py
meme 池的行情载入层：标的清单、`CoinBars` 与本地缓存的读取。

从 `meme_profile.py` 拆出来的是**职责**而不是行数：那边是「四张表怎么算」，这边是
「数据从哪来、什么算脏」。口径改动只影响一侧（换池子/换交易所/改冻结日剔除规则）
不应该碰到另一侧，反之亦然。

缺口径上的坑（错了不报错、只让结论变好看）见 `meme_profile.py` 模块文档；这里只负责：
本地缓存逐只读、**剔除零成交的冻结日**（冻结日的 OHLC 是上一笔的复制，留着会在分段
边界伪造巨幅涨跌）、缺文件显式失败（静默少一只会让中位数悄悄移动）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.config.paths import CRYPTO_ROOT, MARKET_SPOT, crypto_dataset_dir
from src.research.factors.panel import SOURCE_VENUE

# 标的池：以太坊/Solana 生态的 meme + DOGE。MAGIC 是游戏币不是 meme，按已定口径纳入，
# 它是「上线即最高点」的另一类样本。
MEME_SYMBOLS: tuple[str, ...] = (
    "DOGE-USDT",
    "SHIB-USDT",
    "PEPE-USDT",
    "BONK-USDT",
    "FLOKI-USDT",
    "TRUMP-USDT",
    "MAGIC-USDT",
    "PUMP-USDT",
    "MEME-USDT",
    "ORDI-USDT",
    "BOME-USDT",
    "MEW-USDT",
)
# 对照标的：足够长的历史，且是所有加密资产的共同风险源
BENCHMARK = "BTC-USDT"
TIMEFRAME = "1d"
# 至少两个观测点才有方差
MIN_OBSERVATIONS = 2


@dataclass(frozen=True)
class CoinBars:
    """一只标的清洗后的日线收盘序列（已剔除零成交的冻结日）。"""

    symbol: str
    close: pd.Series

    @property
    def bars(self) -> int:
        """日线根数。"""
        return int(self.close.size)

    @property
    def returns(self) -> pd.Series:
        """日收益（简单收益），首日没有前值故不含。"""
        return self.close.pct_change().dropna()

    @property
    def span_days(self) -> int:
        """首根到末根的日历跨度（天），用于年化——根数会因缺口而高估年限。"""
        if self.bars < MIN_OBSERVATIONS:
            return 0
        # stub 把 DatetimeIndex 的元素标成 Index | tuple，运行时必是 Timestamp
        first = cast(pd.Timestamp, self.close.index[0])
        last = cast(pd.Timestamp, self.close.index[-1])
        return int((last - first).days)


def load_bars(
    symbols: Sequence[str] = MEME_SYMBOLS,
    *,
    venue: str = SOURCE_VENUE,
    market: str = MARKET_SPOT,
    timeframe: str = TIMEFRAME,
    root: Path = CRYPTO_ROOT,
) -> dict[str, CoinBars]:
    """读本地缓存；缺文件直接报错——少一只必须显式失败，不能静默变成 11 只。"""
    directory = crypto_dataset_dir(venue, market, timeframe, root=root)
    bars: dict[str, CoinBars] = {}
    missing: list[str] = []
    for symbol in symbols:
        path = directory / f"{symbol}.parquet"
        if not path.exists():
            missing.append(symbol)
            continue
        frame = pd.read_parquet(path)
        traded = frame[frame["volume"] > 0]
        bars[symbol] = CoinBars(
            symbol=symbol, close=cast(pd.Series, traded["close"]).astype(float)
        )
    if missing:
        raise FileNotFoundError(f"{directory} 下缺 {len(missing)} 只：{missing}")
    return bars


def load_universe(
    symbols: Sequence[str] = MEME_SYMBOLS,
    *,
    benchmark: str = BENCHMARK,
    venue: str = SOURCE_VENUE,
    market: str = MARKET_SPOT,
    timeframe: str = TIMEFRAME,
    root: Path = CRYPTO_ROOT,
) -> tuple[dict[str, CoinBars], CoinBars]:
    """读 meme 池与基准：前三个表只看池子，基准单独返回，不混进池子里。

    混在一起（把 BTC 放进池子）会让生命周期、暴涨暴跌的分组悄悄多出一只
    非 meme 标的，中位数跟着动。
    """
    bars = load_bars(
        (*symbols, benchmark),
        venue=venue,
        market=market,
        timeframe=timeframe,
        root=root,
    )
    return {symbol: bars[symbol] for symbol in symbols}, bars[benchmark]
