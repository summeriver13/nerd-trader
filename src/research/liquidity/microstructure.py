"""
src/research/liquidity/microstructure.py
第 19 章微观结构特征：用 1 分钟棒估「交易一只标的实际要付多少成本」。

为什么需要这一层：本仓所有策略的裁决都被成本卡住，而成本一直是**常数**——费率表上的
5bp 手续费加 5bp 假设滑点（`src/core/costs.py` 的 `PERP_FEES.taker_net()`）。滑点是唯一
可以用数据检验的部分，而它恰好决定「反转/价差类策略到底能不能做」。

口径：
- 一天一行/一标的，量纲统一到 bp：既能和常数成本直接对照，也能直接丢进既有分位组合管线
  （`FactorSpec.compute` 返回它即可）。
- 只用 OHLCV：买卖方向由 BVC（bulk volume classification）按收益标准差定，不做逐笔推断——
  1m 棒没有成交方向标记，这是原文对高频数据的做法，不是权宜。
- 价差类估计给的是**往返**价差（买在卖价、卖在买价，相对中间价合计 = 一个价差），故单边
  成本 = 半个价差。
- 估不出时返回 nan，不截到 0：Roll 在一阶自协方差为正时无解（有动量的标的估不出价差），
  把这种情况当成「零成本」正好是本研究最怕的偏差。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BPS: Final[float] = 1e4
# BVC 的方向用**前向**滚动 σ（12 小时窗口）：用当天的 σ 就是同一天内的未来信息
SIGMA_WINDOW: Final[int] = 720
MIN_SIGMA_BARS: Final[int] = 60
# 少于 4 小时的交易日不聚合：半天行情估不出日内价差，硬算只给噪声
MIN_BARS_PER_DAY: Final[int] = 240
VPIN_BUCKETS: Final[int] = 50
MICRO_COLUMNS: Final[tuple[str, ...]] = (
    "bars",
    "notional_musd",
    "ret_bp",
    "cs_spread_bp",
    "roll_spread_bp",
    "kyle_lambda_bp",
    "amihud_bp",
    "vpin",
)


@dataclass(frozen=True)
class MicroOptions:
    """日频聚合口径。"""

    sigma_window: int = SIGMA_WINDOW
    min_bars: int = MIN_BARS_PER_DAY
    vpin_buckets: int = VPIN_BUCKETS


def roll_spread_bp(returns_bp: np.ndarray) -> float:
    """Roll(1984) 有效价差（往返、bp）：s = 2√(−cov(r_t, r_{t−1}))。

    cov ≥ 0 时无解（价差会估成虚数）→ nan。原文假设中间价随机游走、价差造成负的一阶
    自相关；动量强的标的这一条件不成立，用 nan 表示「这段估不出」而不是 0。
    """
    values = np.asarray(returns_bp, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    now, lag = values[1:], values[:-1]
    cov = float(np.mean(now * lag) - np.mean(now) * np.mean(lag))
    if not np.isfinite(cov) or cov >= 0.0:
        return np.nan
    return 2.0 * float(np.sqrt(-cov))


def corwin_schultz_spread_bp(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Corwin-Schultz(2012) 高低价价差（逐日、bp，首日为 nan）。

    β 是两日高低幅平方和、γ 是两日合并高低幅平方：α = (√(2β)−√β)/(3−2√2)
    − √(γ/(3−2√2))，S = 2(e^α−1)/(1+e^α)。α < 0 时按原文取 0（价差不能为负）——这会让
    中位数偏保守，但比丢掉一半样本更接近原文口径。
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if high.size != low.size or high.size < 2:
        return np.full(max(high.size, 0), np.nan)
    hl2 = np.log(high / low) ** 2
    beta = hl2 + np.roll(hl2, 1)
    gamma = np.log(np.fmax(high, np.roll(high, 1)) / np.fmin(low, np.roll(low, 1))) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(alpha < 0.0, 0.0, spread) * BPS
    spread[0] = np.nan
    return spread


def bulk_volume_split(
    returns_bp: np.ndarray, volume: np.ndarray, sigma_bp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """BVC：把每根棒的成交量拆成买/卖两份，方向由 Φ(r/σ) 定。

    σ 必须是**前向**滚动值（调用方保证），否则方向里就混进了未来的波动信息。
    """
    from scipy.stats import norm  # 局部导入：模块其余部分不依赖 scipy

    returns_bp = np.asarray(returns_bp, dtype=float)
    volume = np.asarray(volume, dtype=float)
    ratio = np.divide(
        returns_bp,
        np.asarray(sigma_bp, dtype=float),
        out=np.zeros_like(returns_bp),
        where=np.isfinite(sigma_bp) & (np.asarray(sigma_bp) > 0.0),
    )
    buy_share = np.where(np.isfinite(ratio), norm.cdf(np.nan_to_num(ratio)), 0.5)
    buy = volume * buy_share
    return buy, volume - buy


def kyle_lambda_bp(returns_bp: np.ndarray, signed_musd: np.ndarray) -> float:
    """Kyle(1985) λ：r_bp = λ·(有符号名义, $M) + ε 的最小二乘斜率（bp / $M）。

    直读法：成交 $1M 把价格推 λ 个 bp。不用带截距的回归——无截距时的斜率等价于
    去均值后的协方差比，漂移被吸收进残差而不是算成冲击。
    """
    x = np.asarray(signed_musd, dtype=float)
    y = np.asarray(returns_bp, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return np.nan
    x, y = x[keep], y[keep]
    var = float(np.mean(x * x) - np.mean(x) ** 2)
    if var <= 0.0:
        return np.nan
    return float((np.mean(x * y) - np.mean(x) * np.mean(y)) / var)


def amihud_bp(returns_bp: np.ndarray, notional_musd: float) -> float:
    """Amihud(2002) 非流动性：日内 |r| 均值 / 成交名义（bp / $M）。"""
    values = np.asarray(returns_bp, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or not np.isfinite(notional_musd) or notional_musd <= 0.0:
        return np.nan
    return float(np.mean(np.abs(values)) / notional_musd)


def vpin(buy: np.ndarray, sell: np.ndarray, volume: np.ndarray) -> float:
    """VPIN：等量桶内 |V_buy − V_sell| / 桶量，对**满桶**取均值。

    桶量按当天总量均分；最后一桶通常不满，计进去会把毒性算虚高，故丢弃。
    """
    buy = np.asarray(buy, dtype=float)
    sell = np.asarray(sell, dtype=float)
    volume = np.asarray(volume, dtype=float)
    total = float(np.sum(volume))
    if not np.isfinite(total) or total <= 0.0 or volume.size < 4:
        return np.nan
    chunk = total / VPIN_BUCKETS
    bucket = np.clip(
        np.floor(np.cumsum(volume) / chunk).astype(int), 0, VPIN_BUCKETS - 1
    )
    # 桶量必须按 volume 加权：只数根数会与 chunk（体积单位）量纲不符，判满桶恒为假
    size = np.bincount(bucket, weights=volume, minlength=VPIN_BUCKETS)
    net = np.abs(
        np.bincount(bucket, weights=buy, minlength=VPIN_BUCKETS)
        - np.bincount(bucket, weights=sell, minlength=VPIN_BUCKETS)
    )
    full = size >= chunk * 0.999
    if not bool(full.any()):
        return np.nan
    return float(np.sum(net[full]) / np.sum(size[full]))


def _nanmedian(values: np.ndarray) -> float:
    """全 NaN 时返回 nan（np.nanmedian 会给 RuntimeWarning，本仓 pytest 把警告当异常）。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else np.nan


def daily_microstructure(
    bars: pd.DataFrame, options: MicroOptions | None = None
) -> pd.DataFrame:
    """把一只标的的 1m 棒聚合成日频微观结构表（一天一行，索引为 UTC 日）。

    列见 `MICRO_COLUMNS`；`notional_musd` 单位 $M，`kyle_lambda_bp` 与 `amihud_bp` 都是
    「每 $M 名义」口径，可直接乘交易规模换算成 bp。
    """
    options = options or MicroOptions()
    empty = pd.DataFrame({name: pd.Series(dtype=float) for name in MICRO_COLUMNS})
    frame = bars.loc[:, ["high", "low", "close", "volume"]].astype(float)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    values = frame.to_numpy(dtype=float)
    keep = np.isfinite(values).all(axis=1) & (values[:, :3] > 0.0).all(axis=1)
    values = values[keep]
    # 桩里 DatetimeIndex 没有 normalize/floor，用 .dt 访问器（已用探针核过）
    dates = pd.Series(frame.index[keep]).dt.normalize()
    if values.shape[0] < options.min_bars:
        return empty

    high, low, close, volume = (values[:, i] for i in range(4))
    ret_bp = np.concatenate(([np.nan], np.diff(np.log(close)))) * BPS
    sigma = (
        pd.Series(ret_bp)
        .rolling(options.sigma_window, min_periods=MIN_SIGMA_BARS)
        .std()
        .to_numpy()
    )
    buy, sell = bulk_volume_split(ret_bp, volume, sigma)
    signed_musd = (buy - sell) * close / 1e6
    # CS 在 **1 分钟**尺度上估：日级高低幅里价差早被波动淹没（实测 BTC 半数以上 α<0
    # 被截成 0，等于报「零成本」），1 分钟尺度才能量到价差本身。跨日的那一对窗口
    # （每天首根）占比 1/1440，不另处理
    cs_1m = corwin_schultz_spread_bp(high, low)
    codes, uniques = pd.factorize(dates)
    # 唯一日是升序的（frame 已排序），后面用位置切片取索引，不构造 Timestamp 标量
    day_index = pd.DatetimeIndex(uniques)
    starts = np.searchsorted(codes, np.arange(len(day_index)))

    rows: list[dict[str, float]] = []
    kept: list[int] = []
    for i in range(len(day_index)):
        lo = int(starts[i])
        hi = int(starts[i + 1]) if i + 1 < len(starts) else codes.size
        chunk = slice(lo, hi)
        if hi - lo < options.min_bars:
            continue
        notional = float(np.sum(volume[chunk] * close[chunk]) / 1e6)
        kept.append(i)
        rows.append(
            {
                "bars": float(hi - lo),
                "notional_musd": notional,
                "ret_bp": float(close[hi - 1] / close[lo] - 1.0) * BPS,
                "cs_spread_bp": _nanmedian(cs_1m[chunk]),
                "roll_spread_bp": roll_spread_bp(ret_bp[chunk]),
                "kyle_lambda_bp": kyle_lambda_bp(ret_bp[chunk], signed_musd[chunk]),
                "amihud_bp": amihud_bp(ret_bp[chunk], notional),
                "vpin": vpin(buy[chunk], sell[chunk], volume[chunk]),
            }
        )
    if not rows:
        return empty
    out = pd.DataFrame(rows, index=day_index[kept])
    return out.loc[:, list(MICRO_COLUMNS)]


def load_microstructure(
    symbols: list[str] | None = None,
    *,
    root: Path | None = None,
    market: str = "swap",
    options: MicroOptions | None = None,
) -> pd.DataFrame:
    """读 1m 缓存（`<root>/okx/<market>/1m/<标的>.parquet`）聚合成长表。

    长表列为 date / symbol / 各微观结构量；没有 1m 缓存的标的直接跳过（本仓只有 30 个
    标的有 1m，见 README 第十八节）。
    """
    from src.config.paths import CRYPTO_ROOT
    from src.research.factors.panel import SOURCE_VENUE

    base = (root or CRYPTO_ROOT) / SOURCE_VENUE / market / "1m"
    files = sorted(base.glob("*.parquet"))
    if symbols is not None:
        wanted = set(symbols)
        files = [path for path in files if path.stem in wanted]
    frames: list[pd.DataFrame] = []
    for path in files:
        bars = pd.read_parquet(path)
        daily = daily_microstructure(bars, options)
        if daily.empty:
            continue
        daily = daily.copy()
        daily["symbol"] = path.stem
        frames.append(daily)
        logger.info("微观结构：%s 聚合 %d 天", path.stem, daily.shape[0])
    if not frames:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in MICRO_COLUMNS})
    merged = pd.concat(frames).rename_axis("date").reset_index()
    return merged.loc[:, ["date", "symbol", *MICRO_COLUMNS]]
