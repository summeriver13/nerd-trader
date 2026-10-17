"""meme 币行情体检：四张表的口径与实现。

## 为什么这组标的需要单独体检

meme 币的收益分布不是「高风险版的 BTC」，是**彩票分布**：12 只里只有 4 只为正，
中位总收益 −69.7%、中位最大回撤 −96.3%。这类分布下**均值毫无意义**（少数翻倍币把
均值抬得远高于中位），所以每条结论都要中位与均值、为正占比并列读。

## 四张表

1. `profile_table`：逐只的收益/波动/回撤/偏度/峰度——先看清分布形状。
2. `lifecycle_table`：按首根日线分段，看衰减发生在哪一段（meme 的周期以天计）。
3. `benchmark_table`：与 BTC 的重叠日收益——「放大版的 BTC」还是独立行情。
4. `burst_table`：暴涨暴跌之后是动量还是反转，以及波动是否聚集。

## 口径（错了不会报错，只会让结论变好看的部分）

- **首根日线当起点**：11/12 只的 OKX 上市日与首根日线偏差 < 1 天，但 DOGE 的
  2020-01-01 是缓存起点（OKX 日线数据起点），不是上市日——读 DOGE 的分段时记住这点。
- **剔除零成交的冻结日**：冻结日的 OHLC 是上一笔的复制，没有真实成交，留着会在分段
  边界伪造出巨幅涨跌（carry 风险研究里实测过 ADA 的 +113% 伪涨幅）。
- **相关性只用重叠窗口**：各标的起点不同，只有重叠区间才是可比样本；把缺失日当
  0 收益会稀释 beta，把「独立行情」读成「跟跌」。
- **暴涨暴跌的次日收益从 t+1 起**：事件日自己的涨跌不能算进自己的前瞻收益。
- **偏度与峰度都是样本口径**：峰度是超额峰度（Fisher），正态分布记 0。
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    ANNUAL_DAYS,
    annualize,
    bars_per_year,
    max_drawdown_from_equity,
)
from src.research.meme.meme_bars import MIN_OBSERVATIONS, CoinBars

# 生命周期分段的边界（根）：第 i 段是 [EDGES[i], EDGES[i+1])，最后一段开到数据末尾。
# 标签是事件时间（首根日线后第几天，1 起算）：从第 s 根收盘到第 e 根收盘，持有 s+1..e 天
LIFECYCLE_EDGES: tuple[int, ...] = (0, 7, 30, 90)
LIFECYCLE_LABELS: tuple[str, ...] = ("1-7", "8-30", "31-90", "91+")
# 「暴涨/暴跌」的门槛：单日 ±10%。meme 的日波动中位在 5% 量级，10% 已是尾部
BURST_THRESHOLD = 0.10
# 暴涨暴跌之后的观察窗口（根）
BURST_WINDOWS: tuple[int, ...] = (1, 5)


def profile_table(bars: Mapping[str, CoinBars]) -> pd.DataFrame:
    """表一：逐只的收益/波动/回撤/偏度/峰度（按合约）——先从形状读起。

    收益率用日历跨度年化（不是根数）：缺口标的的根数少于天数，按根数年化会高估年化收益。
    """
    rows: list[dict[str, float | str]] = []
    for symbol, coin in bars.items():
        ret = coin.returns
        total = float(coin.close.iloc[-1] / coin.close.iloc[0] - 1.0)
        rows.append(
            {
                "symbol": symbol,
                "bars": float(coin.bars),
                "years": coin.span_days / ANNUAL_DAYS,
                "total_return": total,
                "annual_return": annualize(
                    total, coin.span_days, periods_per_year=ANNUAL_DAYS
                ),
                "annual_vol": float(ret.std(ddof=1))
                * float(np.sqrt(bars_per_year(coin.close.index))),
                "max_drawdown": max_drawdown_from_equity(coin.close),
                "from_high": float(coin.close.iloc[-1] / coin.close.max() - 1.0),
                "skew": float(ret.skew()),
                "kurt": float(ret.kurt()),
                "up_share": float((ret > 0).mean()),
                "best_day": float(ret.max()),
                "worst_day": float(ret.min()),
            }
        )
    table = pd.DataFrame(rows).set_index("symbol")
    return table.sort_values("total_return", ascending=False)


def _segment_return(coin: CoinBars, start: int, stop: int | None) -> float:
    """从第 start 根收盘持有到第 stop 根收盘的收益；不够根数返回 nan。

    stop=None 表示最后一段（开到数据末尾）。
    """
    if start >= coin.bars:
        return float("nan")
    end = coin.bars - 1 if stop is None else stop
    if end <= start or end >= coin.bars:
        return float("nan")
    return float(coin.close.iloc[end] / coin.close.iloc[start] - 1.0)


def lifecycle_table(bars: Mapping[str, CoinBars]) -> pd.DataFrame:
    """表二：生命周期分段收益。

    每段**独立**计算（不是累计）：累计口径会把「每段都在跌」读成「越跌越少」，
    两者的数字完全不同但都长得像一条收益曲线。
    """
    boundaries = (*LIFECYCLE_EDGES, None)
    rows: list[dict[str, float | str]] = []
    for label, start, stop in zip(
        LIFECYCLE_LABELS, boundaries[:-1], boundaries[1:], strict=True
    ):
        values = np.array(
            [_segment_return(coin, start, stop) for coin in bars.values()], dtype=float
        )
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "segment": label,
                "count": float(finite.size),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "positive": float((finite > 0).mean()) if finite.size else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def _beta(coin: np.ndarray, base: np.ndarray) -> float:
    """一元回归斜率；base 方差为 0 时返回 nan（不是 0，0 会被读成「无关联」）。"""
    if coin.size < MIN_OBSERVATIONS or float(np.var(base, ddof=1)) == 0.0:
        return float("nan")
    return float(np.cov(coin, base, ddof=1)[0, 1] / np.var(base, ddof=1))


def _corr(coin: np.ndarray, base: np.ndarray) -> float:
    """相关系数；任一侧方差为 0 时返回 nan。

    死币（价格横住）的重叠窗口方差正好是 0，`np.corrcoef` 在那里除零丢出
    RuntimeWarning——在 `filterwarnings = error` 的测试配置下直接变成失败。
    """
    if coin.size < MIN_OBSERVATIONS:
        return float("nan")
    if float(coin.std(ddof=1)) == 0.0 or float(base.std(ddof=1)) == 0.0:
        return float("nan")
    return float(np.corrcoef(coin, base)[0, 1])


def benchmark_table(bars: Mapping[str, CoinBars], benchmark: CoinBars) -> pd.DataFrame:
    """表三：与基准的重叠日收益。

    beta 拆成 BTC 上涨日/下跌日两段：meme 的「跌得多、涨得少」如果成立，会在
    beta_down > beta_up 上体现，而这在合起来的单一 beta 里是看不见的。
    """
    base_returns = benchmark.returns
    rows: list[dict[str, float | str]] = []
    for symbol, coin in bars.items():
        joined = pd.concat(
            {"coin": coin.returns, "base": base_returns}, axis=1, join="inner"
        ).dropna()
        coin_ret = joined["coin"].to_numpy(dtype=float)
        base_ret = joined["base"].to_numpy(dtype=float)
        beta = _beta(coin_ret, base_ret)
        corr = _corr(coin_ret, base_ret)
        up = base_ret > 0
        down = base_ret < 0
        alpha_daily = float(coin_ret.mean() - beta * base_ret.mean())
        rows.append(
            {
                "symbol": symbol,
                "days": float(coin_ret.size),
                "corr": corr,
                "beta": beta,
                "beta_up": _beta(coin_ret[up], base_ret[up]),
                "beta_down": _beta(coin_ret[down], base_ret[down]),
                "alpha_annual": alpha_daily * ANNUAL_DAYS,
                "r2": corr**2,
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def _autocorr(values: np.ndarray, lag: int) -> float:
    """滞后 lag 的样本自相关（正 = 动量型，负 = 反转型）。"""
    if values.size <= lag + MIN_OBSERVATIONS:
        return float("nan")
    head, tail = values[:-lag], values[lag:]
    if float(head.std(ddof=1)) == 0.0 or float(tail.std(ddof=1)) == 0.0:
        return float("nan")
    return float(np.corrcoef(head, tail)[0, 1])


def _forward_return(
    returns: np.ndarray, positions: np.ndarray, window: int
) -> np.ndarray:
    """事件日之后 window 根的复利收益；从事件日的下一根起算，事件日自己不算。

    末尾不够 window 根的样本丢弃，不拿半截窗口凑数（半截窗口的系统性偏短会把
    「暴涨后继续涨」读成「暴涨后回落」）。
    """
    keep = positions[positions + window < returns.size]
    if keep.size == 0:
        return np.empty(0, dtype=float)
    path = returns[keep[:, None] + np.arange(1, window + 1)]
    return np.prod(1.0 + path, axis=1) - 1.0


def _median(values: np.ndarray) -> float:
    """中位；空样本返回 nan。"""
    return float(np.median(values)) if values.size else float("nan")


def burst_table(bars: Mapping[str, CoinBars]) -> pd.DataFrame:
    """表四：暴涨暴跌的自相关与后验收益。

    `base_t1` 是无条件基准（全样本次日中位）：没有它，「暴涨后次日 +0.5%」无法判断
    是动量还是本来就有的漂移。
    """
    rows: list[dict[str, float | str]] = []
    for symbol, coin in bars.items():
        values = coin.returns.to_numpy(dtype=float)
        up = np.flatnonzero(values >= BURST_THRESHOLD)
        down = np.flatnonzero(values <= -BURST_THRESHOLD)
        rows.append(
            {
                "symbol": symbol,
                "acf1": _autocorr(values, 1),
                "abs_acf1": _autocorr(np.abs(values), 1),
                "base_t1": _median(_forward_return(values, np.arange(values.size), 1)),
                "up_n": float(up.size),
                "up_t1": _median(_forward_return(values, up, 1)),
                "up_t5": _median(_forward_return(values, up, 5)),
                "down_n": float(down.size),
                "down_t1": _median(_forward_return(values, down, 1)),
                "down_t5": _median(_forward_return(values, down, 5)),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")
