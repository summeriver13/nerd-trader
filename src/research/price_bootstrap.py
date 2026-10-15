"""价格路径块 bootstrap：用自持 1h 收益重抽样求触碰/末值分布（物理测度参照）。

与 `src/research/bootstrap.py` 无关：那是回测样本索引的 bagging/CPCV，这里是**价格路径**
重抽样。用途是给阈值阶梯一个「同样的行情再来一次会怎样」的第三口径，与市场价、
Deribit 风险中性价并列。

口径要点：
- 块重抽样（默认 24h）保住日内自相关与波动聚集；块长 24h/168h 结论一致（已测）；
- 默认去漂移（physical 漂移不能当风险中性漂移用），`demean=False` 保留历史漂移；
- `vol_target` 按目标波动率缩放收益，用来检验「尾部更肥」是不是波动率水平的伪影；
- 分块模拟：`paths × hours` 一次性建矩阵在 20000×1874 时是 300MB，必须切块。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config.paths import CRYPTO_ROOT, MARKET_SPOT, crypto_dataset_dir

SOURCE_VENUE = "binance"
TIMEFRAME = "1h"
DEFAULT_SYMBOL = "BTCUSDT"
HOURS_PER_YEAR = 24 * 365
DAYS_PER_YEAR = 365.0

DEFAULT_PATHS = 20_000
DEFAULT_BLOCK_HOURS = 24
DEFAULT_SEED = 11
CHUNK_PATHS = 2_000

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapSpec:
    """一次模拟的规模与口径；`hours` 是路径步数（1h 一根）。"""

    hours: int
    paths: int = DEFAULT_PATHS
    block_hours: int = DEFAULT_BLOCK_HOURS
    seed: int = DEFAULT_SEED
    demean: bool = True
    vol_target: float | None = None


@dataclass(frozen=True)
class PathStats:
    """每条路径相对起点的对数极值与终点；概率由这三个数组算，路径本身不保留。"""

    max_log: FloatArray
    min_log: FloatArray
    end_log: FloatArray

    def touch_rate(self, barrier_log: float) -> float:
        """触碰比例：上行取 max、下行取 min，方向由障碍的相对位置决定。"""
        if barrier_log > 0.0:
            return float((self.max_log >= barrier_log).mean())
        return float((self.min_log <= barrier_log).mean())

    def terminal_rate(self, threshold_log: float, *, below: bool = False) -> float:
        """末值比例：默认 `P(S_T > K)`，below=True 给 `P(S_T < K)`。"""
        if below:
            return float((self.end_log <= threshold_log).mean())
        return float((self.end_log > threshold_log).mean())


def realized_vol(returns: FloatArray, *, hours_per_year: int = HOURS_PER_YEAR) -> float:
    """小时收益的年化波动率（样本标准差，ddof=1）。"""
    if returns.size < 2:
        raise ValueError("样本不足两根，算不出波动率")
    return float(returns.std(ddof=1) * math.sqrt(hours_per_year))


def hourly_returns(
    symbol: str = DEFAULT_SYMBOL,
    *,
    root: Path = CRYPTO_ROOT,
    lookback_days: int | None = None,
) -> FloatArray:
    """自持 1h 收盘价（ADR-0006）算对数收益；缺文件直接报错，不静默换源。"""
    path = crypto_dataset_dir(SOURCE_VENUE, MARKET_SPOT, TIMEFRAME, root=root)
    frame = pd.read_parquet(path / f"{symbol}.parquet")
    close = pd.Series(frame["close"]).astype(float)
    if lookback_days is not None:
        close = close.iloc[-lookback_days * 24 :]
    return np.diff(np.log(close.to_numpy(dtype=float)))


def _adjust(returns: FloatArray, spec: BootstrapSpec) -> FloatArray:
    """去漂移与波动率目标：先减均值（若要求），再按目标波动率整体缩放。"""
    adjusted = returns - returns.mean() if spec.demean else returns - 0.0
    if spec.vol_target is not None:
        adjusted = adjusted * (spec.vol_target / realized_vol(returns))
    return adjusted


def simulate_paths(returns: FloatArray, spec: BootstrapSpec) -> PathStats:
    """块重抽样出 spec.paths 条路径，只留每条的 max/min/end。"""
    if returns.size < spec.block_hours:
        raise ValueError(f"样本 {returns.size} 根不足一个块（{spec.block_hours} 根）")
    sample = _adjust(returns, spec)
    rng = np.random.default_rng(spec.seed)
    blocks = math.ceil(spec.hours / spec.block_hours)
    offsets = np.arange(spec.block_hours)
    starts_ceiling = sample.size - spec.block_hours + 1
    maxima: list[FloatArray] = []
    minima: list[FloatArray] = []
    ends: list[FloatArray] = []
    for begin in range(0, spec.paths, CHUNK_PATHS):
        count = min(CHUNK_PATHS, spec.paths - begin)
        starts = rng.integers(0, starts_ceiling, size=(count, blocks))
        picked = sample[starts[:, :, None] + offsets[None, None, :]]
        cumulative = np.cumsum(picked.reshape(count, -1)[:, : spec.hours], axis=1)
        # 起点本身也是路径的一部分（障碍正好在起点时即刻触碰）
        maxima.append(np.maximum(cumulative.max(axis=1), 0.0))
        minima.append(np.minimum(cumulative.min(axis=1), 0.0))
        ends.append(cumulative[:, -1])
    return PathStats(
        max_log=np.concatenate(maxima),
        min_log=np.concatenate(minima),
        end_log=np.concatenate(ends),
    )
