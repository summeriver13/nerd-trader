"""
多重检验校正第三族：White 的真实性检验（Reality Check）与 Hansen 的 SPA。

为什么本仓需要它：DSR / Bonferroni / PBO 之外还差一族工具——**「从一堆候选里挑最好的」
这件事本身该怎么算显著性**。Bonferroni 按最坏情况分配 α，候选一多门槛就高到拒绝一切；
DSR 只回答「一个夏普在 N 次试验后还算不算显著」；PBO 回答「挑出来的那个在样本外会不会掉队」。
White (2000) 与 Hansen (2005) 直接对**最大值统计量**做 bootstrap：候选之间的相关性由数据
自己说——相关越高、最大值的零分布越窄，门槛反而更合理，所以它比 Bonferroni 有力、又比
「把挑出来的那个单独做检验」诚实。

三处口径（前两处决定读数能不能信）：

1. **学生化**：Hansen 的 SPA 把每个候选的均值除以自身标准误。候选的波动量级差几个数量级时，
   不学生化的最大值统计量永远落在高波动的那只身上；White 的 RC 保留不学生化版作对照。
2. **平稳 bootstrap**（Politis-Romano）：几何块长、默认 `T^(1/3)`。金融序列有自相关，iid
   重采样会把零分布造得过窄、p 值偏小。
3. **重中心**：`upper` 把非正均值剪到 0（Hansen 的保守上界），`consistent` 只剪「明确为负」
   的候选（阈值 `A_T` 随样本增大收敛到 0），`plain` 不剪（White 的 RC）。

Benjamini-Hochberg 一并放在这里，但它只作**对照臂**：de Prado & Fabozzi (2026) 证明搜索-
选择过程下 FDR 无法从样本内统计量识别（README 第三节），所以它的用途是演示「同一批候选、
同一份 p 值时 BH 与 SPA 差多少」，不是决策依据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_REPLICATIONS: Final[int] = 1000
DEFAULT_SEED: Final[int] = 20261013
DEFAULT_ALPHA: Final[float] = 0.05
MIN_OBSERVATIONS: Final[int] = 30
BLOCK_ROOT: Final[float] = 1.0 / 3.0

_NORMAL: Final[NormalDist] = NormalDist()


@dataclass(frozen=True)
class RealityCheckResult:
    """White 真实性检验的读数：不学生化的最大值统计量。样本不足时数值字段为 nan。"""

    candidates: int
    observations: int
    block: int
    replications: int
    statistic: float
    p_value: float
    best: str
    statistics: pd.Series


@dataclass(frozen=True)
class SpaResult:
    """Hansen SPA 的读数。`p_consistent` 是主读数，`p_upper` 是保守上界。"""

    candidates: int
    observations: int
    block: int
    replications: int
    statistic: float
    p_consistent: float
    p_upper: float
    best: str
    statistics: pd.Series


@dataclass(frozen=True)
class FdrResult:
    """Benjamini-Hochberg 步进的读数：`cutoff` 是最后的 p 值门槛。"""

    alpha: float
    cutoff: float
    labels: tuple[str, ...]


def default_block(observations: int) -> int:
    """平稳 bootstrap 的默认平均块长 `T^(1/3)`：样本越长越允许放宽相关性假设。"""
    return max(1, int(round(observations**BLOCK_ROOT)))


def stationary_bootstrap_indices(
    length: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano 平稳 bootstrap 下标：每根以 `1/block` 概率另起一块，否则顺延一格。"""
    starts = rng.integers(0, length, size=length)
    restarts = rng.random(length) < 1.0 / max(block, 1)
    indices = np.empty(length, dtype=int)
    current = int(starts[0])
    for position in range(length):
        if position == 0 or restarts[position]:
            current = int(starts[position])
        else:
            current = (current + 1) % length
        indices[position] = current
    return indices


def _aligned(series: pd.DataFrame) -> pd.DataFrame:
    """丢掉任何候选缺值的期次：bootstrap 必须在同一个 T 上重采样。"""
    return series.dropna(how="any") if not series.empty else series


def _dispersion(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐候选的均值与标准误；零波动的候选标准误取 inf，学生化后 t 归零。"""
    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    return means, np.where(errors > 0.0, errors, np.inf)


def _recenter(
    means: np.ndarray, errors: np.ndarray, observations: int
) -> dict[str, np.ndarray]:
    """三种重中心：`plain` 不剪、`upper` 剪非正、`consistent` 只剪明确为负。"""
    threshold = math.sqrt(
        2.0 * max(math.log(math.log(observations)), 0.0) / observations
    )
    standardized = means / errors
    return {
        "plain": means,
        "upper": np.maximum(means, 0.0),
        "consistent": np.where(standardized < -threshold, 0.0, means),
    }


def _null_maxima(
    values: np.ndarray,
    centers: dict[str, np.ndarray],
    *,
    studentize: bool,
    block: int,
    replications: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """一次重采样同时给出三种重中心下的最大值零分布（避免跑三遍 bootstrap）。"""
    observations = values.shape[0]
    rng = np.random.default_rng(seed)
    nulls = {kind: np.empty(replications, dtype=float) for kind in centers}
    for index in range(replications):
        rows = stationary_bootstrap_indices(observations, block, rng)
        sample = values[rows]
        means, errors = _dispersion(sample)
        for kind, center in centers.items():
            if studentize:
                deviations = (means - center) / errors
            else:
                deviations = math.sqrt(observations) * (means - center)
            nulls[kind][index] = max(0.0, float(deviations.max()))
    return nulls


def _p_value(null: np.ndarray, statistic: float) -> float:
    """bootstrap p 值：+1 平滑，避免 0 与 1 的边界读数（有限次重采样的标准做法）。"""
    return float((1.0 + float((null >= statistic).sum())) / (len(null) + 1.0))


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


def reality_check(
    series: pd.DataFrame,
    *,
    block: int | None = None,
    replications: int = DEFAULT_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> RealityCheckResult:
    """White (2000) 真实性检验：`max_k √T·mean_k` 的 bootstrap 零分布。"""
    frame = _aligned(series)
    observations = len(frame)
    if observations < MIN_OBSERVATIONS or frame.shape[1] == 0:
        return RealityCheckResult(
            candidates=frame.shape[1],
            observations=observations,
            block=0,
            replications=replications,
            statistic=float("nan"),
            p_value=float("nan"),
            best="",
            statistics=_empty_series(),
        )
    values = frame.to_numpy(dtype=float)
    means, errors = _dispersion(values)
    statistics = math.sqrt(observations) * means
    statistic = max(0.0, float(statistics.max()))
    chosen = block or default_block(observations)
    nulls = _null_maxima(
        values,
        _recenter(means, errors, observations),
        studentize=False,
        block=chosen,
        replications=replications,
        seed=seed,
    )
    return RealityCheckResult(
        candidates=frame.shape[1],
        observations=observations,
        block=chosen,
        replications=replications,
        statistic=statistic,
        p_value=_p_value(nulls["plain"], statistic),
        best=str(frame.columns[int(np.argmax(statistics))]),
        statistics=pd.Series(statistics, index=frame.columns),
    )


def hansen_spa(
    series: pd.DataFrame,
    *,
    block: int | None = None,
    replications: int = DEFAULT_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> SpaResult:
    """Hansen (2005) SPA：学生化最大值检验，同时给一致版与保守上界两个 p 值。"""
    frame = _aligned(series)
    observations = len(frame)
    if observations < MIN_OBSERVATIONS or frame.shape[1] == 0:
        return SpaResult(
            candidates=frame.shape[1],
            observations=observations,
            block=0,
            replications=replications,
            statistic=float("nan"),
            p_consistent=float("nan"),
            p_upper=float("nan"),
            best="",
            statistics=_empty_series(),
        )
    values = frame.to_numpy(dtype=float)
    means, errors = _dispersion(values)
    statistics = means / errors
    statistic = max(0.0, float(statistics.max()))
    chosen = block or default_block(observations)
    nulls = _null_maxima(
        values,
        _recenter(means, errors, observations),
        studentize=True,
        block=chosen,
        replications=replications,
        seed=seed,
    )
    return SpaResult(
        candidates=frame.shape[1],
        observations=observations,
        block=chosen,
        replications=replications,
        statistic=statistic,
        p_consistent=_p_value(nulls["consistent"], statistic),
        p_upper=_p_value(nulls["upper"], statistic),
        best=str(frame.columns[int(np.argmax(statistics))]),
        statistics=pd.Series(statistics, index=frame.columns),
    )


def naive_p_values(series: pd.DataFrame) -> pd.Series:
    """逐候选的**单侧** p 值（正态近似，检验「均值为正」）：不做搜索校正的对照臂。

    单侧是为了与 RC / SPA 同向——那两个检验问的也是「最好的那个是不是真的为正」，
    用双侧 p 值会让对照臂偏保守、把 BH 的对比做得不公平。
    """
    frame = _aligned(series)
    if frame.shape[1] == 0:
        return _empty_series()
    means, errors = _dispersion(frame.to_numpy(dtype=float))
    p_values = [_NORMAL.cdf(-float(value)) for value in means / errors]
    return pd.Series(p_values, index=frame.columns)


def benjamini_hochberg(p_values: pd.Series, alpha: float = DEFAULT_ALPHA) -> FdrResult:
    """BH 步进：接受最大的 k 使 `p_(k) ≤ k/m·α`，返回被拒的候选名。"""
    clean = p_values.dropna()
    count = len(clean)
    if count == 0:
        return FdrResult(alpha=alpha, cutoff=0.0, labels=())
    ordered = clean.sort_values()
    values = ordered.to_numpy(dtype=float)
    passing = [
        rank
        for rank, value in enumerate(values, start=1)
        if value <= rank / count * alpha
    ]
    if not passing:
        return FdrResult(alpha=alpha, cutoff=0.0, labels=())
    last = passing[-1]
    names = ordered.index.to_numpy()[:last]
    return FdrResult(
        alpha=alpha,
        cutoff=float(last / count * alpha),
        labels=tuple(str(name) for name in names),
    )
