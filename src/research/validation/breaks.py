"""结构突变检测：SADF（de Prado《金融机器学习》第 17 章）。

核心是 **SADF**（Supremum Augmented Dickey-Fuller，Phillips-Shi-Yu 2015）：在每个
时点上，对所有可能的起点取 ADF 统计量的**上确界**。标准 ADF 是左尾检验（原假设
单位根、备择平稳），SADF 用**右尾**——统计量越大说明序列越「爆炸」，即越偏离随机
游走向上发散。

**为什么这对我们有用**：反转与动量都只看「过去涨了多少」，分不出**可持续的趋势**
和**不可持续的爆炸**。SADF 是统计意义上的爆炸性——理论上能把两者分开，因为爆炸
是均值回归的、趋势是持续的。

**口径上的四个决定**：

1. 手写 ADF 回归而不用 `statsmodels.adfuller`：SADF 要跑几十万次回归，`adfuller`
   每次调用的开销（参数校验、结果对象构造）占九成以上。正确性靠单窗口对拍保证
   （`tests/research/test_breaks.py`，与 `adfuller` 逐位一致）。
2. 用**前缀和**把每个窗口降到 O(1)：`X'X`、`X'y`、`y'y` 都是可累加的量，先整体
   算一遍再按窗口相减，避免每个窗口重建设计矩阵。实测 n=2400 从 35 秒降到 1 秒。
3. 设计矩阵**整体**按列标准化一次。t 值对解释变量的缩放不变，所以全局标准化与
   逐窗口标准化等价，但前者能让前缀和直接复用（BTC 的价格列是 6 万量级，不标准化
   会丢精度）。
4. 临界值用蒙特卡洛自算（随机游走下 SADF 的分布），不硬编码别人的表——表是别人
   的样本长度与窗口下的，我们换了参数就不适用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd

MIN_ROWS: Final[int] = 10
"""ADF 回归至少要这么多行，否则自由度不足、t 值无意义。"""

DEFAULT_MIN_WINDOW: Final[int] = 60
"""SADF 的最短窗口（观测数）：窗口太短时 ADF 的势极低，全是被噪声撑起来的假爆炸。"""

DEFAULT_LAGS: Final[int] = 1
"""ADF 的滞后阶数：留一阶差分项吸收序列相关。"""

DEFAULT_STEP: Final[int] = 5
"""起点网格步长：上确界对起点连续变化不敏感，粗网格换几十倍速度。"""

DEFAULT_QUANTILE: Final[float] = 0.95
"""蒙特卡洛临界值取的分位（右尾 5%）。"""

DEFAULT_REPLICATIONS: Final[int] = 200
"""蒙特卡洛重复次数。"""

SINGULAR_TOLERANCE: Final[float] = 1e-12
"""Gram 行列式的下限：低于它当奇异处理（局部完全无波动的窗口会让差分列退化）。"""


@dataclass(frozen=True)
class SadfOptions:
    """SADF 的三个参数。"""

    min_window: int = DEFAULT_MIN_WINDOW
    lags: int = DEFAULT_LAGS
    step: int = DEFAULT_STEP

    def validate(self) -> None:
        """参数自检：窗口太短会让 ADF 退化成噪声。"""
        if self.min_window < MIN_ROWS:
            raise ValueError(f"min_window 至少 {MIN_ROWS}，收到 {self.min_window}")
        if self.lags < 0:
            raise ValueError(f"lags 不能为负，收到 {self.lags}")
        if self.step < 1:
            raise ValueError(f"step 至少 1，收到 {self.step}")


class AdfSweep:
    """一次构造、反复查询的 ADF t 值。

    记 `d[j] = levels[j+1] - levels[j]`。对 `j` 从 `lags` 起回归：

        d[j] = a + b·levels[j] + Σ c_i·d[j-i]

    第二列（**水平项**）的系数 `b` 就是爆炸性的载体：原假设 `b=0`（单位根），
    `b>0` 表示向上发散。窗口用**水平下标** [start, end] 指定，对应行区间
    `[start, end-1-lags]`。
    """

    def __init__(self, levels: np.ndarray, lags: int) -> None:
        self.lags = lags
        diff = np.diff(levels)
        size = diff.size
        columns = [np.ones(size - lags), levels[lags:size]]
        columns.extend(diff[lags - i : size - i] for i in range(1, lags + 1))
        design = np.column_stack(columns)
        self.response = diff[lags:]
        # 整体标准化一次即可：t 值对解释变量的缩放不变
        scale = np.sqrt((design * design).sum(axis=0))
        scale[scale == 0.0] = 1.0
        self.design = design / scale
        self._gram = _prefix_outer(self.design)
        self._cross = _prefix_rows(self.design * self.response[:, None])
        self._squares = np.concatenate(([0.0], np.cumsum(self.response**2)))

    @property
    def columns(self) -> int:
        """回归的解释变量个数（含常数项与全部滞后差分）。"""
        return self.lags + 2

    def tstat(self, start: int, end: int) -> float:
        """水平项系数的 t 值；样本不足或矩阵奇异时返回 NaN。"""
        rows = end - self.lags - start
        if rows <= self.columns:
            return math.nan
        return float(
            _batched_tstat(
                (self._gram[end - self.lags] - self._gram[start])[None, ...],
                (self._cross[end - self.lags] - self._cross[start])[None, ...],
                np.array([self._squares[end - self.lags] - self._squares[start]]),
                np.array([rows]),
                self.columns,
            )[0]
        )

    def sup_tstat(self, end: int, starts: np.ndarray) -> float:
        """同一端点下对一组起点取 t 值的**上确界**（批量求解，SADF 的热循环）。

        全部起点共享端点，所以 Gram 只需算 `prefix[end] - prefix[starts]`——一次
        广播相减加一次批量 solve，替掉逐个窗口的 Python 循环。
        """
        rows = end - self.lags - starts
        keep = rows > self.columns
        if not bool(keep.any()):
            return math.nan
        stats = _batched_tstat(
            self._gram[end - self.lags] - self._gram[starts[keep]],
            self._cross[end - self.lags] - self._cross[starts[keep]],
            self._squares[end - self.lags] - self._squares[starts[keep]],
            rows[keep],
            self.columns,
        )
        finite = stats[np.isfinite(stats)]
        return float(finite.max()) if finite.size else math.nan


def _batched_tstat(
    gram: np.ndarray,
    cross: np.ndarray,
    total: np.ndarray,
    rows: np.ndarray,
    columns: int,
) -> np.ndarray:
    """批量 OLS：一组窗口的水平项 t 值，奇异窗口返回 NaN。

    `total` 是各窗口的 `y'y`。残差平方和用 `y'y - 2·β'X'y + β'(X'X)β` 算，
    避免为每个窗口重新做一次回归。
    """
    result = np.full(gram.shape[0], np.nan)
    keep = np.abs(np.linalg.det(gram)) > SINGULAR_TOLERANCE
    if not bool(keep.any()):
        return result
    matrices, vectors = gram[keep], cross[keep]
    beta = np.linalg.solve(matrices, vectors[..., None])[..., 0]
    covariance = np.linalg.inv(matrices)
    residual = (
        total[keep]
        - 2.0 * np.einsum("ij,ij->i", beta, vectors)
        + np.einsum("ij,ijk,ik->i", beta, matrices, beta)
    )
    variance = (residual / (rows[keep] - columns)) * covariance[:, 1, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        result[keep] = np.where(variance > 0.0, beta[:, 1] / np.sqrt(variance), np.nan)
    return result


def _prefix_outer(design: np.ndarray) -> np.ndarray:
    """逐行外积的前缀和：查询任一窗口的 `X'X` 只需相减。"""
    return np.concatenate(
        (
            np.zeros((1, design.shape[1], design.shape[1])),
            np.cumsum(_outers(design), axis=0),
        )
    )


def _outers(design: np.ndarray) -> np.ndarray:
    """逐行外积 `x·x'`，形状 (rows, cols, cols)。"""
    return design[:, :, None] * design[:, None, :]


def _prefix_rows(values: np.ndarray) -> np.ndarray:
    """逐行向量的前缀和。"""
    return np.concatenate((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))


def adf_tstat(levels: np.ndarray, lags: int = DEFAULT_LAGS) -> float:
    """整段序列的 ADF t 值，与 `statsmodels.adfuller(..., regression="c")` 同口径。"""
    if levels.size < MIN_ROWS + lags:
        return math.nan
    return AdfSweep(levels, lags).tstat(0, levels.size - 1)


def sadf_series(levels: pd.Series, options: SadfOptions | None = None) -> pd.Series:
    """逐端点的 SADF：对全部可行起点取 ADF t 值的上确界。

    前 `min_window` 个点没有足够的历史，返回 NaN——**这不是缺陷**，是这个统计量的
    定义域：没有窗口就没有检验。
    """
    settings = options or SadfOptions()
    settings.validate()
    values = levels.to_numpy(dtype=float)
    sweep = AdfSweep(values, settings.lags)
    result = np.full(values.size, math.nan)
    for end in range(settings.min_window, values.size):
        starts = np.arange(0, end - settings.min_window + 1, settings.step)
        result[end] = sweep.sup_tstat(end, starts)
    return pd.Series(result, index=levels.index, name="sadf")


def monte_carlo_critical(
    length: int,
    options: SadfOptions | None = None,
    replications: int = DEFAULT_REPLICATIONS,
    quantile: float = DEFAULT_QUANTILE,
    seed: int = 0,
) -> float:
    """随机游走下 SADF 的分位临界值（Phillips-Shi-Yu 的做法）。

    原假设是单位根，所以临界值必须由**随机游走**的分布定，不能拿真实数据的分位
    当门槛——真实数据里到处是爆炸，用自己的分位等于「比过去更爆炸才算爆炸」。
    """
    settings = options or SadfOptions()
    settings.validate()
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal((replications, length)), axis=1)
    peaks = np.array(
        [np.nanmax(sadf_series(pd.Series(row), settings).to_numpy()) for row in walk]
    )
    return float(np.nanquantile(peaks, quantile))


def explosive_mask[SADFLike: (pd.Series, pd.DataFrame)](
    sadf: SADFLike, critical: float
) -> SADFLike:
    """SADF 超过临界值的时点（右尾）：统计意义上的爆炸期。

    面板与单标的共用：SADF 在面板上就是逐列同算，没必要分两个函数。
    """
    return cast(SADFLike, sadf > critical)
