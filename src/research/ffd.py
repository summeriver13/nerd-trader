"""de Prado《金融机器学习》第 5 章：分数阶差分（FFD）。

口径（改代码前先读）：

一、**只实现固定窗版本**（Snippet 5.2/5.4 的 `getWeights_FFD`/`fracDiff_FFD`）。原文先给的
   展开窗版本（Snippet 5.1/5.3）是作者随后自己否定的形式：权重按整段样本长度算，d<1 时
   权重衰减慢于可和条件，同一序列在不同时期的定义都不一样。更要命的是两个版本同名的
   `thres` 语义不同——展开窗是「累计权重质量」，固定窗是「单个权重的绝对值」——是原文最
   容易踩的坑，因此本模块不提供展开窗（不做双实现）。
二、权重递推 `w_0=1`、`w_k = -w_{k-1}(d-k+1)/k`，`|w_k| < threshold` 时截断；返回**滱后序**
   的一维数组（`w[k]` 对应滱后 k 根，与 `(1-B)^d` 的定义一致），于是直接与 `np.convolve`
   的 `mode="valid"` 配套：`valid[i]` 对应窗口末端 `i+width` 那一行。
   原文把权重反转后再与窗口切片点积（`np.dot(w.T, window)`），两者等价；但若**照抄反转
   又与 convolve 配对**，符号与时间方向会同时反过来（d=1 变成「未来减当前」），而对称核
   （d=2 的 [1,-2,1]）里两个错误正好抵消、看不出来——本仓的 d=1 恒等式就是这样抓到的。
三、**窗口宽度对阈值极敏感**：d<1 时 `|w_k| ~ k^(-d-1)`，宽度约 `threshold^(-1/(d+1))`。
   实测（d=0.4）：threshold 从 1e-1 到 1e-5 对应宽度 2 / 10 / 54 / 281 / 1457 根。原文默认
   `1e-5` 是按分钟级样本设计的（宽约 1457 根 ≈ 分钟数据 1 天）；搬到**日线**上就是四年历史，
   本仓日线面板每个标的中位历史只有 958 根、根本喂不满，故日线实用值是 1e-2 ~ 1e-3
   （见 `FIXED_WINDOW_THRESHOLD`）。
四、**截断还使权重之和不再为 0**（原文未提）：不截断时 `Σw = (1-1)^d = 0`，而固定窗把
   尾巴扔掉了。实测（d=0.4）：宽 2 → +0.48、宽 10 → +0.264、宽 54 → +0.136、宽 281 →
   +0.070；整数阶（d=1）恰好为 0。含义是**价格水平会泄漏进特征**——常数序列的 FFD 不是 0
   而是 `Σw × 常数`。所以在截面上用这个特征时：要么取更宽的窗，要么做去均值/标准化，并且
   始终把「与（对数）水平的截面相关」当作必看的诊断。
四、值只在「原始输入非缺」的行上产出（原文的 exclude NAs 口径）：先 ffill 再算，算完把原始
   缺失的行设回 NaN。注意 ffill 出来的陈旧值**会进入后续窗口**——这是原文口径，不是疏漏。
五、`minimum_d` 用原文 `plotMinFFD` 的判据：从 d=0 向右扫，**第一个 ADF 统计量低于 5% 临界
   值**的 d 即最小阶数；`adfuller` 固定 `maxlag=1`、`regression="c"`（原文如此，为跨 d 可比）。
   论文层面的意义：d>0 时权重绝对可和 ⇒ 序列平稳，但记忆随 d 增大而流失，故取「刚好平稳」
   的那个 d。
六、`memory_retention` 是原文的 `corr` 列：变换后序列与（对数）价格水平的相关系数，衡量
   「还剩多少水平信息」。它是粗糙的记忆代理——与水平相关不等于有用，必须再过预测层。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ADFullerResult, adfuller

# 日线口径的默认阈值：宽度 10 根 @ d=0.4（原书默认 1e-5 是分钟级口径，日线要 1457 根）
FIXED_WINDOW_THRESHOLD = 1e-2
# 权重个数的硬上限：阈值取小 10 倍、宽度约涨 5~10 倍，写错一个量级就会要十万根历史
MAX_WEIGHT_COUNT = 20_000
MAX_ORDER = 1.0  # minimum_d 的搜索上界（原文扫到 d=1）
ORDER_STEP = 0.05  # minimum_d 的搜索步长（比原文的 0.1 细一档）
ADF_MAXLAG = 1
ADF_REGRESSION = "c"
ADF_CRITICAL_KEY = "5%"


def _adfuller(series: pd.Series) -> tuple[float, float, float]:
    """调 statsmodels 的 ADF，返回（统计量、p 值、5% 临界值）。

    必须显式传 `result_object=True`：仓库的 pytest 把 `filterwarnings` 设成 error，而
    statsmodels 0.14 的默认行为会发 FutureWarning（长度可变的元组返回值即将被换成
    ADFullerResult）。

    两处 cast 都是 statsmodels 自家源文件的类型标注问题（它没有 .pyi 桩）：
    `autolag` 被标成 `str` 而运行时要靠 `None` 关闭自动选阶；返回值标注是多种元数的
    联合，没把 `result_object=True` 这条分支说出来。仓库规则禁止 `# pyright: ignore`。
    """
    result = cast(
        ADFullerResult,
        adfuller(
            series.to_numpy(dtype=float),
            maxlag=ADF_MAXLAG,
            regression=ADF_REGRESSION,
            autolag=cast(str, None),
            result_object=True,
        ),
    )
    return (
        float(result.statistic),
        float(result.pvalue),
        float(result.critical_values[ADF_CRITICAL_KEY]),
    )


def fractional_weights(
    d: float,
    threshold: float = FIXED_WINDOW_THRESHOLD,
    *,
    max_count: int = MAX_WEIGHT_COUNT,
) -> np.ndarray:
    """固定窗分数阶差分权重，滱后序（``w[k]`` 对应滱后 k 根，可直接喂给 ``np.convolve``）。"""
    if d < 0 or not np.isfinite(d):
        raise ValueError(f"阶数必须是非负有限数：d={d}")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError(f"阈值必须为正有限数：threshold={threshold}")
    weights = [1.0]
    while True:
        k = len(weights)
        nxt = -weights[-1] / k * (d - k + 1)
        if abs(nxt) < threshold:
            break
        if k >= max_count:
            raise ValueError(
                f"d={d}、threshold={threshold} 需要的窗口超过 {max_count} 根："
                f"宽度约 threshold**(-1/(d+1))，请放宽阈值或缩小 d"
            )
        weights.append(nxt)
    return np.array(weights, dtype=float)


def window_width(d: float, threshold: float = FIXED_WINDOW_THRESHOLD) -> int:
    """窗口宽度（= 输出序列开头的暖机行数）。"""
    return len(fractional_weights(d, threshold)) - 1


def fractional_difference(
    series: pd.Series,
    d: float,
    threshold: float = FIXED_WINDOW_THRESHOLD,
) -> pd.Series:
    """对一条序列做固定窗分数阶差分，索引不变，暖机段与原始缺失行都是 NaN。"""
    weights = fractional_weights(d, threshold)
    width = len(weights) - 1
    raw = series.to_numpy(dtype=float)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if width >= len(series):
        return result
    filled = series.ffill().to_numpy(dtype=float)
    result.iloc[width:] = np.convolve(filled, weights, mode="valid")
    result[~np.isfinite(raw)] = np.nan
    return result


def scan_orders(
    series: pd.Series,
    *,
    threshold: float = FIXED_WINDOW_THRESHOLD,
    max_d: float = MAX_ORDER,
    step: float = ORDER_STEP,
) -> pd.DataFrame:
    """原文 ``plotMinFFD`` 的读数表：逐 d 的 ADF 统计量、5% 临界值与水平记忆。

    列：``adf_stat``、``p_value``、``critical_5pct``、``memory``、``passed``；索引是 d。
    """
    orders = np.round(np.arange(0.0, max_d + step / 2, step), 6)
    rows: list[dict[str, float]] = []
    for d in orders:
        values = fractional_difference(series, float(d), threshold)
        sample = values.dropna()
        if len(sample) < MIN_ADF_OBSERVATIONS:
            rows.append(
                {
                    "adf_stat": np.nan,
                    "p_value": np.nan,
                    "critical_5pct": np.nan,
                    "memory": np.nan,
                    "passed": False,
                }
            )
            continue
        stat, p_value, critical_5pct = _adfuller(sample)
        rows.append(
            {
                "adf_stat": float(stat),
                "p_value": float(p_value),
                "critical_5pct": critical_5pct,
                "memory": memory_retention(series, float(d), threshold),
                "passed": bool(stat < critical_5pct),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(orders, name="d"))


def minimum_d(
    series: pd.Series,
    *,
    threshold: float = FIXED_WINDOW_THRESHOLD,
    max_d: float = MAX_ORDER,
    step: float = ORDER_STEP,
) -> float:
    """刚好通过 ADF 的最小阶数；扫到 ``max_d`` 都没通过就返回 NaN。"""
    table = scan_orders(series, threshold=threshold, max_d=max_d, step=step)
    passed = table.index[table["passed"].to_numpy(dtype=bool)]
    if not len(passed):
        return float("nan")
    return float(np.asarray(passed, dtype=float)[0])


def memory_retention(
    series: pd.Series, d: float, threshold: float = FIXED_WINDOW_THRESHOLD
) -> float:
    """变换后序列与原始（水平）序列的相关系数：还剩多少水平信息。"""
    values = fractional_difference(series, d, threshold)
    raw = series.to_numpy(dtype=float)
    valid = np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(raw)
    if valid.sum() < MIN_CORRELATION_OBSERVATIONS:
        return float("nan")
    return float(np.corrcoef(values.to_numpy(dtype=float)[valid], raw[valid])[0, 1])


# ADF 至少要有这么多个有效观测（低于这个数统计量没有意义）
MIN_ADF_OBSERVATIONS = 50
MIN_CORRELATION_OBSERVATIONS = 2
