"""
src/research/entropy.py
de Prado 第 18 章：熵特征——信息论视角的「可预测性」。

四个估计量（单位 bit/根，越大越不可预测）：

- `shannon_bits`：等频分箱后的香农熵
- `ordinal_bits`：顺序（排列）熵，对厚尾稳健、不用分箱
- `lempel_ziv_bits`：LZ76 子串计数归一成熵率（1 ≈ 随机，0 ≈ 规律）
- `kontoyiannis_bits`：最长匹配熵率，**只在前缀里找匹配**

后两个是序列类估计量，在**二值化**（默认中位数）后的序列上算，并在窗口内做同分布**置换对照**：
`*_excess` = 本序列 − 同窗置换的均值。这一步是必需的——这两个估计量在小样本下系统性偏低
（常数序列的 Kontoyiannis 读数在 n=100 时仍约 0.3，而真实熵率为 0），扣掉置换基线后才只剩
「序列特有的结构」，跨标的可比。

口径（踩过的坑）：

- Kontoyiannis 若在整个序列里找匹配就等于用了未来数据，会把熵压低；这里严格只用前缀。
- 样本量必须一起看：少于 `MIN_SAMPLES` 直接缺失，不做外推。
- 常数序列熵为 0（不是缺失）；等频分箱遇到重复分位点时箱数变少，是数据离散而非缺陷。
- 只用**历史**窗口：第 t 天的特征是 [t−window+1, t] 的读数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.research.factors.panel import Panel

DEFAULT_BINS: int = 10
DEFAULT_ORDER: int = 3
DEFAULT_WINDOW: int = 100
MIN_SAMPLES: int = 30
SURROGATE_ROUNDS: int = 3
SURROGATE_SEED: int = 0

FEATURE_COLUMNS: tuple[str, ...] = (
    "shannon_bits",
    "ordinal_bits",
    "lz_bits",
    "lz_excess",
    "konto_bits",
    "konto_excess",
)


@dataclass(frozen=True, slots=True)
class EntropyOptions:
    """熵特征参数：窗口长度、分箱数、顺序熵阶数、置换对照次数。"""

    window: int = DEFAULT_WINDOW
    bins: int = DEFAULT_BINS
    order: int = DEFAULT_ORDER
    surrogates: int = SURROGATE_ROUNDS
    seed: int = SURROGATE_SEED


def shannon_bits(values: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """等频分箱的香农熵（bit/根）。重复分位点会让实际箱数变少。"""
    if values.size < MIN_SAMPLES:
        return np.nan
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0
    counts = np.bincount(np.digitize(values, edges[1:-1]), minlength=edges.size - 1)
    shares = counts[counts > 0] / float(values.size)
    return float(-(shares * np.log2(shares)).sum())


def ordinal_bits(values: np.ndarray, order: int = DEFAULT_ORDER) -> float:
    """顺序熵（bit/根）：长度为 order 的排序模式分布，免分箱、对厚尾稳健。"""
    if values.size < max(MIN_SAMPLES, order + 1):
        return np.nan
    windows = np.lib.stride_tricks.sliding_window_view(values, order)
    patterns = np.argsort(windows, axis=1, kind="stable")
    codes = np.ravel_multi_index(patterns.T, (order,) * order)
    counts = np.bincount(codes, minlength=order**order)
    shares = counts[counts > 0] / float(codes.size)
    return float(-(shares * np.log2(shares)).sum())


def binarize(values: np.ndarray) -> np.ndarray:
    """按中位数二值化（含 0 时即「涨/跌」）；常数序列全 0。"""
    return (values > np.median(values)).astype(np.int8)


def _lz76_count(seq: np.ndarray) -> int:
    """LZ76 子串（phrase）个数 c(n)：Kaspar-Schuster 1987 的算法。"""
    size = int(seq.size)
    cursor, match, length, count = 0, 1, 1, 1
    while length + match <= size:
        if seq[cursor + match - 1] == seq[length + match - 1]:
            match += 1
        else:
            cursor += 1
            if cursor == length:
                count += 1
                length += match
                cursor, match = 0, 1
            else:
                match = 1
    return count + 1


def lempel_ziv_bits(values: np.ndarray) -> float:
    """LZ76 熵率（bit/根）：c(n)·log2(n)/n，随机序列趋近 1、周期序列趋近 0。"""
    if values.size < MIN_SAMPLES:
        return np.nan
    size = int(values.size)
    return float(_lz76_count(binarize(values)) * np.log2(size) / size)


def kontoyiannis_bits(values: np.ndarray) -> float:
    """Kontoyiannis 最长匹配熵率（bit/根）：n·log2(n)/Σ L_i，只在**前缀**里匹配。"""
    if values.size < MIN_SAMPLES:
        return np.nan
    text = "".join(map(str, binarize(values).tolist()))
    size = len(text)
    total = 0
    for index in range(1, size):
        prefix = text[:index]
        length = 0
        while (
            index + length < size and prefix.find(text[index : index + length + 1]) >= 0
        ):
            length += 1
        total += length
    if total <= 0:
        return 0.0
    return float(size * np.log2(size) / total)


def window_bits(values: np.ndarray, options: EntropyOptions) -> dict[str, float]:
    """一个窗口的全部读数：序列类估计量各扣掉同窗置换基线。"""
    point = {
        "shannon_bits": shannon_bits(values, options.bins),
        "ordinal_bits": ordinal_bits(values, options.order),
        "lz_bits": lempel_ziv_bits(values),
        "konto_bits": kontoyiannis_bits(values),
    }
    generator = np.random.default_rng(options.seed)
    shuffled_lz: list[float] = []
    shuffled_konto: list[float] = []
    for _ in range(options.surrogates):
        surrogate = generator.permutation(values)
        shuffled_lz.append(lempel_ziv_bits(surrogate))
        shuffled_konto.append(kontoyiannis_bits(surrogate))
    point["lz_excess"] = point["lz_bits"] - float(np.nanmean(shuffled_lz))
    point["konto_excess"] = point["konto_bits"] - float(np.nanmean(shuffled_konto))
    return point


def entropy_panel(
    returns: pd.DataFrame, options: EntropyOptions | None = None
) -> pd.DataFrame:
    """逐标的滚窗算熵，返回长表（date/symbol + FEATURE_COLUMNS）。

    `returns` 是宽表（日期 × 标的）的**逐根收益**；第 t 天的特征只用 [t-window+1, t]，
    窗口不足或读数缺失的标的日不出现在结果里。
    """
    settings = options or EntropyOptions()
    rows: list[dict[str, object]] = []
    dates = returns.index
    for symbol in returns.columns:
        series = returns[symbol].to_numpy(dtype=float)
        for position in range(settings.window - 1, len(series)):
            chunk = series[position - settings.window + 1 : position + 1]
            if not np.isfinite(chunk).all():
                continue
            row: dict[str, object] = {"date": dates[position], "symbol": symbol}
            row.update(window_bits(chunk, settings))
            rows.append(row)
    return pd.DataFrame(rows)


def load_panel_returns(panel: Panel) -> pd.DataFrame:
    """从 Panel 取逐根收益宽表（close 的逐根简单收益）。"""
    return cast(pd.DataFrame, panel.close.pct_change()).dropna(how="all")
