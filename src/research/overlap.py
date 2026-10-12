"""
样本重叠：唯一性权重（de Prado Ch 4）与重叠剔除（Ch 7）。

标签在时间上重叠时，观测既不独立也无法归因——同一段收益被算了 h 次。
这两件工具针对的是同一件事：

- **唯一性权重**：不丢数据，把重叠的观测按 `1/并发标签数` 打折；
- **purge / embargo**：分块交叉验证时，把标签跨进测试块的训练观测剔掉。

仓库原先用 `iloc[::horizon]` 抽稀回避重叠，代价是丢掉 (h−1)/h 的数据
（h=30 丢 97%）、起始相位任意，且 h≥15 时非重叠观测少到 PBO 分不出块。
改用权重 + purge 后全部数据都能用，purge 也才有东西可剔。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 4 & Ch 7。
"""

from __future__ import annotations

import numpy as np

DEFAULT_EMBARGO_RATIO = 0.01  # 禁运长度按观测数比例取，作者原文 h ≈ 0.01T


def concurrency(size: int, starts: np.ndarray, horizon: int) -> np.ndarray:
    """逐根的并发标签数：某根上同时存活的标签个数。

    `starts` 是各观测的起点（根序号），每个观测的标签存活 `horizon` 根。
    """
    ends = np.minimum(starts + horizon, size)
    delta = np.zeros(size + 1, dtype=float)
    np.add.at(delta, starts, 1.0)
    np.add.at(delta, ends, -1.0)
    return np.cumsum(delta[:-1])


def average_uniqueness(size: int, starts: np.ndarray, horizon: int) -> np.ndarray:
    """每个观测的平均唯一性 = 标签存活期内 `1/并发标签数` 的均值。

    完全不重叠时恒为 1（并发恒为 1）；逐根采样 + h 根标签时趋近 `1/h`。
    返回值与 `starts` 等长，可直接当样本权重（de Prado 的 `getIndMatrix` +
    `getAvgUniqueness` 的矢量化等价物）。
    """
    if starts.size == 0:
        return np.zeros(0, dtype=float)
    ends = np.minimum(starts + horizon, size)
    alive = np.maximum(concurrency(size, starts, horizon), 1.0)
    inverse = 1.0 / alive
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])
    totals = cumulative[ends] - cumulative[starts]
    lengths = np.maximum(ends - starts, 1)
    return totals / lengths


def purge_mask(is_test: np.ndarray, purge: int, embargo: int) -> np.ndarray:
    """训练观测里该剔除的位置：标签跨进测试块（purge）或紧跟在测试块之后（embargo）。

    purge 的单位是**观测**（不是根）：一个观测的标签若覆盖到任何测试观测，就必须
    剔除——它的收益里混着测试期的信息。逐根采样 + h 根标签时 purge=h；非重叠采样
    （每 h 根取一个观测）时 purge=0，因为标签恰好首尾相接、不跨观测。

    embargo 防的是另一件事：标签虽不重叠，但序列相关仍会让测试块边界处的观测泄漏。
    作者取 h ≈ 0.01T，见 `DEFAULT_EMBARGO_RATIO`。
    """
    keep = ~is_test
    if purge <= 0 and embargo <= 0:
        return keep
    size = is_test.size
    # 前缀和：区间 [a, b) 内有没有测试行 = csum[b] − csum[a]
    cumulative = np.concatenate([[0], np.cumsum(is_test.astype(int))])
    position = np.arange(size)
    if purge > 0:
        end = np.minimum(position + purge + 1, size)
        keep &= (cumulative[end] - cumulative[position]) == 0
    if embargo > 0:
        start = np.maximum(position - embargo, 0)
        keep &= (cumulative[position] - cumulative[start]) == 0
    return keep


def embargo_from_ratio(observations: int, ratio: float = DEFAULT_EMBARGO_RATIO) -> int:
    """按观测数比例折算禁运长度（作者口径 h ≈ 0.01T）。"""
    return max(int(observations * ratio), 0)
