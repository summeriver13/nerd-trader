"""
顺序 bootstrap 采样器（de Prado Ch 4 §4.5/4.6）。

标签在时间上重叠时，均匀 bootstrap 会反复抽到**互相重叠**的观测：袋子看着有 N 个样本，
承载的独立信息却远少于 N。顺序 bootstrap 把「抽哪一条」的概率设成「把这条加进当前袋子
后，它**自己**的平均唯一性」，于是每一步都优先补进尚未被重复覆盖的观测。

算法（Snippet 4.3/4.4）：φ 从空开始，重复到长度 length——

1. 对每个候选 i，算 u_i = 把 i 加进 φ 后 i 自己在袋内的平均唯一性；
2. 按 u_i / Σu 抽一个 i 加进 φ（有放回，φ 里可以有重复）。

实现要点：u_i 只依赖袋子的逐根并发数 c(t)——u_i = mean_{t∈span(i)} 1/(c(t)+1)——所以能用
前缀和 O(1) 算出来，不必像原书伪码那样为每个候选重建一次袋矩阵（那是 O(length·N·T)）。
`candidate_uniqueness` 与 `bag_uniqueness` 都有测试与朴素重建法逐位对拍。

区间约定：`[start, max(end, start + 1))`，**标签至少覆盖 1 根**——退化标签（end == start）
若按空区间处理会让所有抽签概率为 0。注意这与 `overlap.uniqueness_from_ends` 在退化标签
上给出的 0 不同；`labels.triple_barrier` 与 `labels.trend_scan` 的 t1 恒 ≥ t0+1，所以这个
差别只在人为构造的输入上可见。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 4。
"""

from __future__ import annotations

import numpy as np

from src.research import overlap

DEFAULT_SEED = 0


def span_ends(size: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """把区间裁进 [0, size] 并保证每段至少 1 根（左闭右开）。"""
    return np.maximum(np.clip(ends, 0, size), starts + 1)


def indicator_matrix(size: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """T×N 的 0/1 指示矩阵（de Prado 的 `getIndMatrix`）：第 j 列在其区间内为 1。

    只用于诊断与测试对拍——采样热路径走 `candidate_uniqueness` 的前缀和。
    """
    clipped = span_ends(size, starts, ends)
    position = np.arange(size)[:, None]
    return ((position >= starts[None, :]) & (position < clipped[None, :])).astype(float)


def bag_concurrency(
    size: int, starts: np.ndarray, ends: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    """袋子的逐根并发数：`draws` 里每抽到一次就累加一份（重复抽同一标签会叠加）。"""
    if draws.size == 0:
        return np.zeros(size, dtype=float)
    picked = np.asarray(draws, dtype=np.int64)
    return overlap.concurrency_from_ends(
        size, starts[picked], span_ends(size, starts, ends)[picked]
    )


def bag_uniqueness(
    size: int, starts: np.ndarray, ends: np.ndarray, draws: np.ndarray
) -> float:
    """袋子的平均唯一性 = 袋内每个「(被抽中的标签, 它覆盖的根)」单元的 1/并发数 的均值。

    与 de Prado 的 `getAvgUniqueness` 同定义：区间长的标签在均值里权重更大。袋子为空时
    返回 1.0（没有重叠可言）。
    """
    if draws.size == 0:
        return 1.0
    clipped = span_ends(size, starts, ends)
    alive = np.maximum(bag_concurrency(size, starts, ends, draws), 1.0)
    cumulative = np.concatenate([[0.0], np.cumsum(1.0 / alive)])
    picked = np.asarray(draws, dtype=np.int64)
    covered = np.maximum(clipped[picked] - starts[picked], 1)
    return float(
        (cumulative[clipped[picked]] - cumulative[starts[picked]]).sum() / covered.sum()
    )


def candidate_uniqueness(
    size: int, starts: np.ndarray, ends: np.ndarray, concurrency: np.ndarray
) -> np.ndarray:
    """把每个候选加进当前袋子后，**它自己**的平均唯一性（即抽签权重）。

    `concurrency` 是当前袋子的逐根并发数（尚未含候选）。返回长度等于标签数。
    """
    clipped = span_ends(size, starts, ends)
    alive = np.maximum(concurrency + 1.0, 1.0)
    cumulative = np.concatenate([[0.0], np.cumsum(1.0 / alive)])
    lengths = np.maximum(clipped - starts, 1)
    return (cumulative[clipped] - cumulative[starts]) / lengths


def sequential_bootstrap(
    size: int,
    starts: np.ndarray,
    ends: np.ndarray,
    length: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """顺序 bootstrap：抽 `length` 个标签（有放回），概率 ∝ 候选自己的袋内平均唯一性。

    长度为 0、长度为负或没有标签时返回空数组。
    """
    count = int(starts.size)
    if length <= 0 or count == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    clipped = span_ends(size, starts, ends)
    delta = np.zeros(size + 1, dtype=float)
    draws = np.empty(length, dtype=np.int64)
    for step in range(length):
        weights = candidate_uniqueness(size, starts, ends, np.cumsum(delta[:-1]))
        draws[step] = rng.choice(count, p=weights / weights.sum())
        delta[starts[draws[step]]] += 1.0
        delta[clipped[draws[step]]] -= 1.0
    return draws


def uniform_bootstrap(
    count: int, length: int, *, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """均匀 bootstrap（对照臂）：等概率、有放回地抽 `length` 个标签。

    这是 sklearn 各集成器的内部做法，也是顺序 bootstrap 要比的对象。
    """
    if length <= 0 or count <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, count, size=length).astype(np.int64)
