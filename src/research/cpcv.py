"""
de Prado 第 12 章：组合式 purged 交叉验证（CPCV）与回测路径。

为什么本仓需要它：结论长期是「一个数」——单次切分给出一个夏普，于是「三种配权方法没差别」
「调参 4 正 4 负」这类判断分不清「差别本来就小」和「噪声太大」。CPCV 把时间轴切成 N 组，
每次拿 k 组做测试（共 C(N,k) 组切分），再按原书的分配规则把这些切分拼成
φ = C(N,k)·k/N 条**回测路径**：每条路径都覆盖全时段，于是每个方案得到的是一组读数。

三处口径，前两处由本仓数据形态决定、第三处是原文实现里的一处缺陷：

1. 切分与 purge/embargo 的单位都是**日期**，不是行——本仓面板一行 = 一天一只标的，按行切
   会把同一天其它标的漏进训练集（第 9 章的 PurgedKFold 已经踩过这个坑）。
2. 训练集是**测试组之外的全部日期**（含测试组之后的），与原书第 7 章 purged K 折一致；
   因此 k=1 时 CPCV 与 `tuning.PurgedKFold` 完全等价（由用例钉住，不是靠约定）。
3. purge 取每个测试块**之前** `purge` 天、embargo 取**之后** `embargo` 天。原文的 mlfinlab
   派生实现在主循环里把 `self.embargo = 0` 写死（等于没有 embargo），本仓不照抄。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
import pandas as pd

DEFAULT_GROUPS: Final[int] = 10
DEFAULT_TEST_GROUPS: Final[int] = 2


@dataclass(frozen=True)
class CpcvSplit:
    """一组切分：`test_groups` 这几个组做测试，其余日期做训练（已挖掉 purge/embargo）。

    三个数组存的都是**唯一日期轴**上的下标，不是行号。
    """

    test_groups: tuple[int, ...]
    test_dates: np.ndarray
    train_dates: np.ndarray


@dataclass(frozen=True)
class CombinatorialPurgedCv:
    """组合式 purged 交叉验证。构造后所有方法都是纯函数，不持有数据。"""

    n_groups: int = DEFAULT_GROUPS
    k_test: int = DEFAULT_TEST_GROUPS
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_groups < 3:
            raise ValueError("组数至少 3，否则组合式切分退化成普通 K 折")
        if not 1 <= self.k_test < self.n_groups:
            raise ValueError("测试组数必须落在 [1, 组数) 之间")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge 与 embargo 不能为负")

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.k_test)

    @property
    def n_paths(self) -> int:
        """回测路径数 φ = C(N,k)·k/N。"""
        return self.n_splits * self.k_test // self.n_groups

    def group_dates(self, dates: int) -> tuple[np.ndarray, ...]:
        """`dates` 个唯一日期切成 `n_groups` 段，返回每段的下标（边界与 PurgedKFold 同式）。"""
        if dates < self.n_groups:
            raise ValueError("唯一日期数少于组数")
        bounds = np.linspace(0, dates, self.n_groups + 1, dtype=int)
        return tuple(np.arange(bounds[g], bounds[g + 1]) for g in range(self.n_groups))

    def split(self, dates: int) -> list[CpcvSplit]:
        """产出全部 C(N,k) 组切分，顺序按测试组的字典序（与 `paths()` 的编号对齐）。"""
        groups = self.group_dates(dates)
        splits: list[CpcvSplit] = []
        for combo in combinations(range(self.n_groups), self.k_test):
            test = np.concatenate([groups[group] for group in combo])
            splits.append(
                CpcvSplit(
                    test_groups=combo,
                    test_dates=test,
                    train_dates=self._train_dates(dates, groups, combo),
                )
            )
        return splits

    def _train_dates(
        self,
        dates: int,
        groups: tuple[np.ndarray, ...],
        combo: tuple[int, ...],
    ) -> np.ndarray:
        """训练日期 = 测试组之外的全部，再挖掉每个测试块前 purge 天、后 embargo 天。"""
        dropped = [groups[group] for group in combo]
        for group in combo:
            start = int(groups[group][0])
            stop = int(groups[group][-1]) + 1
            dropped.append(np.arange(max(start - self.purge, 0), start))
            dropped.append(np.arange(stop, min(stop + self.embargo, dates)))
        mask = np.ones(dates, dtype=bool)
        mask[np.concatenate(dropped)] = False
        return np.flatnonzero(mask)

    def paths(self) -> np.ndarray:
        """φ×N 矩阵：`[p, g]` = 路径 p 的第 g 组该用哪次切分的测试段。

        分配规则等价于原书 `_fill_backtest_paths` 的贪心：按切分顺序遍历，把每个测试组
        放进「该组槽位还没被占」的第一条路径。每个组恰好出现在 C(N-1,k-1) = φ 次切分里，
        所以每条路径的每组都能拿到唯一来源、且每条路径覆盖全部 N 组。
        """
        assignment = np.full((self.n_paths, self.n_groups), -1, dtype=int)
        for index, combo in enumerate(combinations(range(self.n_groups), self.k_test)):
            for group in combo:
                free = np.flatnonzero(assignment[:, group] < 0)
                if free.size == 0:
                    raise ValueError(
                        f"第 {group} 组的路径槽位不够，组数与测试组数不匹配"
                    )
                assignment[free[0], group] = index
        return assignment


def path_series(
    split_returns: Sequence[pd.Series],
    *,
    paths: np.ndarray,
    group_dates: Sequence[np.ndarray],
) -> pd.DataFrame:
    """把「各切分在它测试段上的收益」拼成 φ 条完整路径（列 = 路径名）。

    `split_returns[i]` 必须**对齐到全时段**（索引是全部日期，测试段之外为 NaN）——按位置
    取数，所以对齐是调用方的责任，拼错会得到静默错位的序列。路径 p 的第 g 组取自
    `paths[p, g]` 指定的那次切分：收益来自该次切分的测试段，权重/模型来自它的训练段。
    """
    rows = len(split_returns[0])
    if any(len(series) != rows for series in split_returns):
        raise ValueError("split_returns 的长度不一致：必须都对齐到全时段")
    values = np.full((rows, paths.shape[0]), np.nan, dtype=float)
    for column in range(paths.shape[0]):
        for group, split_index in enumerate(paths[column]):
            positions = group_dates[group]
            values[positions, column] = split_returns[split_index].to_numpy()[positions]
    labels = [f"路径{index + 1}" for index in range(paths.shape[0])]
    data = {label: values[:, index] for index, label in enumerate(labels)}
    return pd.DataFrame(data, index=split_returns[0].index)
