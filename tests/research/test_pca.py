"""Ch 8 正交化与确认性秩相关（原文 Snippet 8.5 / 8.6）的用例。

被测模块：`src.research.importance.pca`。用例分三组：正交化的**定义性性质**、无监督
排名的**判别力**（能指出「模型追的不是数据的主结构」）、以及 `rank_agreement` 对原文
调用语义的钉死。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import weightedtau

from src.research.importance import pca

N_ROWS = 5000
LABEL = "label"


def _decoy_frame(seed: int = 0) -> pd.DataFrame:
    """一簇主导方差、但与标签**无关**的特征（decoy）+ 一个与标签相关的正交特征。

    两次运行同一 seed 结果一致；decoy 之间高度相关（0.96），所以它们撑起第一主成分，
    而 `signal` 与它们正交、自己只顶起很小的方差方向。
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=N_ROWS)
    signal = rng.normal(size=N_ROWS)
    return pd.DataFrame(
        {
            "decoy_a": common + 0.2 * rng.normal(size=N_ROWS),
            "decoy_b": common + 0.2 * rng.normal(size=N_ROWS),
            "signal": signal,
            LABEL: (signal + 0.5 * rng.normal(size=N_ROWS) > 0).astype(int),
        }
    )


def _features(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name != LABEL]


def _rank_of(score: pd.Series) -> pd.Series:
    order = np.argsort(-score.to_numpy(dtype=float), kind="stable")
    ranks = np.empty(score.size, dtype=int)
    ranks[order] = np.arange(1, score.size + 1)
    return pd.Series(ranks, index=score.index)


def _scipy_tau(importance: pd.Series, rank: pd.Series) -> float:
    """scipy 默认 weigher（双曲加权 1/(r+1)）下的秩相关，作为独立参照。"""
    # scipy 返回具名元组，桩把它推成未知类型，显式收窄（仓库规则：不用 pyright: ignore）
    statistic = cast(
        float,
        weightedtau(importance.to_numpy(dtype=float), 1.0 / rank.to_numpy(dtype=float))[
            0
        ],
    )
    return float(statistic)


def _scipy_tau_unweighted(importance: pd.Series, rank: pd.Series) -> float:
    """均匀权重，退化成不加权的 Kendall's tau——用来证明加权确实起了作用。"""
    statistic = cast(
        float,
        weightedtau(
            importance.to_numpy(dtype=float),
            1.0 / rank.to_numpy(dtype=float),
            weigher=lambda rank: 1.0,
        )[0],
    )
    return float(statistic)


def test_orthogonal_features_are_uncorrelated() -> None:
    """正交特征的定义：两两不相关（离单位阵的最大偏差应该是数值噪声）。"""
    frame = _decoy_frame()
    ortho = pca.orthogonal_features(frame, _features(frame))
    corr = np.corrcoef(ortho.scores.to_numpy(dtype=float), rowvar=False)
    assert np.abs(corr - np.eye(corr.shape[0])).max() < 1e-10


def test_the_projection_reconstructs_the_standardised_features() -> None:
    """得分与载荷必须自洽：`scores @ loadings.T` 还原标准化后的特征。"""
    frame = _decoy_frame()
    columns = _features(frame)
    # 要全保留才谈得上「还原」：少留一个主成分，重建就是有损的（截断口径另有用例）
    ortho = pca.orthogonal_features(frame, columns, variance_threshold=1.0)
    values = frame[columns].to_numpy(dtype=float)
    standardised = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    rebuilt = (
        ortho.scores.to_numpy(dtype=float) @ ortho.loadings.to_numpy(dtype=float).T
    )
    assert np.abs(rebuilt - standardised).max() < 1e-8


def test_explained_ratios_match_the_variance_of_the_scores() -> None:
    """解释方差比例要能从得分独立算出来，而不是只能信 PCA 的返回值。"""
    frame = _decoy_frame()
    columns = _features(frame)
    ortho = pca.orthogonal_features(frame, columns)
    values = frame[columns].to_numpy(dtype=float)
    # PCA 是在**标准化**后的数据上算的，所以总方差恒等于特征数，不是原始方差之和
    standardised = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    total = float(standardised.var(axis=0, ddof=1).sum())
    for name in ortho.explained.index:
        variance = float(ortho.scores[name].to_numpy(dtype=float).var(ddof=1))
        assert float(ortho.explained[name]) == pytest.approx(variance / total, rel=1e-9)


def test_the_threshold_keeps_the_smallest_covering_set() -> None:
    """保留数要**最小**：达到阈值，但去掉最后一个就不够。"""
    frame = _decoy_frame()
    columns = _features(frame)
    ortho = pca.orthogonal_features(frame, columns, variance_threshold=0.95)
    ratios = ortho.explained.to_numpy(dtype=float)
    assert ratios.sum() >= 0.95
    assert ratios[:-1].sum() < 0.95
    assert ortho.loadings.shape[1] == len(ratios)
    assert ortho.scores.shape[1] == len(ratios)


def test_keeping_every_component_at_threshold_one() -> None:
    frame = _decoy_frame()
    columns = _features(frame)
    ortho = pca.orthogonal_features(frame, columns, variance_threshold=1.0)
    assert ortho.loadings.shape[1] == len(columns)
    assert ortho.explained.sum() == pytest.approx(1.0)


def test_a_single_component_rank_follows_the_pc1_loading() -> None:
    """文档串里写的退化：只保留一个主成分时，排名就是按 |PC1 载荷| 排。"""
    frame = _decoy_frame()
    columns = _features(frame)
    ortho = pca.orthogonal_features(frame, columns, variance_threshold=0.5)
    assert ortho.loadings.shape[1] == 1
    matrix = ortho.loadings.to_numpy(dtype=float)
    loading = pd.Series(np.abs(matrix[:, 0]), index=ortho.loadings.index)
    assert pca.unsupervised_rank(ortho).to_dict() == _rank_of(loading).to_dict()


def test_the_unsupervised_rank_can_disagree_with_a_supervised_one() -> None:
    """这个工具存在的意义：无监督排名会**背离**有监督排名。

    数据里有一簇主导方差但与标签无关的特征（decoy）和一个与标签强相关的正交特征。
    PCA 把 decoy 簇排前面（它撑起第一主成分），而按与标签的相关排则 `signal` 第一。
    两者的秩相关因此为负——这正是「模型有没有在追数据的主结构」这个问题的读数。
    """
    frame = _decoy_frame()
    columns = _features(frame)
    rank = pca.unsupervised_rank(pca.orthogonal_features(frame, columns))
    matrix = frame[[*columns, LABEL]].to_numpy(dtype=float)
    supervised = pd.Series(
        [
            abs(float(np.corrcoef(matrix[:, position], matrix[:, -1])[0, 1]))
            for position in range(len(columns))
        ],
        index=columns,
    )
    # 两个 decoy 在统计上完全对称，谁排第一是抽样噪声决定的，只该断言**相对次序**
    assert rank["decoy_a"] < rank["signal"]
    assert rank["decoy_b"] < rank["signal"]
    assert rank["signal"] == len(columns)
    assert _rank_of(supervised)["signal"] == 1
    assert pca.rank_agreement(supervised, rank) < 0


def test_unsupervised_ranking_uses_variance_not_the_label() -> None:
    """把标签整列换掉，无监督排名必须逐行不变（它压根不看标签）。"""
    frame = _decoy_frame()
    columns = _features(frame)
    polluted = frame.copy()
    polluted[LABEL] = 1 - polluted[LABEL]
    first = pca.unsupervised_rank(pca.orthogonal_features(frame, columns))
    second = pca.unsupervised_rank(pca.orthogonal_features(polluted, columns))
    assert first.to_dict() == second.to_dict()


def test_rank_agreement_matches_the_book_snippet() -> None:
    """原文 Snippet 8.6 的语义：第二个位置参数是布尔 `rank`，不是权重。

    原文默认例（imp 0.55/0.33/0.07/0.05、pcRank 1/2/4/3）给 0.813；不加权的
    Kendall's tau 只有 0.667——差值来自 scipy 默认的双曲加权。
    """
    index = list("abcd")
    importance = pd.Series([0.55, 0.33, 0.07, 0.05], index=index)
    rank = pd.Series([1.0, 2.0, 4.0, 3.0], index=index)
    value = pca.rank_agreement(importance, rank)
    assert value == pytest.approx(_scipy_tau(importance, rank), rel=1e-12)
    assert value == pytest.approx(0.813333, abs=1e-5)
    assert value > _scipy_tau_unweighted(importance, rank)


def test_rank_agreement_extremes() -> None:
    index = list("abc")
    importance = pd.Series([0.6, 0.3, 0.1], index=index)
    assert pca.rank_agreement(
        importance, pd.Series([1, 2, 3], index=index)
    ) == pytest.approx(1.0)
    assert pca.rank_agreement(
        importance, pd.Series([3, 2, 1], index=index)
    ) == pytest.approx(-1.0)


def test_rank_agreement_rejects_unusable_input() -> None:
    three = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    with pytest.raises(ValueError, match="至少需要"):
        pca.rank_agreement(three.iloc[:2], pd.Series([1, 2], index=list("ab")))
    with pytest.raises(ValueError, match="NaN"):
        pca.rank_agreement(three, pd.Series([1.0, 2.0, 3.0], index=["a", "b", "d"]))


def test_orthogonal_features_reject_degenerate_input() -> None:
    with pytest.raises(ValueError, match="至少需要一个特征"):
        pca.orthogonal_features(_decoy_frame(), [])
    with pytest.raises(ValueError, match="有限"):
        bad = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        pca.orthogonal_features(bad, ["a", "b"])
    with pytest.raises(ValueError, match="常数"):
        flat = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
        pca.orthogonal_features(flat, ["a", "b"])
