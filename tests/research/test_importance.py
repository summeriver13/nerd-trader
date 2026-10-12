"""Ch 8 特征重要性的用例。

核心是**替代效应**：造一份已知真相的合成数据（有信息 / 冗余 / 纯噪声），
然后钉死三件事——

1. MDI 会给纯噪声特征分到重要性（样本内、有偏）；
2. 有替身时，一个好特征的逐特征 MDA 被**显著稀释**（替身越像，稀释越狠）；
3. 聚类 MDA 把整簇一起打乱，能读出这块信息的**联合**重要性。

**测试里一律用 `n_jobs=1` 的森林**：sklearn 并行拟合与并行预测都有
5.6e-16 量级的浮点抖动（求和顺序不同），会让「两次跑完全一样」这类断言假红。
该抖动对结论无影响，但它不是本模块的行为，所以由测试自己控制。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.research import importance

FEATURES = ("informative", "redundant", "noise")
LABEL = "label"
WEIGHT = "weight"
FOLDS = 4
N_ROWS = 6000
TREE_JOBS = 1


def _synthetic() -> pd.DataFrame:
    """有信息 / 冗余 / 纯噪声三类特征，标签只由底层信号决定。

    `redundant` 是 `informative` 的替身（同一信号加 0.1 的独立噪声，相关 0.99），
    所以两个都是好特征、但可以互相替代——这正是替代效应的舞台。
    """
    rng = np.random.default_rng(0)
    signal = rng.normal(size=N_ROWS)
    frame = pd.DataFrame(
        {
            "informative": signal + 0.1 * rng.normal(size=N_ROWS),
            "redundant": signal + 0.1 * rng.normal(size=N_ROWS),
            "noise": rng.normal(size=N_ROWS),
        }
    )
    probability = 1.0 / (1.0 + np.exp(-2.0 * signal))
    frame[LABEL] = (rng.random(N_ROWS) < probability).astype(int)
    frame[WEIGHT] = 1.0
    return frame


def _fit_folds(
    frame: pd.DataFrame, features: tuple[str, ...] = FEATURES
) -> list[importance.FoldFit]:
    """切成 FOLDS 块，每块只用**之前**的数据训练——样本外，无前视。"""
    block = N_ROWS // FOLDS
    fits: list[importance.FoldFit] = []
    for index in range(1, FOLDS):
        train = frame.iloc[: index * block]
        rows = frame.index[index * block : (index + 1) * block]
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=20,
            random_state=0,
            n_jobs=TREE_JOBS,
        )
        model.fit(
            train[list(features)].to_numpy(),
            train[LABEL].to_numpy(),
            sample_weight=train[WEIGHT].to_numpy(),
        )
        fits.append(importance.FoldFit(model=model, rows=rows))
    return fits


def _cluster_name(clusters: dict[str, tuple[str, ...]], feature: str) -> str:
    """特征所在簇的簇名。"""
    for name, members in clusters.items():
        if feature in members:
            return name
    raise AssertionError(f"{feature} 不属于任何簇")


class TestFeatureClusters:
    def test_correlated_pair_lands_in_one_cluster(self) -> None:
        clusters = importance.feature_clusters(_synthetic(), FEATURES)
        assert set(clusters[_cluster_name(clusters, "informative")]) == {
            "informative",
            "redundant",
        }

    def test_independent_feature_stays_alone(self) -> None:
        clusters = importance.feature_clusters(_synthetic(), FEATURES)
        assert clusters[_cluster_name(clusters, "noise")] == ("noise",)

    def test_higher_threshold_never_makes_more_clusters(self) -> None:
        """阈值是「距离 ≤ t 才合并」，所以簇数必须随 t 单调不增。"""
        frame = _synthetic()
        counts = [
            len(importance.feature_clusters(frame, FEATURES, threshold=value))
            for value in (0.01, 0.05, 0.1, 0.5, 0.8)
        ]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] == 3  # 0.99 相关的两个特征也能拆开
        assert counts[-1] == 1  # 阈值高到把无关特征也并进来

    def test_single_feature_is_its_own_cluster(self) -> None:
        clusters = importance.feature_clusters(_synthetic(), ["noise"])
        assert clusters == {"c1": ("noise",)}


class TestMeanDecreaseAccuracy:
    def test_pure_noise_feature_has_no_importance(self) -> None:
        """纯噪声列被打乱不影响样本外损失——这是 MDA 相对 MDI 的价值所在。"""
        frame = _synthetic()
        table = importance.mean_decrease_accuracy(_fit_folds(frame), frame, FEATURES)
        assert abs(float(table.loc["noise", "mean"])) < 0.005

    def test_substitution_dilutes_a_real_feature(self) -> None:
        """替代效应：同一特征在有替身时的重要性远低于它是唯一信号时。"""
        frame = _synthetic()
        alone = importance.mean_decrease_accuracy(
            _fit_folds(frame, ("informative", "noise")),
            frame,
            ["informative", "noise"],
        )
        together = importance.mean_decrease_accuracy(_fit_folds(frame), frame, FEATURES)
        diluted = float(together.loc["informative", "mean"])
        assert diluted < 0.4 * float(alone.loc["informative", "mean"])

    def test_one_member_cannot_explain_the_whole_block(self) -> None:
        """整簇打乱的损失增量必须明显大于簇里任何单列——替代效应的直接体现。"""
        frame = _synthetic()
        per_feature = importance.mean_decrease_accuracy(
            _fit_folds(frame), frame, FEATURES
        )
        clusters = importance.feature_clusters(frame, FEATURES)
        clustered = importance.mean_decrease_accuracy(
            _fit_folds(frame), frame, FEATURES, groups=clusters
        )
        block = _cluster_name(clusters, "informative")
        single = max(
            float(per_feature.loc["informative", "mean"]),
            float(per_feature.loc["redundant", "mean"]),
        )
        assert float(clustered.loc[block, "mean"]) > 2.0 * single

    def test_clustered_mda_separates_signal_from_noise(self) -> None:
        frame = _synthetic()
        clusters = importance.feature_clusters(frame, FEATURES)
        clustered = importance.mean_decrease_accuracy(
            _fit_folds(frame), frame, FEATURES, groups=clusters
        )
        signal = float(clustered.loc[_cluster_name(clusters, "informative"), "mean"])
        noise = float(clustered.loc[_cluster_name(clusters, "noise"), "mean"])
        assert signal > 0.05
        assert signal > 10.0 * abs(noise)

    def test_same_seed_gives_identical_numbers(self) -> None:
        frame = _synthetic()
        first = importance.mean_decrease_accuracy(_fit_folds(frame), frame, FEATURES)
        second = importance.mean_decrease_accuracy(_fit_folds(frame), frame, FEATURES)
        assert first.equals(second)

    def test_rows_outside_the_test_folds_do_not_matter(self) -> None:
        """重要性只该由「模型没见过的那几行」决定。"""
        frame = _synthetic()
        fits = _fit_folds(frame)
        baseline = importance.mean_decrease_accuracy(fits, frame, FEATURES)
        poisoned = frame.copy()
        outside = poisoned.index[: N_ROWS // FOLDS]
        poisoned.loc[outside, list(FEATURES)] = 999.0
        afterwards = importance.mean_decrease_accuracy(fits, poisoned, FEATURES)
        assert baseline.equals(afterwards)

    def test_unknown_member_is_rejected(self) -> None:
        frame = _synthetic()
        with pytest.raises(ValueError, match="未知特征"):
            importance.mean_decrease_accuracy(
                _fit_folds(frame), frame, FEATURES, groups={"c1": ("nope",)}
            )

    def test_empty_group_is_rejected(self) -> None:
        frame = _synthetic()
        with pytest.raises(ValueError, match="是空的"):
            importance.mean_decrease_accuracy(
                _fit_folds(frame), frame, FEATURES, groups={"c1": ()}
            )

    def test_table_shape(self) -> None:
        frame = _synthetic()
        table = importance.mean_decrease_accuracy(_fit_folds(frame), frame, FEATURES)
        assert list(table.columns) == list(importance.IMPORTANCE_COLUMNS)
        assert set(table.index) == set(FEATURES)
        assert (table["folds"] == FOLDS - 1).all()
        assert float(table["mean"].iloc[0]) >= float(table["mean"].iloc[-1])


class TestImpurityImportance:
    def test_mdi_gives_a_noise_feature_positive_importance(self) -> None:
        """MDI 的偏：样本内算的，纯噪声特征也会分到一块重要性。"""
        frame = _synthetic()
        mdi = importance.impurity_importance(_fit_folds(frame), FEATURES)
        assert float(mdi["noise"]) > 0.0

    def test_mdi_sums_to_one(self) -> None:
        frame = _synthetic()
        mdi = importance.impurity_importance(_fit_folds(frame), FEATURES)
        assert float(mdi.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_no_fits_gives_zeros(self) -> None:
        assert float(importance.impurity_importance([], FEATURES).sum()) == 0.0
