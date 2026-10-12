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
from sklearn.metrics import log_loss

from src.research import importance, meta

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


TREES = 80
DEPTH = 6
LEAF = 20
# SFI 要在**同一批折、同一组超参**上重训（只换特征集），所以超参只在这里定义一次：
# `_forest()` 给 MDA 用，`OPTIONS` 交给 SFI 内部的 `meta.fit_classifier`。
OPTIONS = meta.MetaLabelOptions(
    n_estimators=TREES, max_depth=DEPTH, min_samples_leaf=LEAF, seed=0
)


def _forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=TREES,
        max_depth=DEPTH,
        min_samples_leaf=LEAF,
        random_state=0,
        n_jobs=TREE_JOBS,
    )


def _xor_frame() -> pd.DataFrame:
    """异或：两个特征各自都没有信息，只有**一起看**才有（联合效应的纯粹形态）。"""
    rng = np.random.default_rng(1)
    left = rng.integers(0, 2, size=N_ROWS)
    right = rng.integers(0, 2, size=N_ROWS)
    frame = pd.DataFrame({"left": left.astype(float), "right": right.astype(float)})
    frame[LABEL] = (left != right).astype(int)
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
        model = _forest()
        model.fit(
            train[list(features)].to_numpy(),
            train[LABEL].to_numpy(),
            sample_weight=train[WEIGHT].to_numpy(),
        )
        fits.append(importance.FoldFit(model=model, rows=rows, train_rows=train.index))
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


class TestImportanceTable:
    def test_one_row_per_feature_with_all_four_tools(self) -> None:
        """合并表：一行一个特征，四种工具的读数并排（分歧才是要看的东西）。"""
        frame = _synthetic()
        table = importance.importance_table(
            _fit_folds(frame), frame, FEATURES, options=OPTIONS
        )
        assert set(table["feature"]) == set(FEATURES)
        assert {"mda_logloss", "cluster_t", "mdi_gini", "sfi_skill"} <= set(
            table.columns
        )
        assert float(table["mdi_gini"].sum()) == pytest.approx(1.0, abs=1e-6)


class TestSingleFeatureImportance:
    """SFI 的两个**定义性**性质：没有替代效应、看不到联合效应。"""

    def test_a_copy_is_not_diluted_but_noise_scores_zero(self) -> None:
        """替身（副本）拿到与原件同等的读数——这正是逐特征 MDA 做不到的。

        对照 `TestMeanDecreaseAccuracy` 那条稀释用例：同一份数据，MDA 让
        `informative` 从 0.539 缩到 0.099；SFI 这里副本与原件基本持平。
        """
        frame = _synthetic()
        table = importance.single_feature_importance(
            _fit_folds(frame), frame, FEATURES, options=OPTIONS
        )
        skill = {str(name): float(row) for name, row in table["skill"].items()}
        assert skill["informative"] > 0.15
        assert skill["redundant"] >= 0.8 * skill["informative"]
        assert abs(skill["noise"]) < 0.02

    def test_misses_a_pure_joint_effect(self) -> None:
        """异或：两个特征单独都无用，SFI 读数都是 0。单纯看这个会误判这两个特征没用。"""
        frame = _xor_frame()
        table = importance.single_feature_importance(
            _fit_folds(frame, ("left", "right")),
            frame,
            ("left", "right"),
            options=OPTIONS,
        )
        assert (table["skill"].abs() < 0.01).all()
        assert (table["logloss"] > 0.68).all()

    def test_joint_effect_is_reachable_in_the_same_frame(self) -> None:
        """上一条的对照面：同一份异或数据，两个特征一起看就没有问题。

        没有这条用例，「SFI 读数为 0」既可能是它真没用，也可能是数据本来就没信息。
        """
        frame = _xor_frame()
        cut = N_ROWS * (FOLDS - 1) // FOLDS
        train, test = frame.iloc[:cut], frame.iloc[cut:]
        model = _forest()
        model.fit(train[["left", "right"]].to_numpy(), train[LABEL].to_numpy())
        probability = np.asarray(
            model.predict_proba(test[["left", "right"]].to_numpy())
        )
        assert float(log_loss(test[LABEL], probability, labels=[0, 1])) < 0.05

    def test_folds_line_up_with_the_meta_folds(self) -> None:
        """SFI 用的是同一批折——否则「哪些特征有用」和元标签不是同一个实验。"""
        frame = _synthetic()
        fits = _fit_folds(frame)
        table = importance.single_feature_importance(
            fits, frame, FEATURES, options=OPTIONS
        )
        assert (table["folds"] == len(fits)).all()
        assert float(table["skill"].iloc[0]) >= float(table["skill"].iloc[-1])
