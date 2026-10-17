"""元标签的用例。"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.research.labeling import meta


def _frame(periods: int = 60, symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    """造一份特征与标签真相关的长表：feature 高则标签为 1。"""
    generator = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2024-01-01", periods=periods, freq="D"), symbols)
    feature = generator.normal(size=periods * symbols)
    return pd.DataFrame(
        {
            meta.DATE_COLUMN: dates,
            "signal": feature,
            "noise": generator.normal(size=periods * symbols),
            meta.LABEL_COLUMN: (
                feature + 0.5 * generator.normal(size=periods * symbols) > 0
            ).astype(int),
            meta.WEIGHT_COLUMN: np.ones(periods * symbols),
        }
    )


def test_meta_label_is_one_when_the_bet_made_money() -> None:
    returns = pd.Series([0.02, -0.02, 0.02, -0.02])
    long = pd.Series([1, 1, -1, -1])
    # 前两注做多（跌了的那注亏），后两注做空（涨了的那注亏）
    assert meta.meta_label(returns, long).tolist() == [1, 0, 0, 1]


def test_meta_label_rejects_missing_returns() -> None:
    """收益比方向短时必须报错：宁缺一注，也不能静默标错。"""
    with pytest.raises(ValueError, match="未对齐"):
        meta.meta_label(pd.Series([0.01, 0.02]), pd.Series([1, 1, 1]))


def test_validate_rejects_degenerate_options() -> None:
    for bad in ({"min_train": 10}, {"test_span": 0}, {"embargo": -1}):
        with pytest.raises(ValueError):
            meta.MetaLabelOptions(**bad).validate()  # type: ignore[arg-type]


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        meta.walk_forward_meta(_frame(), ["signal", "absent"], meta.MetaLabelOptions())


def test_first_prediction_starts_after_the_minimum_training_window() -> None:
    """预测只能出现在 min_train + embargo 之后。"""
    options = meta.MetaLabelOptions(
        min_train=100, test_span=40, embargo=10, n_estimators=10
    )
    frame = _frame()
    prediction = meta.walk_forward_meta(frame, ["signal", "noise"], options)
    first = int(np.flatnonzero(prediction.probability.notna().to_numpy())[0])
    assert first >= 100 + 10
    assert prediction.folds > 0


def test_no_lookahead_into_future_rows() -> None:
    """把第一个测试窗之后的数据全改掉，第一个窗的预测必须逐位不变。"""
    options = meta.MetaLabelOptions(
        min_train=100, test_span=40, embargo=10, n_estimators=10
    )
    frame = _frame()
    baseline = meta.walk_forward_meta(frame, ["signal", "noise"], options).probability

    poisoned = frame.copy()
    cutoff = int(np.flatnonzero(baseline.notna().to_numpy())[-1]) + 1
    poisoned.loc[cutoff:, ["signal", "noise"]] = 1e6
    poisoned.loc[cutoff:, meta.LABEL_COLUMN] = (
        1 - poisoned.loc[cutoff:, meta.LABEL_COLUMN]
    )
    poisoned = meta.walk_forward_meta(
        poisoned, ["signal", "noise"], options
    ).probability

    first_window = baseline.index.tolist()[:40]
    pd.testing.assert_series_equal(
        poisoned.reindex(first_window), baseline.reindex(first_window)
    )


def test_all_symbols_of_one_date_land_in_the_same_fold() -> None:
    """同一天的标的必须整批进同一个窗，否则同日信息就会跨窗泄漏。"""
    options = meta.MetaLabelOptions(
        min_train=100, test_span=40, embargo=10, n_estimators=10
    )
    frame = _frame()
    prediction = meta.walk_forward_meta(frame, ["signal", "noise"], options)
    known = prediction.probability.notna()
    counts = frame.loc[known].groupby(meta.DATE_COLUMN).size()
    assert counts.nunique() == 1  # 每个被预测的日期都有全部标的


def test_embargo_is_counted_in_days_not_observations() -> None:
    """禁运按天算：按观测算时「200 行」在每天上百只的窄板上只有约 1 天，3 日标签照样重叠。

    造一份每天 `symbols` 只标的的帧，1 天禁运必须挖掉整整一天的观测：训练集末尾停在测试窗
    首日的**前一天**，而不是「测试窗起始行 − symbols 行」。
    """
    symbols = 5
    options = meta.MetaLabelOptions(
        min_train=100, test_span=40, embargo=1, n_estimators=10, collect_fits=True
    )
    frame = _frame(periods=60, symbols=symbols)
    prediction = meta.walk_forward_meta(frame, ["signal", "noise"], options)
    assert prediction.folds > 0
    fit = prediction.fits[0]
    train_days = frame.loc[fit.train_rows, meta.DATE_COLUMN].to_numpy()
    test_days = frame.loc[fit.rows, meta.DATE_COLUMN].to_numpy()
    gap = (test_days.min() - train_days.max()) / np.timedelta64(1, "D")
    assert gap == options.embargo + 1  # 挖掉一整天：训练止于测试窗首日的前一天
    assert len(fit.train_rows) % symbols == 0  # 按天切，不会切出半天


def test_filter_summary_baseline_takes_every_bet() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.04])
    probability = pd.Series([0.2, 0.4, 0.6, 0.8])
    table = meta.filter_summary(returns, probability, thresholds=(0.0, 0.5, 0.9))
    assert table["bets"].tolist() == [4, 2, 0]
    assert table.loc[0, "per_bet_bp"] == pytest.approx(150.0)  # 四注均值
    assert table.loc[1, "per_bet_bp"] == pytest.approx(300.0)  # 只剩高的两注
    assert table.loc[1, "lift_bp"] == pytest.approx(150.0)
    assert table.loc[0, "lift_bp"] == pytest.approx(0.0)


def test_filter_summary_ignores_unpredicted_rows() -> None:
    """未预测的行（概率为 NaN）不能进基线。"""
    returns = pd.Series([0.01, 0.02, 0.03])
    probability = pd.Series([0.3, np.nan, 0.9])
    table = meta.filter_summary(returns, probability, thresholds=(0.0,))
    assert table.loc[0, "bets"] == 2


def _symbol_major_frame(
    symbols: int = 5, periods: int = 60, seed: int = 0
) -> pd.DataFrame:
    """按**标的**拼成的长表（日期非单调）——研究脚本的真实形状。

    `_frame()` 是日期序的，而按位置对齐的 bug 在日期序下看不出来，所以回归用例
    必须用这个形状，并且保留非默认的乱序索引。
    """
    frame = _frame(periods=periods, symbols=symbols, seed=seed)
    frame["symbol"] = np.tile([f"S{i}" for i in range(symbols)], periods)
    return frame.sort_values("symbol", kind="stable")


def test_probability_index_follows_the_input_row_order() -> None:
    """概率必须与输入 frame 的索引逐行同序。

    内部按日期排序后若把概率挂在**新** RangeIndex 上，两边就都是 0..N-1，调用方的
    `reindex` 成了空操作，概率被静默配到别的观测上。
    """
    options = meta.MetaLabelOptions(
        min_train=100, test_span=40, embargo=10, n_estimators=10
    )
    frame = _symbol_major_frame()
    assert not bool(frame[meta.DATE_COLUMN].is_monotonic_increasing)
    prediction = meta.walk_forward_meta(frame, ["signal", "noise"], options)
    assert prediction.probability.index.equals(frame.index)


def test_duplicate_index_is_rejected() -> None:
    """索引不唯一时按标签回填会歧义，宁可报错也不能默默配错。"""
    frame = _frame()
    frame.index = pd.Index([0] * len(frame))
    with pytest.raises(ValueError, match="索引必须唯一"):
        _ = meta.walk_forward_meta(frame, ["signal", "noise"], meta.MetaLabelOptions())


def test_filter_summary_pairs_probability_with_the_same_row() -> None:
    """端到端：把收益设成元标签本身，则「高概率那批」的命中率必须真的更高。

    配错行时概率来自别的观测，命中率不会随阈值上升。这个检验不依赖任何位置假设，
    只看概率与收益是否落在同一行。
    """
    # 帧要够大、叶子要够小：这条用例查的是「概率与收益是否落在同一行」，
    # 若模型平滑到给不出 0.7 以上的概率，bets=0 会让断言与「配错行」无法区分。
    options = meta.MetaLabelOptions(
        min_train=100,
        test_span=40,
        embargo=10,
        n_estimators=50,
        min_samples_leaf=10,
    )
    frame = _symbol_major_frame(symbols=5, periods=200)
    prediction = meta.walk_forward_meta(frame, ["signal", "noise"], options)
    table = meta.filter_summary(
        cast(pd.Series, frame[meta.LABEL_COLUMN].astype(float)),
        prediction.probability,
        (0.0, 0.7),
    )
    assert table.loc[1, "bets"] > 0
    assert float(table.loc[1, "hit_rate"]) > float(table.loc[0, "hit_rate"]) + 0.05


def test_rf_param_space_covers_the_pipeline_defaults() -> None:
    """默认配置必须与流水线默认值一致、且落在搜索空间内。

    否则「调参 vs 默认」比的不是同一件事：默认臂会落在空间之外，成为候选里根本没有的
    配置，Ch 9 的结论就变成「拿一个从没被搜索过的配置当基线」。
    """
    options = meta.MetaLabelOptions()
    assert set(meta.RF_DEFAULT_PARAMS) == set(meta.RF_PARAM_SPACE)
    for key, value in meta.RF_DEFAULT_PARAMS.items():
        assert getattr(options, key) == value
        assert value in meta.RF_PARAM_SPACE[key]
