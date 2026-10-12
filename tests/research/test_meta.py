"""元标签的用例。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import meta


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
