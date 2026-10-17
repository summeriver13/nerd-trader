"""Ch 6 基础设施：跨度接线（会静默算错的那一步）与估计器族。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.backtest import ensembles
from src.research.labeling import bagging, meta

FEATURES = ["mom_7", "age_days"]


def _close() -> pd.DataFrame:
    """两只标的、起始日不同（BBB 晚两天）——用行号当根号就会在这里出错。"""
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "BBB": [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-05")],
            "symbol": ["BBB", "BBB"],
            "barrier_t1": [7, 3],
        }
    )


def test_attach_spans_counts_bars_from_each_symbols_own_start() -> None:
    """起始根是 close.dropna() 里的位置：BBB 的第 1 根是总第 3 天，不是第 1 天。"""
    placed = ensembles.attach_spans(_frame(), _close())
    assert placed[bagging.GROUP_COLUMN].tolist() == [1, 1]
    assert placed[bagging.START_COLUMN].tolist() == [0, 2]
    assert placed[bagging.END_COLUMN].tolist() == [7, 3]
    assert placed[bagging.SIZE_COLUMN].tolist() == [4, 4]


def test_attach_spans_rejects_dates_off_the_bar_grid() -> None:
    """观测日不在该标的的根上（BBB 在 01-02 还没上市）= 坐标对不上，必须报错。"""
    frame = _frame()
    frame.loc[0, "date"] = pd.Timestamp("2026-01-02")
    with pytest.raises(ValueError, match="没有根"):
        ensembles.attach_spans(frame, _close())


def test_attach_spans_rejects_unknown_symbol() -> None:
    frame = _frame()
    frame.loc[0, "symbol"] = "CCC"
    with pytest.raises(ValueError, match="收盘价矩阵里没有"):
        ensembles.attach_spans(frame, _close())


def test_bag_uniqueness_reports_both_samplers() -> None:
    """袋唯一性诊断：两个采样器各一个值，都落在 (0, 1]（谁更高是研究脚本的读数）。"""
    placed = ensembles.attach_spans(_frame(), _close())
    placed[bagging.END_COLUMN] = placed[bagging.START_COLUMN] + 1
    values = ensembles.bag_uniqueness(placed, rows=len(placed), seed=0)
    assert sorted(values.index) == ["sequential", "uniform"]
    assert bool(((values > 0.0) & (values <= 1.0)).all())


def test_arm_fitters_fits_every_arm() -> None:
    """五个臂都能在带跨度坐标的小帧上拟合，并给出两列合法概率。"""
    rng = np.random.default_rng(0)
    rows = 60
    frame = pd.DataFrame(
        {
            "mom_7": rng.normal(size=rows),
            "age_days": rng.integers(30, 900, size=rows).astype(float),
            meta.LABEL_COLUMN: rng.integers(0, 2, size=rows),
            meta.WEIGHT_COLUMN: np.ones(rows),
            bagging.GROUP_COLUMN: np.zeros(rows, dtype=np.int64),
            bagging.START_COLUMN: np.arange(rows, dtype=np.int64),
            bagging.END_COLUMN: np.arange(rows, dtype=np.int64) + 1,
            bagging.SIZE_COLUMN: np.full(rows, rows, dtype=np.int64),
        }
    )
    options = meta.MetaLabelOptions(n_estimators=5, seed=1, collect_fits=False)
    values = frame[FEATURES].to_numpy(dtype=float)
    fitters = ensembles.arm_fitters(trees=5)
    assert set(fitters) == {"single", "rf", "et", "uni", "seq"}
    for name, fit in fitters.items():
        probabilities = np.asarray(fit(frame, FEATURES, options).predict_proba(values))
        assert probabilities.shape == (rows, 2), name
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0)), name
