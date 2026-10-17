"""三重障碍标签的用例。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.labeling import labels


def _series(returns: list[float], start: float = 100.0) -> pd.Series:
    """按给定收益率序列造价格：returns[k] 是第 k 根到第 k+1 根的收益。"""
    prices = [start]
    for ret in returns:
        prices.append(prices[-1] * (1.0 + ret))
    index = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.Series(prices, index=index, dtype=float)


def _noise(count: int, amplitude: float = 0.001) -> list[float]:
    """交替涨跌的噪声：波动率可预测且不触发宽障碍。"""
    return [amplitude if i % 2 == 0 else -amplitude for i in range(count)]


def _options(**overrides: float | int) -> labels.TripleBarrierOptions:
    base: dict[str, float | int] = {"vol_span": 2, "vertical": 5}
    base.update(overrides)
    return labels.TripleBarrierOptions(**base)  # type: ignore[arg-type]


def _start_position(close: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    """把 frame 的行映射回 close 里的位置。

    前几根的 EWMA 波动率是 NaN（ddof=1 的单点方差），会被过滤掉，
    所以 frame 的行号**不等于** close 的位置号——必须按索引回查。
    """
    return close.index.get_indexer(frame.index)


def _offset(close: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    return frame["t1"].to_numpy() - _start_position(close, frame)


def test_short_series_has_no_events_but_keeps_the_columns() -> None:
    frame = labels.triple_barrier(_series([0.01] * 3), _options())
    assert list(frame.columns) == list(labels.BARRIER_COLUMNS)
    assert frame["label"].isna().to_numpy().all()


def test_validate_rejects_degenerate_options() -> None:
    for bad in (
        {"profit_take": 0.0},
        {"stop_loss": -1.0},
        {"vertical": 0},
        {"vol_span": 1},
    ):
        with pytest.raises(ValueError):
            _options(**bad).validate()


def test_side_must_be_one_or_minus_one() -> None:
    with pytest.raises(ValueError):
        labels.triple_barrier(_series(_noise(30)), _options(), side=0)


def test_flat_prices_produce_no_events_because_volatility_is_zero() -> None:
    """波动率为 0 时障碍宽度为 0，不该硬造事件。"""
    frame = labels.triple_barrier(_series([0.0] * 40), _options())
    assert frame["label"].isna().to_numpy().all()


def test_upward_jump_labels_long_as_profit_take() -> None:
    close = _series(_noise(25) + [0.10] + _noise(5))
    frame = labels.triple_barrier(close, _options(), side=1)
    event = frame.loc[close.index[24]]  # 起跳前一根
    assert event["label"] == 1
    # returns[25] 把 prices[25] 拉到 prices[26]，所以触及发生在位置 26
    assert event["t1"] == 26
    assert event["ret"] > 0


def test_downward_jump_labels_long_as_stop_loss() -> None:
    close = _series(_noise(25) + [-0.10] + _noise(5))
    frame = labels.triple_barrier(close, _options(), side=1)
    event = frame.loc[close.index[24]]
    assert event["label"] == -1
    assert event["t1"] == 26
    assert event["ret"] < 0


def test_short_side_mirrors_the_barriers() -> None:
    """做空时价格**上涨**才是止损，下跌才是止盈。"""
    up = labels.triple_barrier(
        _series(_noise(25) + [0.10] + _noise(5)), _options(), side=-1
    )
    down = labels.triple_barrier(
        _series(_noise(25) + [-0.10] + _noise(5)), _options(), side=-1
    )
    up_close = _series(_noise(25) + [0.10] + _noise(5))
    down_close = _series(_noise(25) + [-0.10] + _noise(5))
    assert up.loc[up_close.index[24]]["label"] == -1
    assert down.loc[down_close.index[24]]["label"] == 1


def test_vertical_barrier_ends_the_path_when_no_barrier_is_hit() -> None:
    """障碍宽到不可能触及，标签必须落在垂直障碍上。"""
    close = _series(_noise(40))
    options = _options(profit_take=100.0, stop_loss=100.0, vertical=5)
    frame = labels.triple_barrier(close, options)
    assert set(_offset(close, frame).tolist()) == {5}


def test_label_is_always_the_sign_of_the_return_at_touch() -> None:
    """作者的 getBins 口径：标签是触及时刻收益的符号，不是「哪个障碍被触及」。"""
    frame = labels.triple_barrier(_series(_noise(60)), _options())
    assert (frame["label"].to_numpy() == np.sign(frame["ret"].to_numpy())).all()


def test_t1_never_exceeds_the_vertical_window() -> None:
    close = _series(_noise(60))
    frame = labels.triple_barrier(close, _options())
    offset = _offset(close, frame)
    assert (offset >= 1).all()
    assert (offset <= 5).all()


def test_barrier_width_is_proportional_to_volatility() -> None:
    """宽度 = profit_take × 波动率：相对宽度比上波动率应是常数。"""
    close = _series(_noise(60, 0.02))
    frame = labels.triple_barrier(close, _options(profit_take=2.0))
    entry = close.to_numpy()[_start_position(close, frame)]
    relative = (frame["upper"].to_numpy() / entry - 1.0) / frame["vol"].to_numpy()
    assert relative == pytest.approx(2.0)


def test_louder_series_gets_wider_barriers() -> None:
    quiet = labels.triple_barrier(_series(_noise(60, 0.001)), _options())
    loud = labels.triple_barrier(_series(_noise(60, 0.05)), _options())
    assert loud["vol"].median() > 10 * quiet["vol"].median()


def test_barriers_bracket_the_entry_price() -> None:
    close = _series(_noise(40))
    frame = labels.triple_barrier(close, _options(), side=1)
    entry = close.to_numpy()[_start_position(close, frame)]
    assert (frame["upper"].to_numpy() > entry).all()
    assert (frame["lower"].to_numpy() < entry).all()


def test_barriers_swap_under_short_side() -> None:
    """做空的「上障碍」是更低的价格：upper < entry < lower。"""
    close = _series(_noise(40))
    frame = labels.triple_barrier(close, _options(), side=-1)
    entry = close.to_numpy()[_start_position(close, frame)]
    assert (frame["upper"].to_numpy() < entry).all()
    assert (frame["lower"].to_numpy() > entry).all()


def test_ewma_volatility_matches_pandas() -> None:
    close = _series(_noise(30, 0.01))
    expected = close.pct_change().ewm(span=7).std()
    pd.testing.assert_series_equal(labels.ewma_volatility(close, 7), expected)


def _naive_barriers(
    close: pd.Series, options: labels.TripleBarrierOptions, side: int = 1
) -> tuple[list[float], list[float], list[float]]:
    """逐事件循环，按作者 applyPtSlOnT1 的写法——给向量化实现对拍用。

    作者的循环体是「路径收益 = 价格 / 起点价 − 1，再乘 side」，然后取
    `path[path > pt].index.min()` 与 `path[path < sl].index.min()` 的较早者。
    这里保持逐事件、逐个候选位置比较，不做任何前缀和或批量求解。
    """
    values = close.to_numpy(dtype=float)
    vol = labels.ewma_volatility(close, options.vol_span).to_numpy(dtype=float)
    bins: list[float] = []
    touches: list[float] = []
    returns: list[float] = []
    for start in range(len(close) - options.vertical):
        width = vol[start]
        if not np.isfinite(width) or width <= 0:
            continue
        window = values[start + 1 : start + options.vertical + 1]
        path = (window / values[start] - 1.0) * side
        up = np.flatnonzero(path > options.profit_take * width)
        down = np.flatnonzero(path < -options.stop_loss * width)
        candidates = [hit[0] for hit in (up, down) if hit.size]
        column = min(candidates) if candidates else len(path) - 1
        bins.append(float(np.sign(path[column])))
        touches.append(float(start + column + 1))
        returns.append(float(path[column]))
    return bins, touches, returns


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("vertical", [3, 7, 20])
def test_vectorized_matches_the_naive_loop(side: int, vertical: int) -> None:
    """对拍：路径依赖的首次触及是向量化最容易藏错的地方。"""
    generator = np.random.default_rng(0)
    steps = generator.normal(0.0, 0.03, size=400)
    close = _series(steps.tolist())
    options = _options(vertical=vertical, vol_span=20, profit_take=1.5, stop_loss=0.8)
    frame = labels.triple_barrier(close, options, side=side)
    bins, touches, returns = _naive_barriers(close, options, side=side)
    assert frame["label"].to_numpy().tolist() == [int(b) for b in bins]
    assert frame["t1"].to_numpy().tolist() == [int(t) for t in touches]
    assert frame["ret"].to_numpy() == pytest.approx(returns)
