"""
`observations.build_meta_frame` 的用例：研究脚本的数据集口径。

测的是**筛选口径**（列齐不齐、可交易与非零方向有没有滤干净、有没有把缺失当信号），
不是因子数值——所以面板是合成的手算行情（见 conftest）。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from src.research import labels, observations
from src.research.factors.panel import Panel

type PanelFactory = Callable[[dict[str, list[float]]], Panel]

# 池子要求上市 ≥180 天，特征最长暖机 60 根：面板必须够长，否则整表为空
BARS = 220
NAMES = 6
FEATURE_KEYS: tuple[str, ...] = ("mom_7", "vol_30", "dth_60")
STUDY_COLUMNS: tuple[str, ...] = (
    "side",
    "forward",
    "barrier_label",
    "weight_unique",
    "tradable",
    observations.AGE_COLUMN,
    *FEATURE_KEYS,
)


def _panel(panel_factory: PanelFactory) -> Panel:
    """锯齿行情：涨跌交替，正反方向都有观测，不会退化成单边。"""
    closes = {
        f"N{index:02d}-USDT": [
            100.0 + index + (0.5 if bar % 2 else -0.5) * (bar // 2)
            for bar in range(BARS)
        ]
        for index in range(NAMES)
    }
    return panel_factory(closes)


def _frame(panel: Panel) -> pd.DataFrame:
    return observations.build_meta_frame(
        panel, primary_key="rev_3", feature_keys=FEATURE_KEYS, vertical=3
    )


def test_to_long_round_trips_a_wide_frame() -> None:
    """宽表→长表必须可逆：列名、顺序、(date, symbol) 键都要对得上。"""
    wide = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0]},
        index=pd.date_range("2020-01-01", periods=2, tz="UTC"),
    )
    long = observations.to_long(wide, "value")
    assert list(long.columns) == ["date", "symbol", "value"]
    # stack 按列优先：先 A 的两行再 B 的两行
    assert long["value"].tolist() == [1.0, 3.0, 2.0, 4.0]
    back = long.pivot(index="date", columns="symbol", values="value")
    pd.testing.assert_frame_equal(
        back[wide.columns], wide, check_names=False, check_freq=False
    )


def test_build_meta_frame_carries_every_study_column(
    panel_factory: PanelFactory,
) -> None:
    frame = _frame(_panel(panel_factory))
    assert set(STUDY_COLUMNS) <= set(frame.columns)
    assert len(frame) > 0
    # 行序是「标的优先」（`build_observations` 按标的拼），要按日期说话得自己排序；
    # 但一行一只标的、一天只出现一次是硬约束，重复会让权重与标签静默错位
    assert not frame.duplicated(subset=["date", "symbol"]).any()


def test_build_meta_frame_drops_zero_sides_and_missing_labels(
    panel_factory: PanelFactory,
) -> None:
    """主模型没观点的行必须丢掉——元标签只回答「这一注要不要下」。"""
    frame = _frame(_panel(panel_factory))
    assert (frame["side"] != 0).all()
    assert bool(frame[["side", "forward", "barrier_label"]].notna().to_numpy().all())
    assert set(frame["side"].unique()) <= {-1.0, 1.0}


def test_build_meta_frame_never_leaves_a_missing_feature(
    panel_factory: PanelFactory,
) -> None:
    """特征列一条缺失都不许有。

    这一条不是显然的：`dropna` 只点名了 `side`/`forward`/`barrier_label` 三列，特征是否
    干净全靠「池子的满窗流动性要求（30 日）比特征暖机（最长 60 根）更严」+ 内连接。
    真实 swap 面板上实测 168442 行零缺失；这里把它钉住，免得哪天改了池子口径，
    缺失值静默进训练集（树模型收 NaN 不报错，只会悄悄换一种分裂方式）。
    """
    frame = _frame(_panel(panel_factory))
    assert int(frame[list(FEATURE_KEYS)].isna().sum().sum()) == 0
    assert int(frame[observations.AGE_COLUMN].isna().sum()) == 0


def test_build_meta_frame_respects_the_tradable_window(
    panel_factory: PanelFactory,
) -> None:
    """上市不满 180 天的那一段不能进表——池子口径必须真的生效。"""
    panel = _panel(panel_factory)
    frame = _frame(panel)
    assert bool(frame["tradable"].all())
    listed = (frame["date"].min() - panel.index[0]).days
    assert listed >= 180
    assert len(frame) < panel.close.notna().to_numpy().sum()  # 确实滤掉了东西


def test_trend_labels_follow_each_symbols_own_bars(
    panel_factory: PanelFactory,
) -> None:
    """面板级趋势扫描：行数按各标的**有报价的根**算，太短的标的整只跳过。

    它是标签不是信号（`trend_label` 用到了未来窗口），所以这里只钉口径：覆盖哪些行、
    暖机处留 NaN、一只都不够长时抛错而不是惄惄返回空表。
    """
    panel = _panel(panel_factory)
    long = observations.trend_labels(panel, labels.TrendScanOptions(span=10))
    assert set(long["symbol"]) == set(panel.close.columns)
    counts = long.groupby("symbol")["trend_label"].size()
    expected = {
        symbol: int(panel.close[symbol].notna().sum()) for symbol in panel.close.columns
    }
    assert counts.to_dict() == expected
    assert int(long["trend_label"].isna().sum()) > 0
    assert int(long["trend_label"].notna().sum()) > 0
    with pytest.raises(ValueError):
        observations.trend_labels(
            panel_factory({"SHORT-USDT": [100.0] * 5}),
            labels.TrendScanOptions(span=10),
        )
