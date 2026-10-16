"""合成床的不变量：真值要能在读数里看见，读数之间又不许错位。

用例一律用种子固定的小面板 + 只挑要断言的因子——``portfolio_stats`` 是逐期循环，因子数
是主要开销，跑全池 7 个因子会把闸门拖慢好几倍。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.factors.evaluation import spearman
from src.research.factors.panel import UniverseRule, tradable_mask
from src.research.synthetic import (
    BED_ANCHOR,
    BEDS,
    IC_MIRROR,
    IC_VISIBLE,
    SyntheticSpec,
    synthetic_panel,
)
from src.research.synthetic_checks import run_bed, truth_checks

TEST_DAYS = 520
TEST_KEYS = ("rev_3", "mom_7", "liq_30")


def _bed(key: str) -> SyntheticSpec:
    return next(spec for spec in BEDS if spec.key == key)


def test_panel_is_tradable_only_after_the_listing_floor() -> None:
    panel = synthetic_panel(_bed("noise"), 0)
    mask = tradable_mask(panel)
    assert not bool(mask.iloc[:BED_ANCHOR].to_numpy().any())
    assert bool(mask.iloc[BED_ANCHOR:].to_numpy().all())
    assert float(panel.notional.min().min()) >= UniverseRule().min_notional_usd


def test_notional_and_volatility_are_independent() -> None:
    """σ 与名义额不能同序：同序会让 liq_30 变成 vol_30 的替身（标定时真踩过）。"""
    panel = synthetic_panel(_bed("noise"), 0)
    returns = panel.close.pct_change().to_numpy()
    realized = pd.Series(np.nanstd(returns, axis=0)).rank().to_numpy()
    level = panel.notional.iloc[-1].rank().to_numpy()
    assert abs(spearman(realized, level)) < 0.4


def test_reversal_bed_lights_the_expected_factors() -> None:
    bed = run_bed(_bed("reversal"), 0, keys=TEST_KEYS, days=TEST_DAYS)
    ic = bed.ic.set_index("因子")
    assert float(ic.loc["rev_3", "IC"]) >= IC_VISIBLE
    assert float(ic.loc["mom_7", "IC"]) <= -IC_MIRROR
    assert all(check.hit for check in bed.checks())


def test_cost_flips_the_money_reading_on_the_same_prices() -> None:
    cheap = run_bed(_bed("reversal"), 0, keys=TEST_KEYS, days=TEST_DAYS)
    dear = run_bed(_bed("reversal_costly"), 0, keys=TEST_KEYS, days=TEST_DAYS)
    gross = float(cheap.money.set_index("因子").loc["rev_3", "毛(bp)"])
    assert float(dear.money.set_index("因子").loc["rev_3", "毛(bp)"]) == pytest.approx(
        gross
    )
    assert float(cheap.money.set_index("因子").loc["rev_3", "净(bp)"]) > 0.0
    assert float(dear.money.set_index("因子").loc["rev_3", "净(bp)"]) < 0.0


def test_noise_bed_reads_silent() -> None:
    bed = run_bed(_bed("noise"), 0, keys=TEST_KEYS, days=TEST_DAYS)
    assert all(check.hit for check in bed.checks())


def test_spread_matrix_shares_one_grid() -> None:
    """同 anchor 才拼得成家族矩阵：相位错开会让每行只剩一个非 NaN（README 第二十四节）。"""
    reads = run_bed(_bed("noise"), 0, keys=TEST_KEYS, days=TEST_DAYS)
    assert reads.spreads.shape[0] == int(reads.money["期数"].iloc[0])
    empty_rows = np.asarray(reads.spreads.isna().all(axis=1))
    assert not bool(empty_rows.any())


def test_truth_checks_route_readings_to_claims() -> None:
    """纯逻辑用例：读数表怎么填，检查项就该怎么落。"""
    spec = SyntheticSpec(
        key="t",
        label="t",
        reversal=0.0,
        expected=("rev_3",),
        mirror=("mom_7",),
        profitable=True,
    )
    ic = pd.DataFrame(
        [
            {"因子": "rev_3", "IC": 0.05},
            {"因子": "mom_7", "IC": -0.03},
            {"因子": "liq_30", "IC": 0.004},
        ]
    )
    money = pd.DataFrame(
        [
            {"因子": "rev_3", "净(bp)": 12.0},
            {"因子": "mom_7", "净(bp)": -3.0},
            {"因子": "liq_30", "净(bp)": 1.0},
        ]
    )
    hits = {
        (check.layer, check.subject): check.hit
        for check in truth_checks(spec, ic, money)
    }
    assert hits[("该亮", "rev_3")]
    assert hits[("镜像", "mom_7")]
    assert hits[("应沉默", "liq_30")]
    assert hits[("钱", "rev_3")]
    worse = pd.DataFrame([{"因子": "rev_3", "净(bp)": -2.0}])
    assert not truth_checks(spec, ic, worse)[-1].hit
