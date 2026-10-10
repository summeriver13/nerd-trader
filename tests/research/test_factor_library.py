"""因子库的三条不变量：预热下标、方向已并入取值、只吃历史。

历史教训：因子若混入未来信息（前向 shift、把当根收盘价当已知），回测不会报错，
IC 会漂亮到不像话。唯一可靠的防线是「篡改未来、过去必须不变」这类不变量测试。
"""

import numpy as np
import pandas as pd
import pytest

from src.research.factors.library import FACTOR_LIBRARY, resolve_factors
from src.research.factors.panel import Panel

BARS = 200


def _panel(closes: dict[str, list[float]]) -> Panel:
    """合成面板：名义额固定为 100 万美元级别，只有价格在动。"""
    length = len(next(iter(closes.values())))
    index = pd.date_range("2020-01-01", periods=length, freq="1D", tz="UTC")
    close = pd.DataFrame(closes, index=index)
    volume = pd.DataFrame(1e6, index=index, columns=close.columns)
    return Panel(timeframe="1d", close=close, volume=volume, notional=close * volume)


def _rising(symbol: str) -> Panel:
    return _panel({symbol: list(np.linspace(100.0, 300.0, BARS))})


def test_warmup_is_exactly_first_valid_index() -> None:
    """warmup 就是第一个有效值的下标：前面全 NaN，那根起有值。"""
    panel = _rising("A-USDT")
    for spec in FACTOR_LIBRARY.values():
        values = spec.values(panel)["A-USDT"]
        assert values.iloc[: spec.warmup].isna().all(), f"{spec.key} 预热前应为 NaN"
        assert values.iloc[spec.warmup :].notna().all(), f"{spec.key} 预热后应有值"


def test_momentum_matches_hand_calculation() -> None:
    """动量 = 窗口两端价格比减一；反转是同一动量的负值（方向在这一层翻转）。"""
    panel = _panel({"A-USDT": [100.0 + 10.0 * step for step in range(8)]})

    momentum = FACTOR_LIBRARY["mom_7"].values(panel)["A-USDT"]
    assert np.isnan(momentum.iloc[6])
    assert momentum.iloc[7] == pytest.approx(170.0 / 100.0 - 1)

    reversal = FACTOR_LIBRARY["rev_3"].values(panel)["A-USDT"]
    assert reversal.iloc[3] == pytest.approx(-(130.0 / 100.0 - 1))


def test_direction_is_baked_into_values() -> None:
    """低波动标的的 vol 因子值更大：方向已在 values() 里翻转。"""
    calm = np.linspace(100.0, 110.0, BARS)
    wild = 100.0 + 10.0 * np.sin(np.linspace(0.0, 20.0 * np.pi, BARS))
    panel = _panel({"CALM-USDT": list(calm), "WILD-USDT": list(wild)})
    spec = FACTOR_LIBRARY["vol_30"]

    signed = spec.values(panel).iloc[-1]
    raw = spec.compute(panel).iloc[-1]
    assert signed["CALM-USDT"] > signed["WILD-USDT"]
    assert raw["CALM-USDT"] < raw["WILD-USDT"]


def test_distance_to_high_never_positive() -> None:
    """距高点因子只能 ≤ 0（当根即最高点时等于 0）。"""
    values = FACTOR_LIBRARY["dth_60"].values(_rising("A-USDT")).dropna()
    assert (values <= 1e-12).all().all()


def test_factors_read_only_the_past() -> None:
    """篡改第 60 根之后的价格，前 60 根的因子值必须一模一样。"""
    base = [100.0 + i for i in range(BARS)]
    cut = 60
    intact = _panel({"A-USDT": base})
    tampered = _panel({"A-USDT": base[:cut] + [value * 3 for value in base[cut:]]})

    for spec in FACTOR_LIBRARY.values():
        left = spec.values(intact).iloc[:cut]
        right = spec.values(tampered).iloc[:cut]
        pd.testing.assert_frame_equal(left, right)


def test_resolve_factors_defaults_to_library() -> None:
    """省略键取全库；未知键当场报错，不静默忽略。"""
    assert [spec.key for spec in resolve_factors()] == list(FACTOR_LIBRARY)
    assert [spec.key for spec in resolve_factors("mom_7, vol_30")] == [
        "mom_7",
        "vol_30",
    ]

    with pytest.raises(KeyError, match="未知因子"):
        resolve_factors("mom_7,not_a_factor")


def test_factor_keys_match_spec_fields() -> None:
    """注册表的键必须与 spec.key 一致（否则 CLI 解析与报告会对不上）。"""
    for key, spec in FACTOR_LIBRARY.items():
        assert key == spec.key
