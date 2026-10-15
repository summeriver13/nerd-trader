"""三口径对照与路径 bootstrap 的口径测试（全部合成数据）。

抓的是真出过错的边界：Deribit 到期码的日可以是一位数（6NOV26）、永续不是到期合约、
隐含 σ 反解必须收敛（漂移依赖 σ，不迭代会差 2~3pp）、去漂移后上下行要对称。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.research.polymarket_digital import (
    Smile,
    comparison_table,
    deribit_touch,
    implied_touch_sigma,
    smile_from_chain,
)
from src.research.polymarket_ladder import ladder_row
from src.research.price_bootstrap import BootstrapSpec, PathStats, simulate_paths
from src.utils.data_sources.base import parse_instant
from src.utils.data_sources.deribit import (
    DeribitChain,
    OptionQuote,
    expiry_instant,
    forward_curve,
    is_expiry_code,
    parse_option,
)

RULE_TEXT = (
    'This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin '
    '(BTC/USDT) between November 24, 2025, 14:00 and December 31, 2026, 23:59 in the ET timezone '
    'has a final "High" price equal to or greater than the price specified in the title.'
)


def test_expiry_instant_handles_one_digit_day_and_rejects_perpetual() -> None:
    """6NOV26 与 25DEC26 都要能解析；PERPETUAL 不是到期合约（当年份会算出无穷 carry）。"""
    assert expiry_instant("6NOV26") == pd.Timestamp("2026-11-06T08:00:00Z")
    assert expiry_instant("25DEC26") == pd.Timestamp("2026-12-25T08:00:00Z")
    assert not is_expiry_code("PERPETUAL") and is_expiry_code("6NOV26")


def test_parse_option_reads_iv_as_fraction_and_zero_bid_as_none() -> None:
    quote = parse_option(
        {
            "instrument_name": "BTC-25DEC26-82000-C",
            "mark_price": 0.05,
            "mark_iv": 37.4,
            "bid_price": 0.0,
            "ask_price": 0.04,
            "open_interest": 12.0,
            "underlying_price": 81000.0,
        }
    )
    assert quote is not None
    assert quote.mark_iv == pytest.approx(0.374)
    assert quote.bid is None and quote.ask == 0.04
    assert parse_option({"instrument_name": "BTC-PERPETUAL"}) is None


def chain(fetched: str, futures: dict[str, float], options: list[OptionQuote]) -> DeribitChain:
    """造一条链：指数固定 100,000，只关心期货与期权字段。"""
    return DeribitChain(
        fetched_at=parse_instant(fetched, "fetched_at"),
        currency="BTC",
        index_price=100_000.0,
        futures=futures,
        options=tuple(options),
    )


def quote(strike: float, kind: str, vol: float, *, bid: float = 0.009, ask: float = 0.011):
    return OptionQuote(
        instrument=f"BTC-25DEC26-{strike:.0f}-{kind}",
        expiry="25DEC26",
        kind=kind,
        strike=strike,
        mark_price=0.01,
        mark_iv=vol,
        bid=bid,
        ask=ask,
        open_interest=1.0,
        underlying=80_000.0,
    )


def test_forward_curve_skips_perpetual_and_extrapolates_carry() -> None:
    """carry 必须由最近的到期期货定，永续进不了分母（否则 T 无穷大）。"""
    subject = chain(
        "2026-01-01T00:00:00Z",
        {"BTC-PERPETUAL": 100_100.0, "BTC-25DEC26": 105_000.0},
        [],
    )
    point = forward_curve(subject, parse_instant("2027-01-01T00:00:00Z", "目标"))
    assert point.via == "BTC-25DEC26"
    expected_carry = math.log(105_000.0 / 100_000.0) / (358 / 365)
    assert point.carry == pytest.approx(expected_carry, rel=1e-9)
    assert point.forward == pytest.approx(100_000.0 * math.exp(expected_carry), rel=1e-6)


def test_smile_uses_out_of_the_money_side_and_does_not_extrapolate() -> None:
    """K ≥ F 取 call、K < F 取 put；区间外返回 None，绝不外推。"""
    subject = chain(
        "2026-10-01T00:00:00Z",
        {},
        [
            quote(70_000.0, "C", 0.50),
            quote(70_000.0, "P", 0.45),
            quote(90_000.0, "C", 0.55),
            quote(90_000.0, "P", 0.60),
        ],
    )
    smile = smile_from_chain(subject, "25DEC26", forward=80_000.0)
    assert smile.vols.tolist() == [0.45, 0.55]
    assert smile.iv_at(90_000.0) == pytest.approx(0.55)
    # 插值自变量是 log(K/F) 而不是 K：平值近似（不是精确）落在两端中点
    assert smile.iv_at(80_000.0) == pytest.approx(0.5, abs=0.005)
    assert smile.iv_at(120_000.0) is None and smile.iv_at(60_000.0) is None


def test_smile_relative_spread_uses_nearest_listed_option() -> None:
    """对冲腿成本取离该档最近的虚值期权报价；没有双边报价就返回 None，不能当零成本。"""
    smile = Smile(
        forward=80_000.0,
        strikes=np.array([70_000.0, 90_000.0, 200_000.0]),
        vols=np.array([0.45, 0.55, 0.80]),
        bids=np.array([0.009, 0.011, 0.0]),
        asks=np.array([0.011, 0.013, 0.0]),
    )
    # 90,000 那档：bid 0.011 / ask 0.013 → 0.002 / 0.012
    assert smile.relative_spread_at(90_000.0) == pytest.approx(1 / 6)
    assert smile.relative_spread_at(205_000.0) is None


def test_implied_touch_sigma_recovers_input() -> None:
    """σ 反解必须收敛回原值——漂移依赖 σ，不迭代就会差 2~3pp。"""
    price = deribit_touch(
        spot=80_000.0, forward=81_000.0, horizon=0.21, strike=110_000.0, sigma=0.45
    )
    recovered = implied_touch_sigma(
        spot=80_000.0, forward=81_000.0, horizon=0.21, strike=110_000.0, price=price
    )
    assert recovered == pytest.approx(0.45, rel=1e-6)


def test_comparison_table_leaves_missing_deribit_quotes_empty() -> None:
    """Deribit 没有该档报价时留空，且不许假装 market_iv 存在（价格在边界上）。"""
    smile = Smile(
        forward=80_000.0,
        strikes=np.array([70_000.0, 90_000.0]),
        vols=np.array([0.45, 0.55]),
        bids=np.array([0.009, 0.011]),
        asks=np.array([0.011, 0.013]),
    )
    loaded = ladder_row(
        {
            "slug": "will-bitcoin-reach-90000-by-december-31-2026",
            "outcomePrices": '["0.5", "0.5"]',
            "volumeNum": 1.0,
            "description": RULE_TEXT,
            "createdAt": "2025-11-24T15:44:31Z",
            "endDate": "2027-01-01T00:00:00Z",
            "spread": "0.01",
            "bestBid": "0.499",
            "bestAsk": "0.501",
        }
    )
    assert loaded is not None
    rows = [
        replace(loaded, strike=90_000.0),
        replace(loaded, strike=500_000.0, price=1.0),
    ]
    inside, outside = comparison_table(
        rows, spot=80_000.0, horizon=0.21, smile=smile
    )
    assert inside.deribit_touch is not None and inside.beyond_deribit_band is False
    assert inside.diff_vs_deribit is not None and inside.market_iv is not None
    assert inside.deribit_option_spread == pytest.approx(1 / 6)
    assert inside.market_relative_spread == pytest.approx(0.01 / 0.5)
    assert outside.deribit_touch is None and outside.beyond_deribit_band is None
    assert outside.market_iv is None


def test_simulate_paths_is_deterministic_and_monotone_in_barrier() -> None:
    """同 seed 必须逐位可复现；障碍越远触碰率越低。"""
    returns = np.array([0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, 0.002] * 40)
    spec = BootstrapSpec(hours=48, paths=500, block_hours=8, seed=7)
    first, second = simulate_paths(returns, spec), simulate_paths(returns, spec)
    assert np.array_equal(first.max_log, second.max_log)
    assert first.touch_rate(0.001) >= first.touch_rate(0.05) >= first.touch_rate(0.20)
    assert first.touch_rate(-0.001) >= first.touch_rate(-0.20)


def test_simulate_paths_demean_removes_drift() -> None:
    """去漂移后上下行对称（零均值下 max 与 min 同分布）；保留漂移时上行偏高、下行偏低。"""
    returns = np.random.default_rng(0).normal(0.0002, 0.001, 1_000)
    common = {"hours": 48, "paths": 1_000, "block_hours": 8, "seed": 3}
    drifted = simulate_paths(returns, BootstrapSpec(demean=False, **common))
    flat = simulate_paths(returns, BootstrapSpec(demean=True, **common))
    assert drifted.touch_rate(0.01) > flat.touch_rate(0.01)
    assert drifted.touch_rate(-0.01) < flat.touch_rate(-0.01)
    assert flat.touch_rate(0.01) == pytest.approx(flat.touch_rate(-0.01), abs=0.05)


def test_simulate_paths_rejects_sample_shorter_than_block() -> None:
    with pytest.raises(ValueError, match="不足一个块"):
        simulate_paths(np.zeros(5), BootstrapSpec(hours=24, block_hours=8))


def test_path_stats_terminal_rate_flags_below() -> None:
    """末值口径必须能切到 P(S_T < K)：算成 P(S_T > K) 会让下行档全算成 1。"""
    stats = PathStats(
        max_log=np.array([0.1, 0.2]),
        min_log=np.array([-0.1, -0.2]),
        end_log=np.array([0.05, -0.05]),
    )
    assert stats.terminal_rate(0.0) == pytest.approx(0.5)
    assert stats.terminal_rate(0.0, below=True) == pytest.approx(0.5)
