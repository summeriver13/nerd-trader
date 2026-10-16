"""meme 体检测试：只覆盖会静默出错的地方。

四个高风险点（错了都不报错，只会让结论变好看）：

1. **暴涨暴跌的前瞻收益从 t+1 起**——把事件日自己算进去，「当天涨 10%」会被当成
   「事件后涨 10%」，反转结论直接变成动量结论。
2. **相关性只用重叠窗口**——各标的起点不同，把缺失日当 0 收益会稀释 beta，
   把「1.6 倍的 BTC」读成「跟跌不跟涨」。
3. **生命周期分段独立**——写成累计会把「每段都在跌」读成「越跌越少」。
4. **冻结日必须剔除**——零成交日的 OHLC 是上一笔的复制，留着会在分段边界伪造涨跌
   （carry 风险研究里实测过 ADA 的 +113% 伪涨幅）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.meme_bars import CoinBars, load_bars, load_universe
from src.research.meme_profile import (
    _forward_return,
    benchmark_table,
    burst_table,
    lifecycle_table,
    profile_table,
)

START = "2022-01-01"
BENCHMARK = "BTC-USDT"


def _coin(symbol: str, closes: list[float], start: str = START) -> CoinBars:
    """合成标的：收盘价序列即全部输入。"""
    index = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC")
    return CoinBars(symbol=symbol, close=pd.Series(closes, index=index))


def _closes(start: float, rates: list[float]) -> list[float]:
    """由起始价与逐根收益率生成收盘价序列（n 根收益 → n+1 根收盘）。"""
    closes = [start]
    for rate in rates:
        closes.append(closes[-1] * (1.0 + rate))
    return closes


def test_forward_return_starts_after_event_day() -> None:
    """事件日自己不能进前瞻窗口——否则「当天涨 10%」被算成「事件后涨 10%」。"""
    returns = np.array([0.10, 0.20, -0.50])
    assert _forward_return(returns, np.array([0]), 1) == pytest.approx([0.20])
    # 两根窗口：事件日之后的两根复利 = 1.2 × 0.5 − 1
    assert _forward_return(returns, np.array([0]), 2) == pytest.approx([-0.40])


def test_forward_return_drops_short_tail() -> None:
    """末尾不够窗口的样本丢弃，不拿半截窗口凑数。"""
    returns = np.array([0.10, 0.20, 0.30])
    # 位置 2 之后没有下一根，位置 1 之后只剩一根：两者都不够窗口，必须丢弃
    assert _forward_return(returns, np.array([2]), 1).size == 0
    assert _forward_return(returns, np.array([1]), 2).size == 0
    assert _forward_return(returns, np.array([0]), 2).size == 1


def test_burst_table_hand_computed() -> None:
    """暴涨后次日 = 事件日的下一根；基准是无条件中位，用来分辨漂移与反转。"""
    closes = [100.0, 140.0, 112.0, 106.0, 118.0, 120.0, 110.0, 121.0]
    table = burst_table({"X-USDT": _coin("X-USDT", closes)})
    row = table.loc["X-USDT"]
    assert float(row["up_n"]) == 3  # +40%、+11.32%、+10%
    assert float(row["up_t1"]) == pytest.approx((-0.20 + 0.016949153) / 2, rel=1e-4)
    assert float(row["down_n"]) == 1  # 只有 −20% 那一天过 10% 门槛
    assert float(row["down_t1"]) == pytest.approx(-0.053571428, rel=1e-4)
    # 基准 = 全部次日收益的中位（最后一天没有次日，不参与）
    assert float(row["base_t1"]) == pytest.approx(
        (-0.053571428 + 0.016949153) / 2, rel=1e-4
    )


def test_burst_threshold_is_on_event_day_not_forward() -> None:
    """门槛只看事件日：次日涨幅再大也不该被当成事件。"""
    closes = [100.0, 105.0, 130.0, 140.0]
    table = burst_table({"X-USDT": _coin("X-USDT", closes)})
    assert float(table.loc["X-USDT", "up_n"]) == 1  # 只有最后一根 +23.8% 过线


def test_lifecycle_segments_are_independent() -> None:
    """每段从自身起点重算：累计口径的末段会等于全期总收益。"""
    closes = [100.0 * 1.01**bar for bar in range(400)]
    table = lifecycle_table({"X-USDT": _coin("X-USDT", closes)})
    assert float(table.loc["1-7", "median"]) == pytest.approx(1.01**7 - 1)
    assert float(table.loc["8-30", "median"]) == pytest.approx(1.01**23 - 1)
    assert float(table.loc["31-90", "median"]) == pytest.approx(1.01**60 - 1)
    # 末段开到末尾：400 根里第 90→399 根
    assert float(table.loc["91+", "median"]) == pytest.approx(
        closes[-1] / closes[90] - 1
    )


def test_lifecycle_skips_coins_without_enough_bars() -> None:
    """根数不够的标的从该段的样本里剔除，不拿现有根数凑（凑数会让长段凭空变短）。"""
    long_coin = _coin("L-USDT", [100.0 * 1.01**bar for bar in range(400)])
    short_coin = _coin("S-USDT", [100.0, 90.0, 80.0])
    table = lifecycle_table({"L-USDT": long_coin, "S-USDT": short_coin})
    assert float(table.loc["1-7", "count"]) == 1  # 3 根够 1-7 段（持有到第 7 根不够）
    assert float(table.loc["91+", "count"]) == 1


def test_benchmark_beta_uses_overlap_only() -> None:
    """coin = 2 × BTC 时 beta 必须正好是 2：起点不同不能把缺失日当 0 收益。"""
    btc_rates = [0.01, -0.02, 0.03, -0.04, 0.05]
    coin_rates = [rate * 2.0 for rate in btc_rates[2:]]  # 只与 BTC 的后 3 天重叠
    table = benchmark_table(
        {"X-USDT": _coin("X-USDT", _closes(200.0, coin_rates), start="2022-01-03")},
        _coin(BENCHMARK, _closes(100.0, btc_rates)),
    )
    row = table.loc["X-USDT"]
    assert float(row["days"]) == 3  # 只有 3 天重叠（4 根收盘 → 3 根收益）
    assert float(row["beta"]) == pytest.approx(2.0)
    assert float(row["corr"]) == pytest.approx(1.0)
    assert float(row["r2"]) == pytest.approx(1.0)
    assert float(row["alpha_annual"]) == pytest.approx(0.0, abs=1e-12)


def test_benchmark_flat_coin_yields_nan() -> None:
    """价格横住的死币不能让 corrcoef 除零：相关无从定义（nan），beta 是 0 不是 nan。

    横住的币确实不跟跌（beta=0），但「相关」在零方差下没有定义——两者不能混成一个值。
    """
    btc_rates = [0.01, -0.02, 0.03]
    table = benchmark_table(
        {"X-USDT": _coin("X-USDT", [100.0, 100.0, 100.0, 100.0])},
        _coin(BENCHMARK, _closes(100.0, btc_rates)),
    )
    row = table.loc["X-USDT"]
    assert np.isnan(float(row["corr"]))
    assert float(row["beta"]) == pytest.approx(0.0)


def test_benchmark_splits_up_and_down_days() -> None:
    """上涨日/下跌日的 beta 分开算：单一 beta 会把「跌得更多」平均掉。"""
    btc_rates = [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04]
    coin_rates = [rate * (3.0 if rate < 0 else 1.0) for rate in btc_rates]
    table = benchmark_table(
        {"X-USDT": _coin("X-USDT", _closes(100.0, coin_rates))},
        _coin(BENCHMARK, _closes(100.0, btc_rates)),
    )
    row = table.loc["X-USDT"]
    assert float(row["beta_up"]) == pytest.approx(1.0)
    assert float(row["beta_down"]) == pytest.approx(3.0)


def test_profile_table_handles_drawdown_and_high() -> None:
    """回撤与距高点用首根起的收盘价算，与复权、分红无关。"""
    table = profile_table({"X-USDT": _coin("X-USDT", [100.0, 50.0, 75.0, 60.0])})
    row = table.loc["X-USDT"]
    assert float(row["total_return"]) == pytest.approx(-0.40)
    assert float(row["max_drawdown"]) == pytest.approx(-0.50)
    assert float(row["from_high"]) == pytest.approx(-0.40)  # 60/100 − 1


def test_load_bars_drops_frozen_days(tmp_path: Path) -> None:
    """零成交的冻结日必须剔除：它的 OHLC 是上一笔的复制，留着会伪造分段收益。"""
    index = pd.date_range(START, periods=4, freq="1D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 120.0],
            "high": [100.0, 100.0, 100.0, 120.0],
            "low": [100.0, 100.0, 100.0, 120.0],
            "close": [100.0, 100.0, 100.0, 120.0],
            "volume": [5.0, 0.0, 0.0, 5.0],
        },
        index=index,
    )
    target = tmp_path / "okx" / "spot" / "1d"
    target.mkdir(parents=True)
    frame.to_parquet(target / "X-USDT.parquet")

    bars = load_bars(("X-USDT",), root=tmp_path)
    assert bars["X-USDT"].bars == 2
    assert float(bars["X-USDT"].returns.iloc[0]) == pytest.approx(0.20)


def test_load_bars_fails_on_missing_symbol(tmp_path: Path) -> None:
    """少一只必须显式失败——静默变成 11 只会让「12 只里 4 只为正」这类结论失真。"""
    with pytest.raises(FileNotFoundError):
        load_bars(("X-USDT",), root=tmp_path)


def test_load_universe_keeps_benchmark_out_of_pool(tmp_path: Path) -> None:
    """基准不进池子：混进去会让生命周期与暴涨暴跌的中位数多算一只非 meme 标的。"""
    target = tmp_path / "okx" / "spot" / "1d"
    target.mkdir(parents=True)
    for symbol in ("X-USDT", BENCHMARK):
        frame = pd.DataFrame(
            {"close": 100.0, "volume": 5.0},
            index=pd.date_range(START, periods=3, freq="1D", tz="UTC"),
        )
        frame.to_parquet(target / f"{symbol}.parquet")

    bars, benchmark = load_universe(("X-USDT",), benchmark=BENCHMARK, root=tmp_path)
    assert list(bars) == ["X-USDT"]
    assert benchmark.symbol == BENCHMARK
