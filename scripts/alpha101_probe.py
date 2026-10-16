"""
scripts/alpha101_probe.py
Alpha101 可行性探针：把论文当**算子库**而不是策略库，量一量直接复刻的期望值。

用法：
    uv run python scripts/alpha101_probe.py                     # 四个探针全跑
    uv run python scripts/alpha101_probe.py --probe base
    uv run python scripts/alpha101_probe.py --probe decay
    uv run python scripts/alpha101_probe.py --horizons 1,3,7,30

四个探针（`docs/factors/README.md` 第二十八节的出处）：
- `base`    7 条纯 OHLCV 条目的 IC / IC t / 日换手 / 平衡成本 / 净年化；
- `horizon` 把调仓放慢到 h=1/3/7/30，**穷举每个 h 的全部 h 个网格相位**报中位与范围：
  只报单相位就是那个「28 格里挑最大值」的陷阱（实测同一条 #23、h=3，从第 0 行起算的
  毛年化 14.4%、从第 10 行起算 3.3%）；
- `decay`   用 `decay_linear` 抢救本仓 `rev_3`（毛边际真实、被成本吃掉的那个）。

口径（四条，与主库一致，任一条不满足读数就不可执行）：
- 信号当根收盘确认、最早下一根成交：网格里权重取 `values.iloc[t-1]`、收益取
  `close[t+h]/close[t]-1`（少这根延迟会把快信号的读数翻号）；
- 权重 = 截面 rank 去均值（多空等权），平衡成本 = 毛价差 / (2 × 换手)；
- 成本 = `PERP_FEES.taker_net()`（永续 taker + 滑点 = 10bp/边）；
- 只吃 OHLCV 与面板：`vwap`/`cap`/`IndNeutralize` 类条目不在清单内（本仓无对应数据）。

**这不是验证。** 探针只回答「这条线值不值得开」，要立结论得走 factor_verify.py。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd

from src.config.paths import crypto_dataset_dir
from src.core.costs import PERP_FEES
from src.research.factors.operators import correlation, decay_linear
from src.research.factors.panel import Panel, load_panel
from src.research.factors.portfolio import break_even_cost

BARS_PER_YEAR: Final[int] = 365
MIN_BARS: Final[int] = 365
MIN_NAMES: Final[int] = 10
OHLCV_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low")
HORIZON_SCAN: Final[tuple[int, ...]] = (1, 3, 7, 30)
PHASE_HORIZON: Final[int] = 30
DECAY_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
INTERACTION_WINDOWS: Final[dict[str, int]] = {"#6": 10, "#44": 5}
PROBES: Final[tuple[str, ...]] = ("base", "horizon", "decay")
COST: Final[float] = PERP_FEES.taker_net()


@dataclass(frozen=True)
class ProbeData:
    """探针用的数据：面板 + 从 parquet 补的 OHLC（panel 只有 close/volume/notional）。"""

    panel: Panel
    symbols: list[str]
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame

    @property
    def close(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.panel.close[self.symbols])

    @property
    def volume(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.panel.volume[self.symbols])


def cross_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """截面 rank 到 [0,1]（Alpha101 的 rank 算子）。"""
    return frame.rank(axis=1, pct=True)


def sign(frame: pd.DataFrame) -> pd.DataFrame:
    """逐元素符号（−1/0/+1，Alpha101 的 sign 算子）。"""
    return cast(pd.DataFrame, np.sign(frame))


def rank_within(row: pd.Series) -> pd.Series:
    """单根（一天）内的 rank：网格取样时的权重构造。"""
    return row.rank(pct=True)


def load_probe_data(market: str, timeframe: str) -> ProbeData:
    """面板只给 close/volume，OHLC 另三类字段从同一份 parquet 缓存里补。"""
    panel = load_panel(timeframe, market=market)
    symbols = [s for s in panel.symbols if panel.close[s].notna().sum() >= MIN_BARS]
    root = crypto_dataset_dir("okx", market, timeframe)
    columns = {
        field: pd.DataFrame(
            {path.stem: pd.read_parquet(path)[field] for path in sorted(root.glob("*.parquet"))}
        ).reindex(panel.index)
        for field in OHLCV_FIELDS
    }
    return ProbeData(
        panel=panel,
        symbols=symbols,
        open_=cast(pd.DataFrame, columns["open"][symbols]),
        high=cast(pd.DataFrame, columns["high"][symbols]),
        low=cast(pd.DataFrame, columns["low"][symbols]),
    )


def alpha_values(data: ProbeData) -> dict[str, pd.DataFrame]:
    """7 条只用 OHLCV、无歧义实现的条目（论文编号保持原样，方便回查）。"""
    close, volume = data.close, data.volume
    open_, high, low = data.open_, data.high, data.low
    mean_high, mean_low = high.rolling(20).mean(), low.rolling(20).mean()
    above = mean_high < high
    inside = (mean_high >= high) & (mean_low > low)
    base = pd.DataFrame(0.0, index=high.index, columns=high.columns)
    return {
        "#101 (close-open)/(high-low)": (close - open_) / (high - low + 0.001),
        "#12 sign(Δvol)·(−Δclose)": sign(volume.diff(1)) * (-close.diff(1)),
        "#23 高位反/低位顺（20 日均）": base.mask(above, -high.diff(2)).mask(
            inside, low.diff(2)
        ),
        "#33 open/close−1": open_ / close - 1.0,
        "#54 −(low−close)·open^5/((low−high)·close^5)": -(low - close)
        * open_**5
        / ((low - high) * close**5),
        "#6 −corr(open, volume, 10)": -correlation(open_, volume, INTERACTION_WINDOWS["#6"]),
        "#44 −corr(high, rank(volume), 5)": -correlation(
            high, cross_rank(volume), INTERACTION_WINDOWS["#44"]
        ),
    }


def gross_turnover(
    values: pd.DataFrame, close: pd.DataFrame, horizon: int, anchor: int
) -> tuple[list[float], list[float]]:
    """网格非重叠取样：返回每期毛价差与换手（权重整体滞后一根）。"""
    first = values.first_valid_index()
    if first is None:
        return [], []
    start = max(int(cast(int, values.index.get_loc(first))), MIN_NAMES) + anchor
    gross: list[float] = []
    turnover: list[float] = []
    previous: pd.Series | None = None
    for step in range(start, len(values) - horizon, horizon):
        row = values.iloc[step - 1]
        valid = row.notna() & close.iloc[step + horizon].notna()
        if int(valid.sum()) < MIN_NAMES:
            continue
        weights = rank_within(row.where(valid)).sub(0.5).mul(2.0)
        weights = weights.div(weights.abs().sum())
        forward = close.iloc[step + horizon] / close.iloc[step] - 1.0
        gross.append(float((weights * forward.where(valid)).sum()))
        delta = 0.0 if previous is None else float((weights - previous).abs().sum())
        turnover.append(delta)
        previous = weights
    return gross, turnover


def read_grid(
    values: pd.DataFrame, close: pd.DataFrame, horizon: int, anchor: int = 0
) -> tuple[float, float, float, float]:
    """一次网格取样 → (毛年化、换手/期、平衡成本 bp、净年化)。

    净年化按单利扣成本（`毛年化 − 年换手 × 成本`）：探针只比较口径量级，不用复利口径。
    """
    gross, turnover = gross_turnover(values, close, horizon, anchor)
    if len(gross) < MIN_NAMES:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean_gross, mean_turn = float(np.mean(gross)), float(np.mean(turnover))
    annual = mean_gross * (BARS_PER_YEAR / horizon)
    cost_drag = mean_turn * (BARS_PER_YEAR / horizon) * COST
    return annual, mean_turn, break_even_cost(mean_gross, mean_turn) * 1e4, annual - cost_drag


def probe_base(alphas: dict[str, pd.DataFrame], data: ProbeData) -> None:
    """探针一：日频全换手下，7 条的 IC 与成本线。"""
    forward = data.close.pct_change(fill_method=None).shift(-1)
    print(f"{'alpha':<40}{'IC':>7}{'IC t':>7}{'毛年化':>9}{'日换手':>8}{'平衡成本':>10}{'净年化':>9}")
    best = ("", 0.0)
    for label, values in alphas.items():
        valid = values.notna() & forward.notna()
        daily_ic = values.where(valid).corrwith(forward, axis=1, method="spearman").dropna()
        annual, turn, be, net = read_grid(values, data.close, horizon=1)
        ic_t = float(daily_ic.mean()) / float(daily_ic.std()) * np.sqrt(len(daily_ic))
        print(
            f"{label:<40}{float(daily_ic.mean()):>7.3f}{ic_t:>7.1f}{annual:>8.1%}"
            f"{turn:>8.2f}{be:>9.1f}bp{net:>8.1%}"
        )
        best = max(best, (label, be), key=lambda item: item[1])
    print(f"\n平衡成本最高的一条：{best[0]} = {best[1]:.1f}bp/边（成本 {COST * 1e4:.0f}bp）")


def probe_horizon(
    alphas: dict[str, pd.DataFrame], data: ProbeData, horizons: tuple[int, ...]
) -> None:
    """探针二：每个 h 穷举全部相位，报中位与范围（单相位读数不作数）。"""
    print(f"{'alpha':<8}{'h':>3}{'相位':>5}{'毛年化(中位)':>13}{'换手':>7}{'平衡中位':>9}{'平衡范围':>16}")
    crossed = 0
    cells = 0
    for label, values in alphas.items():
        for horizon in horizons:
            readings = [
                read_grid(values, data.close, horizon, anchor)
                for anchor in range(horizon)
            ]
            live = [item for item in readings if not np.isnan(item[2])]
            if not live:
                continue
            annuals = [item[0] for item in live]
            turns = [item[1] for item in live]
            breaks = [item[2] for item in live]
            cells += 1
            crossed += int(float(np.median(breaks)) >= COST * 1e4)
            print(
                f"{label.split()[0]:<8}{horizon:>3}{len(live):>5}"
                f"{float(np.median(annuals)):>12.1%}{float(np.median(turns)):>7.2f}"
                f"{float(np.median(breaks)):>8.1f}bp"
                f"{f'{min(breaks):.1f}~{max(breaks):.1f}bp':>16}"
            )
    print(f"\n相位中位平衡成本 ≥ 成本线（{COST * 1e4:.0f}bp）的格子：{crossed}/{cells}")


def probe_phase(alphas: dict[str, pd.DataFrame], data: ProbeData) -> None:
    """探针三：h=30 单看相位散布（同一批数据换锚点，边际本身不应该跳）。"""
    print(f"h={PHASE_HORIZON} ｜ 成本 {COST * 1e4:.1f}bp/边 ｜ 相位 = 0..{PHASE_HORIZON - 1}\n")
    print(f"{'alpha':<8}{'相位':>5}{'平衡最低':>10}{'中位':>10}{'最高':>10}")
    for label, values in alphas.items():
        breaks = [
            read_grid(values, data.close, PHASE_HORIZON, anchor)[2]
            for anchor in range(PHASE_HORIZON)
        ]
        live = [item for item in breaks if not np.isnan(item)]
        if not live:
            continue
        print(
            f"{label.split()[0]:<8}{len(live):>5}{min(live):>9.1f}bp"
            f"{float(np.median(live)):>8.1f}bp{max(live):>8.1f}bp"
        )


def probe_decay(data: ProbeData) -> None:
    """探针四：decay_linear 能否把被成本吃掉的 rev_3 推过成本墙（含毛边际是否还在）。"""
    close = data.close
    reversal = -(close / close.shift(3) - 1.0)  # rev_3：3 根反转，值越大越看多
    variants: dict[str, pd.DataFrame] = {"rev_3 原始": reversal}
    for window in DECAY_WINDOWS:
        variants[f"rev_3 → decay_linear({window})"] = decay_linear(reversal, window)
        variants[f"rev_3 → rank → decay({window})"] = decay_linear(cross_rank(reversal), window)
    print(f"{'口径':<34}{'h=1 毛':>9}{'h=1 平衡':>10}{'h=1 净':>9}{'h=3 毛':>9}{'h=3 平衡':>10}{'h=3 净':>9}")
    for label, values in variants.items():
        cells = ""
        for horizon in (1, 3):
            annual, _, be, net = read_grid(values, close, horizon)
            cells += f"{annual:>8.1%}{be:>9.1f}bp{net:>8.1%}"
        print(f"{label:<34}{cells}")
    print(f"\n过墙线：平衡成本 ≥ {COST * 1e4:.0f}bp/边（等价净 ≥ 0）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha101 可行性探针")
    parser.add_argument("--market", default="swap", help="市场层，默认 swap（探针原始口径）")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument(
        "--probe",
        default="all",
        choices=(*PROBES, "phase", "all"),
        help="跑哪个探针，默认全跑（phase = h=30 的相位散布细节）",
    )
    parser.add_argument(
        "--horizons", default="1,3,7,30", help="horizon 探针的调仓周期（逗号分隔）"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_probe_data(args.market, args.timeframe)
    alphas = alpha_values(data)
    horizons = tuple(int(item) for item in str(args.horizons).split(","))
    print(
        f"Alpha101 探针 | {args.market} {args.timeframe} | 池 {len(data.symbols)} 只（≥{MIN_BARS} 根）"
        f" | 成本 {COST * 1e4:.1f}bp/边（永续 taker + 滑点）\n"
    )
    if args.probe in ("base", "all"):
        print("=== 探针一：base（日频全换手）===")
        probe_base(alphas, data)
    if args.probe in ("horizon", "all"):
        print(f"\n=== 探针二：horizon（h ∈ {horizons}，穷举相位）===")
        probe_horizon(alphas, data, horizons)
    if args.probe == "phase":
        print("\n=== 探针三：phase（h=30 的相位散布）===")
        probe_phase(alphas, data)
    if args.probe in ("decay", "all"):
        print("\n=== 探针四：decay（decay_linear 抢救 rev_3）===")
        probe_decay(data)
    print("\n提醒：探针只回答「这条线值不值得开」；立结论要接 factor_verify.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
