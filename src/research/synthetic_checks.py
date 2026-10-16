"""
src/research/synthetic_checks.py
合成床的读数层与真值判据：IC 层、钱层、家族级价差矩阵，以及「判对没有」的对照。

从 `synthetic.py` 拆出来的是职责：那边是**真值从哪来**（DGP 旋钮与 `synthetic_panel`），
这边是**把读数摆到真值旁边**。两边必须共享同一份口径常量与同一份 `PortfolioOptions`——
IC 层与钱层的网格起点（anchor）错开会让价差矩阵整行被丢（见 `synthetic.py` 里 BED_ANCHOR
的注释），所以口径常量仍留在 `synthetic.py`，本模块只 import。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.factors.evaluation import ic_stats
from src.research.factors.library import FACTOR_LIBRARY, MAIN_POOL_KEYS
from src.research.factors.panel import Panel, tradable_mask
from src.research.factors.portfolio import (
    DEFAULT_ONE_WAY_COST,
    PortfolioOptions,
    break_even_cost,
    portfolio_stats,
)
from src.research.synthetic import (
    BED_ANCHOR,
    BED_DAYS,
    BED_HORIZON,
    BED_QUANTILES,
    BED_SYMBOLS,
    BP,
    IC_MIRROR,
    IC_SILENT,
    IC_VISIBLE,
    SyntheticSpec,
    synthetic_panel,
)


def _portfolio_options(spec: SyntheticSpec, horizon: int) -> PortfolioOptions:
    """同一份口径给 IC 层与钱层共用：只差成本，锚点与分位数必须一致。"""
    return PortfolioOptions(
        horizon=horizon,
        quantiles=BED_QUANTILES,
        anchor=BED_ANCHOR,
        cost_one_way=(
            DEFAULT_ONE_WAY_COST if spec.cost_one_way is None else spec.cost_one_way
        ),
    )


def ic_readings(
    panel: Panel,
    mask: pd.DataFrame,
    horizon: int,
    keys: tuple[str, ...] = MAIN_POOL_KEYS,
) -> pd.DataFrame:
    """逐因子的定向后 IC（``FactorSpec.values`` 已按方向调整，正 = 高分位跑赢）。"""
    rows: list[dict[str, object]] = []
    for key in keys:
        stats = ic_stats(FACTOR_LIBRARY[key], panel, mask, horizon)
        rows.append(
            {
                "因子": key,
                "IC": stats.mean,
                "t": stats.t_stat,
                "期数": float(stats.periods),
            }
        )
    return pd.DataFrame(rows)


def money_readings(
    panel: Panel,
    mask: pd.DataFrame,
    horizon: int,
    spec: SyntheticSpec,
    keys: tuple[str, ...] = MAIN_POOL_KEYS,
) -> pd.DataFrame:
    """逐因子的分位组合读数（定向后，bp/期）：毛、净、换手、盈亏平衡成本。

    成本口径由 ``spec.cost_one_way`` 决定，IC 层与钱层必须用同一份 ``_portfolio_options``，
    否则两层的网格起点会错开。
    """
    rows: list[dict[str, object]] = []
    options = _portfolio_options(spec, horizon)
    for key in keys:
        stats = portfolio_stats(FACTOR_LIBRARY[key], panel, mask, options)
        rows.append(
            {
                "因子": key,
                "期数": float(stats.periods),
                "毛(bp)": stats.gross_spread * BP,
                "净(bp)": stats.net_spread * BP,
                "换手": stats.turnover,
                "平衡成本(bp)": break_even_cost(stats.gross_spread, stats.turnover)
                * BP,
            }
        )
    return pd.DataFrame(rows)


def spread_matrix(
    panel: Panel,
    mask: pd.DataFrame,
    horizon: int,
    spec: SyntheticSpec,
    keys: tuple[str, ...] = MAIN_POOL_KEYS,
) -> pd.DataFrame:
    """家族级检验的输入：同一网格上的逐期价差（同 anchor 才不会整行被丢）。"""
    frames: list[pd.Series] = []
    options = _portfolio_options(spec, horizon)
    for key in keys:
        stats = portfolio_stats(FACTOR_LIBRARY[key], panel, mask, options)
        if stats.periods:
            frames.append(stats.spread_returns.rename(key))
    return pd.concat(frames, axis=1)


@dataclass(frozen=True)
class TruthCheck:
    """一条真值对照：检查项、读数、真值、是否命中。"""

    layer: str
    subject: str
    reading: float
    truth: str
    hit: bool


@dataclass(frozen=True)
class BedReadings:
    """一档床、一颗种子上跑完整条前两层的全部产物。"""

    spec: SyntheticSpec
    seed: int
    horizon: int
    panel: Panel
    mask: pd.DataFrame
    ic: pd.DataFrame
    money: pd.DataFrame
    spreads: pd.DataFrame

    def checks(self) -> list[TruthCheck]:
        return truth_checks(self.spec, self.ic, self.money)


def run_bed(
    spec: SyntheticSpec,
    seed: int = 0,
    horizon: int = BED_HORIZON,
    *,
    keys: tuple[str, ...] = MAIN_POOL_KEYS,
    symbols: int = BED_SYMBOLS,
    days: int = BED_DAYS,
) -> BedReadings:
    """造面板 + 跑 IC 层与钱层（不含校正检验，那是调用方的事）。"""
    panel = synthetic_panel(spec, seed, symbols=symbols, days=days)
    mask = tradable_mask(panel)
    return BedReadings(
        spec=spec,
        seed=seed,
        horizon=horizon,
        panel=panel,
        mask=mask,
        ic=ic_readings(panel, mask, horizon, keys),
        money=money_readings(panel, mask, horizon, spec, keys),
        spreads=spread_matrix(panel, mask, horizon, spec, keys),
    )


def truth_checks(
    spec: SyntheticSpec, ic: pd.DataFrame, money: pd.DataFrame
) -> list[TruthCheck]:
    """把逐因子读数对齐真值：该亮的亮、镜像的反、其余沉默、钱的结论一致。"""
    reads = {str(row["因子"]): float(str(row["IC"])) for _, row in ic.iterrows()}
    nets = {str(row["因子"]): float(str(row["净(bp)"])) for _, row in money.iterrows()}
    checks: list[TruthCheck] = []
    for key, value in reads.items():
        if key in spec.expected:
            checks.append(
                TruthCheck(
                    "该亮", key, value, f"IC ≥ {IC_VISIBLE}", value >= IC_VISIBLE
                )
            )
        elif key in spec.mirror:
            checks.append(
                TruthCheck(
                    "镜像", key, value, f"IC ≤ −{IC_MIRROR}", value <= -IC_MIRROR
                )
            )
        else:
            checks.append(
                TruthCheck(
                    "应沉默", key, value, f"|IC| ≤ {IC_SILENT}", abs(value) <= IC_SILENT
                )
            )
    # 只判「种下的因子」里真出现在读数表里的那些：子集读数是常态（用例只挑要断言的因子），
    # 其余因子的净收益是「样本均值运气」，那是一档的零分布，不是结论
    claimed = [key for key in spec.expected if key in nets]
    if spec.profitable is True:
        for key in claimed:
            value = nets[key]
            checks.append(TruthCheck("钱", key, value, "净 > 0", value > 0.0))
    elif spec.profitable is False:
        for key in claimed:
            value = nets[key]
            checks.append(TruthCheck("钱", key, value, "净 ≤ 0", value <= 0.0))
    return checks
