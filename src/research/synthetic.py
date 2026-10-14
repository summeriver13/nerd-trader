"""真值已知的合成面板：把整条研究管道放到可控 DGP 上对答案。

`scripts/corrections_study.py` 的 ``simulate`` 造的是**候选收益序列**（共同因子 + 个体噪声），
只够量校正检验的尺寸与功效；要问「整条管道在真值已知时判对没有」，输入必须是**面板**
（close/volume/notional）——因子构造、分位组合、成本、平台判据才会真的被执行。真值是 DGP 的
旋钮（种了哪个因子、方向哪边、成本多高），不是事后从数据里挑出来的结论。

三档床各自对应本仓一段真实经历（读数见 ``scripts/synthetic_study.py``）：

- ``noise``：无信号。任何因子都不该亮——它是**其它三档的零分布**。
- ``reversal``：种反转（个股收益的 AR(1) 系数为负）。``rev_3`` 该亮、``mom_*`` 该反着亮，
  默认成本下净为正。
- ``reversal_costly``：**同一份价格**、成本抬到 30bp/边。IC 照旧该亮、净该翻负——第六节那堵
  成本墙的合成版本。

**为什么没有「低波动」档**：低波动这类温和的持续性结构在日频合成床上读不出 IC。真实世界的低
波动溢价约 1bp/期，放大 30 倍后 ``vol_30`` 的 IC 仍只有 ~0.01（日频截面噪声有 3~5%），与噪声床
同量级，做不成「该亮」的判据。要在合成床上复现第十五节的结论，得换更低频的收益口径。

一条必须记住的口径：单标的 T 期的样本均值本身就带运气（T=520 时 SE ≈ 27bp/期），任何与
「这段行情的均值」相关的排序都会读出几十 bp 的价差。所以**钱的读数只在多颗种子上比**，
单颗种子的净收益不能当结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
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

BED_SYMBOLS: Final[int] = 60
BED_DAYS: Final[int] = 2000
BED_QUANTILES: Final[int] = 5
BED_HORIZON: Final[int] = 1
# 家族级检验必须钉同一相位（网格起点）：各因子 warmup 不同，相位错开会拼出「每行只有一个
# 非 NaN」的矩阵，被对齐逻辑整行丢掉（README 第二十四节第 4 小节）。起点取上市门槛的天数，
# 与可交易掩码的起点同量级。
BED_ANCHOR: Final[int] = 180
# 共同因子占比：截面排序与它无关（多空价差里自己抵消），它只保证面板不像 N 条独立随机游走
BED_MARKET_SHARE: Final[float] = 0.5
PRICE_START: Final[float] = 100.0
# 标的间名义额跨 3 倍，最低一档也高于池子门槛（$1M/日中位），liq_30 才有截面差异
NOTIONAL_LOW: Final[float] = 5e6
NOTIONAL_HIGH: Final[float] = 1.5e7
VOL_LOW: Final[float] = 0.02
VOL_HIGH: Final[float] = 0.08
BP: Final[float] = 1e4
# 真值判据：IC 到这个量级才算「亮」，低于它算沉默。两个阈值都要离 0 有距离，
# 否则单颗种子的噪声就能把结论翻过来
IC_VISIBLE: Final[float] = 0.02
# 镜像档（7 根/30 根动量）与种下的 1 根 AR(1) 不同阶，强度天然弱一截，门槛另设
IC_MIRROR: Final[float] = 0.005
IC_SILENT: Final[float] = 0.02
# 弱信号档的成本口径：taker 费 + 滑点 + 池尾点差的量级，比默认口径高一个数量级
COSTLY_ONE_WAY: Final[float] = 0.0030
REVERSAL_STRENGTH: Final[float] = -0.06


@dataclass(frozen=True)
class SyntheticSpec:
    """一档合成床：DGP 旋钮 + 该档的真值。"""

    key: str
    label: str
    # 个股收益的 AR(1) 系数。负值 = 反转（下一根收益反向于上一根），正值 = 动量
    reversal: float
    # 低波动溢价：每根漂移 μ_i = −vol_premium × (σ_i − σ̄)，把未来收益按 σ 反向拉开
    vol_premium: float = 0.0
    # 单边成本口径；None = 用仓库默认（永续 taker 净成本）
    cost_one_way: float | None = None
    # 该亮的因子键（定向后 IC 为正）
    expected: tuple[str, ...] = ()
    # 同一现象的镜像（定向后 IC 为负）；剩下的因子都该沉默
    mirror: tuple[str, ...] = ()
    # 真值层面的钱：True 该净为正、False 该净非正、None 不判（换手过小、平衡成本没有意义）
    profitable: bool | None = None
    note: str = ""


BEDS: tuple[SyntheticSpec, ...] = (
    SyntheticSpec(
        key="noise",
        label="无信号（DGP 里没有任何可预测结构）",
        reversal=0.0,
        note="零分布：七个主池因子都该沉默",
    ),
    SyntheticSpec(
        key="reversal",
        label="种反转（个股 AR(1) 为负）",
        reversal=REVERSAL_STRENGTH,
        # ffd 的方向是「偏离慢趋势越多越看空」，与反转同族，种了反转它也该亮
        expected=("rev_3", "ffd_d0p4"),
        mirror=("mom_7", "mom_30"),
        profitable=True,
        note="rev_3 与 ffd 该亮、动量为负，默认成本下净为正",
    ),
    SyntheticSpec(
        key="reversal_costly",
        label="种反转 + 30bp/边成本（同一份价格）",
        reversal=REVERSAL_STRENGTH,
        cost_one_way=COSTLY_ONE_WAY,
        expected=("rev_3", "ffd_d0p4"),
        mirror=("mom_7", "mom_30"),
        profitable=False,
        note="IC 与上档相同，钱的读数被成本转负",
    ),
    SyntheticSpec(
        key="lowvol_probe",
        label="低波动探针（只读 IC 层，不设真值判据）",
        reversal=0.0,
        # 真实低波动溢价约 1bp/期（λ≈4e-4），这里放大 30 倍；即便这样 IC 也只有 ~0.01，
        # 与噪声床同量级——这一档的作用是把「日频上读不出判据」这件事量出来
        vol_premium=1.2e-2,
        note="探针：低波动溢价被日频截面噪声淹没，IC 层不设真值判据",
    ),
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


def synthetic_panel(
    spec: SyntheticSpec,
    seed: int = 0,
    *,
    symbols: int = BED_SYMBOLS,
    days: int = BED_DAYS,
) -> Panel:
    """按 DGP 造面板：每根收益 = 个股 AR(1) + 漂移 + σ_i 尺度的共同因子与个体噪声。

    平稳标准差恰为 σ_i（AR(1) 的尺度按 sqrt(1−ρ²) 折回去，反转只改时序不改波动），
    所以 ``vol_30`` 排序的就是造数据时给的 σ_i。名义额静止（不随时间变）但跨标的拉开，
    于是 ``liq_30`` 有截面差异、可交易掩码又恒为真。
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=days, freq="1D", tz="UTC")
    names = pd.Index([f"S{position:03d}-USDT" for position in range(symbols)])
    # σ 与名义额必须**独立**排布：同序会让 liq_30 变成 vol_30 的替身（标定时真踩过，
    # 两个因子的 IC 完全同号同量级），那时噪声床的「沉默」判据就全是错的
    sigma = rng.permutation(np.geomspace(VOL_LOW, VOL_HIGH, symbols))
    drift = -spec.vol_premium * (sigma - sigma.mean())
    market = rng.standard_normal(days)
    shock = rng.standard_normal((days, symbols))
    correlated = (
        BED_MARKET_SHARE * market[:, None] + np.sqrt(1.0 - BED_MARKET_SHARE**2) * shock
    )
    scale = float(np.sqrt(1.0 - spec.reversal**2))
    returns = np.empty((days, symbols))
    returns[0] = drift + sigma * scale * correlated[0]
    for step in range(1, days):
        returns[step] = (
            spec.reversal * returns[step - 1] + drift + sigma * scale * correlated[step]
        )
    close = pd.DataFrame(
        PRICE_START * np.cumprod(1.0 + returns, axis=0), index=index, columns=names
    )
    level = rng.permutation(np.geomspace(NOTIONAL_LOW, NOTIONAL_HIGH, symbols))
    notional = pd.DataFrame(
        np.repeat(level[None, :], days, axis=0), index=index, columns=names
    )
    return Panel(
        timeframe="1d", close=close, volume=notional / close, notional=notional
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
