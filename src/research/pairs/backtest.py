"""
src/research/pairs/backtest.py
交易期回测：两腿对冲、成本与资金费都按可执行口径计，逐日贡献留痕。

完整口径见 `docs/pairs/README.md`（单一出处）。这里只记三条不写在别处就不明显的：

1. **每对占 1 单位资金**：多头腿名义 ``1/(1+β)``、空头腿 ``β/(1+β)``，两腿合计恒为
   1 单位（gross ≤ 1，不加杠杆）；所以「多空各半」只在 β≈1 时成立。
2. **逐日贡献留痕**（毛收益/换手/资金费/暴露）：改成本口径、拆归因都不必重跑回测。
3. **参数冻结在形成期末**：α、β、价差均值与标准差都不在交易期重估——滚动重估会
   让「价差」跟着价格漂移，回测里表现为几乎不交易的漂亮结果，实盘那是追价。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.panel import Panel, UniverseRule, tradable_mask
from src.research.pairs.funding import (
    funding_cost,
    funding_coverage,
    funding_symbols,
)
from src.research.pairs.selection import (
    Pair,
    SelectionOptions,
    random_pairs,
    select_pairs,
)

# 一段交易期至少要有这么多根才计入：太短的窗口里「回归」与「噪声」分不开
MIN_TRADING_BARS: int = 10
# 逐日贡献表的列：改成本口径、拆归因都靠它
CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "pair",
    "gross",
    "turnover",
    "funding",
    "exposure",
)
# 每对明细表的列（固定列序，空运行也要给出表头）
TRADE_COLUMNS: tuple[str, ...] = (
    "start",
    "end",
    "left",
    "right",
    "beta",
    "net_exposure",
    "corr",
    "adf_p",
    "half_life",
    "rebalances",
    "bars_held",
    "stopped",
    "gross_sum",
    "cost_sum",
    "funding_sum",
    "net_total",
)


@dataclass(frozen=True)
class PairsOptions:
    """配对研究口径：选对（selection）+ 交易（z 阈值、窗口、成本）。"""

    selection: SelectionOptions = SelectionOptions()
    trading_bars: int = 60
    step_bars: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    cost_one_way: float = PERP_FEES.taker_net()
    # 非 None 时把「协整选对」换成同池随机选对（证伪用的安慰剂组），种子决定随机流
    random_seed: int | None = None

    def describe(self) -> str:
        return (
            f"{self.selection.describe()} | 交易期 {self.trading_bars} 根 "
            f"步长 {self.step_bars} | 开 {self.entry_z:g}σ / 平 {self.exit_z:g}σ / "
            f"止损 {self.stop_z:g}σ | 成本 {self.cost_one_way * 1e4:.1f}bp/边"
        )


@dataclass(frozen=True)
class PairsRun:
    """一次完整运行的产物：逐日贡献 + 每对明细 + 组合口径的派生量。"""

    contributions: pd.DataFrame
    trades: pd.DataFrame
    windows: int
    pairs_per_window: float
    cost_one_way: float
    # 资金费日覆盖（只统计有费率表的标的）；funding_symbols 是标的层面的覆盖
    funding_coverage: float
    funding_symbols: int = field(default=0)
    symbols: int = field(default=0)

    @property
    def daily(self) -> pd.Series:
        """按运行时的成本口径算出的组合逐日净收益。"""
        return self.at_cost(self.cost_one_way)

    @property
    def exposure(self) -> pd.Series:
        """逐日平均 gross 暴露（占组合资金的比例）。"""
        if self.contributions.empty:
            return pd.Series(dtype=float)
        return cast(
            pd.Series,
            self.contributions["exposure"].groupby(level=0).mean().sort_index(),
        )

    @property
    def rebalances(self) -> int:
        """两腿合计的成交次数（每次仓位变动 = 两腿各一笔）。"""
        if self.trades.empty:
            return 0
        return int(self.trades["rebalances"].sum())

    def at_cost(self, cost_one_way: float) -> pd.Series:
        """换一个单边成本口径重算逐日净收益（换手与资金费已留痕，无需重跑）。"""
        return daily_at_cost(self.contributions, cost_one_way)


def daily_at_cost(contributions: pd.DataFrame, cost_one_way: float) -> pd.Series:
    """逐日组合净收益 = 当期选中对子的等权平均（毛 − 成本 − 资金费）。"""
    if contributions.empty:
        return pd.Series(dtype=float)
    net = (
        contributions["gross"]
        - cost_one_way * contributions["turnover"]
        - contributions["funding"]
    )
    grouped = net.groupby(level=0)
    return cast(pd.Series, (grouped.sum() / grouped.size()).sort_index())


def run_pairs(
    panel: Panel,
    options: PairsOptions | None = None,
    funding: pd.DataFrame | None = None,
    rule: UniverseRule | None = None,
) -> PairsRun:
    """滚动「形成期选对 → 交易期回测」，返回等权组合的逐日贡献与明细。

    ``funding`` 为按日求和的费率宽表（正 = 多头付钱）。传 None 表示不计资金费，
    报告必须显式说明——现货研究里没有这一项，永续研究里它是真实的现金项。
    """
    options = options or PairsOptions()
    log_close = cast(pd.DataFrame, np.log(panel.close))
    returns = panel.close.pct_change(fill_method=None)
    mask = tradable_mask(panel, rule)
    stamps = panel.index

    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    windows = 0

    for t0 in _window_starts(len(stamps), options):
        t1 = min(t0 + options.trading_bars, len(stamps))
        if t1 - t0 < MIN_TRADING_BARS:
            continue
        pairs = _usable_pairs(panel, log_close, mask, t0, t1, options)
        if not pairs:
            continue
        windows += 1
        for pair in pairs:
            frame, record = _trade_pair(
                pair, log_close, returns, t0, t1, options, funding
            )
            frame["pair"] = f"{pair.label}@{stamps[t0]:%Y-%m-%d}"
            frames.append(frame)
            records.append(record)

    return PairsRun(
        contributions=(
            pd.concat(frames)
            if frames
            else pd.DataFrame(columns=pd.Index(CONTRIBUTION_COLUMNS))
        ),
        trades=pd.DataFrame(records, columns=pd.Index(TRADE_COLUMNS)),
        windows=windows,
        pairs_per_window=len(records) / windows if windows else 0.0,
        cost_one_way=options.cost_one_way,
        funding_coverage=funding_coverage(panel, funding),
        funding_symbols=funding_symbols(panel, funding),
        symbols=len(panel.symbols),
    )


def _window_starts(bars: int, options: PairsOptions) -> list[int]:
    """交易期起点：第一个形成期结束后每隔 ``step_bars`` 取一个。"""
    first = options.selection.formation_bars
    return list(range(first, bars - MIN_TRADING_BARS, max(options.step_bars, 1)))


def _usable_pairs(
    panel: Panel,
    log_close: pd.DataFrame,
    mask: pd.DataFrame,
    t0: int,
    t1: int,
    options: PairsOptions,
) -> list[Pair]:
    """选对并剔除交易期有缺口的对子（缺口日会伪造 0 收益并漏掉真实波动）。"""
    if options.random_seed is None:
        picked = select_pairs(log_close, mask, t0, options.selection)
    else:
        picked = random_pairs(
            log_close, mask, t0, options.selection, options.random_seed
        )
    out: list[Pair] = []
    for pair in picked:
        window = panel.close.iloc[t0:t1][[pair.left, pair.right]]
        prices = window.to_numpy()
        if bool(np.isfinite(prices).all()) and not bool((prices <= 0).any()):
            out.append(pair)
    return out


def _trade_pair(
    pair: Pair,
    log_close: pd.DataFrame,
    returns: pd.DataFrame,
    t0: int,
    t1: int,
    options: PairsOptions,
    funding: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """交易一对：返回逐日贡献表与明细记录。

    贡献表按「1 单位资金」记账：``gross`` 是毛收益、``turnover`` 是当根换手名义、
    ``funding`` 是当根资金费支出、``exposure`` 是当根 gross 暴露。
    """
    window = slice(t0, t1)
    index = log_close.index[window]
    left_now = np.asarray(log_close[pair.left].iloc[window], dtype=float)
    right_now = np.asarray(log_close[pair.right].iloc[window], dtype=float)
    spread = pd.Series(left_now - (pair.alpha + pair.beta * right_now), index=index)
    pos, stopped = _positions((spread - pair.spread_mean) / pair.spread_std, options)
    executed = cast(pd.Series, pos.shift(1).fillna(0.0))

    weight = 1.0 / (1.0 + pair.beta)
    ret_left = returns[pair.left].iloc[window].to_numpy(dtype=float)
    ret_right = returns[pair.right].iloc[window].to_numpy(dtype=float)
    held = executed.to_numpy(dtype=float)
    gross = weight * held * (ret_left - pair.beta * ret_right)
    turnover = np.abs(np.diff(held, prepend=0.0))
    funding_paid = funding_cost(pair, executed, funding, weight)

    frame = pd.DataFrame(
        {
            "gross": gross,
            "turnover": turnover,
            "funding": funding_paid,
            "exposure": np.abs(held),
        },
        index=index,
    )
    net = gross - options.cost_one_way * turnover - funding_paid
    record: dict[str, object] = {
        "start": index[0],
        "end": index[-1],
        "left": pair.left,
        "right": pair.right,
        "beta": pair.beta,
        "net_exposure": (1.0 - pair.beta) / (1.0 + pair.beta),
        "corr": pair.corr,
        "adf_p": pair.adf_p,
        "half_life": pair.half_life,
        "rebalances": int(np.count_nonzero(np.diff(held, prepend=0.0))),
        "bars_held": int(np.count_nonzero(held)),
        "stopped": stopped,
        "gross_sum": float(gross.sum()),
        "cost_sum": float(turnover.sum() * options.cost_one_way),
        "funding_sum": float(funding_paid.sum()),
        "net_total": float(np.prod(1.0 + net) - 1.0),
    }
    return frame, record


def _positions(z: pd.Series, options: PairsOptions) -> tuple[pd.Series, bool]:
    """由 z 走出的目标仓位（+1 多价差 / −1 空价差 / 0 空仓）与「是否触发过止损」。

    非有限的 z（数据缺口）维持原仓：既不开新仓，也不假装平价离场。
    """
    values = z.to_numpy(dtype=float)
    pos = np.zeros(values.size)
    state = 0.0
    stopped = False
    for i, value in enumerate(values):
        if np.isfinite(value):
            if state == 0.0:
                if not stopped and value <= -options.entry_z:
                    state = 1.0
                elif not stopped and value >= options.entry_z:
                    state = -1.0
            elif state == 1.0 and (
                value >= -options.exit_z or value <= -options.stop_z
            ):
                stopped = stopped or value <= -options.stop_z
                state = 0.0
            elif state == -1.0 and (value <= options.exit_z or value >= options.stop_z):
                stopped = stopped or value >= options.stop_z
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=z.index), bool(stopped)
