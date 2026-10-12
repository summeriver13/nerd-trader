"""下注规模（de Prado《金融机器学习》第 10 章）。

核心命题：「下不下」与「下多少」是两件事。元标签（`src/research/meta.py`）回答了
前者，这一章回答后者——把概率变成连续的仓位，而不是一个开关。

四条口径（改代码前先读）：

1. 概率→规模用作者 `getSignal` 的原式（Snippet 10.1）：

       z      = (p - 1/numClasses) / sqrt(p * (1 - p))
       signal = side * (2 * Z[z] - 1)

   零假设是**纯运气** `1/numClasses`（二元即 1/2），所以 p = 1/2 恰好给出 0 仓位。
2. **p 低于纯运气时该式返回负值**。标准标注模型遇不到这种情况（`pred` 取 argmax，
   概率必然 ≥ 1/numClasses）；元标签里 p 是「主模型这一注会赢」的概率，低于纯运气
   意味着没有证据支持这个方向，翻成反向下注没有依据——所以 `sized_bet` 夹到 0。
3. 在途注重叠时**取平均而不是求和**（Snippet 10.2 的 `avgActiveSignals`）。我们的信号
   是**日历型**的——每天都开一注（规模可以是 0），所以「此刻在途的注」就是最近
   `horizon` 天，实际仓位即一条 h 日移动平均。关键性质是**不重复计数**：连续两天各开
   一注同向的 1 倍仓，暴露仍是 1 倍而不是 2 倍。这正是第 4 章唯一性权重在同一个问题
   上的另一种说法。
4. 规模要量化到网格再执行（Snippet 10.3 的 `discreteSignal`）：连续权重会为 0.001
   的变化反复调仓，而本仓库的实测结论是成本吃掉一切。

无前视的口径：`W[t]` 只由 t 及更早的期望仓位构造，收益取 `r[t→t+1]`，所以
`pnl[t] = W[t] * r[t→t+1]`——改 `r[t+1]` 不会改 `pnl[t]`（有用例钉住）。

成本口径（**别换成 `portfolio.break_even_cost`**）：这里的 `turnover` 是 `Σ|ΔW|`，
即直接可交易的名义额，每一单位只付一次 `cost_one_way`；而 `portfolio.py` 的 `turnover`
是「单腿换名比例」再乘 4 笔调仓交易（2 腿 × 进/出）。两者定义不同，混用会把平衡成本
错算 4 倍——这个不一致是写本模块的用例时被一次失败的回推发现的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.factors.portfolio import DEFAULT_ONE_WAY_COST

CHANCE_PROBABILITY: float = 0.5
"""二元情形的「纯运气」基准（作者 `getSignal` 的 `1/numClasses`）。"""

PROBABILITY_EPSILON: float = 1e-12
"""概率夹取边界。

标准误 `sqrt(p(1-p))` 在 p→0/1 时趋于 0，直接相除得到 inf 并触发 RuntimeWarning
——本仓库 `filterwarnings = ["error"]`，warning 会直接炸用例。夹到 eps 后 z 趋于
±5e5，`Z[z]` 就是 0 或 1，那正是正确答案的极限。
"""

PERIODS_PER_YEAR: float = 365.0
"""加密全年无休，日频年化用 365。"""


def bet_size_from_probability(
    probability: pd.Series, chance: float = CHANCE_PROBABILITY
) -> pd.Series:
    """概率 → 下注规模。

    作者 `getSignal` 的原式。三条性质（测试钉住）：p = `chance` 时恰为 0；对 p 单调
    递增；`m(1-p) = -m(p)`（做多做空对称）。**返回值带符号**——p < `chance` 时是负的，
    见模块文档第 2 条。
    """
    values = np.clip(
        probability.to_numpy(dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    standard_error = np.sqrt(values * (1.0 - values))
    z = (values - chance) / standard_error
    size = 2.0 * norm.cdf(z) - 1.0
    return pd.Series(size, index=probability.index, name=probability.name)


def sized_bet(
    probability: pd.Series, side: pd.Series, chance: float = CHANCE_PROBABILITY
) -> pd.Series:
    """元标签的落地口径：概率低于纯运气就**不下注**，而不是反向下注（模块文档第 2 条）。"""
    size = bet_size_from_probability(probability, chance).clip(lower=0.0)
    return side.reindex(size.index) * size


def average_active_bets(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """在途注重叠时取平均（Snippet 10.2）。

    `weights[i, t]` 是 t 日对 i 的期望仓位（含符号）。因为是日历型信号，「此刻在途的
    注」就是最近 `horizon` 天，所以实际仓位是它们的均值。前 `horizon - 1` 天留空——那是
    warmup，不是小仓位。

    不变量：期望仓位恒定时，实际仓位等于该常数（连续加注不会把暴露翻倍）。
    """
    if horizon < 1:
        raise ValueError("持有期必须 ≥ 1")
    return cast(pd.DataFrame, weights.rolling(horizon, min_periods=horizon).mean())


def quantize_bet_size(weights: pd.DataFrame, step: float) -> pd.DataFrame:
    """把规模量化到网格（Snippet 10.3 的 `discreteSignal`）。

    连续权重会为 0.001 的变化反复调仓；量化后只在大到跨格时才动仓，换手因此下降。
    **前提是抖动相对网格小**——信号本身大幅跳变时四舍五入可上可下，换手不一定降
    （试过把「量化必然降换手」写成不变量，被实测推翻）。`step <= 0` 视为不量化。
    """
    if step <= 0.0:
        return weights
    return cast(pd.DataFrame, (weights / step).round() * step)


@dataclass(frozen=True)
class PositionOptions:
    """持仓与成本口径。"""

    horizon: int = 3
    """一期注持有的天数（通常等于标签的垂直障碍）。"""

    cost_one_way: float = DEFAULT_ONE_WAY_COST
    """单边成本（比例），默认取仓库永续 taker 净费率。"""

    periods_per_year: float = PERIODS_PER_YEAR
    """年化因子。"""


@dataclass(frozen=True)
class BetStats:
    """一条仓位路径的读数。序列留着，是为了能查「改动未来会不会改过去」。"""

    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    exposure: float
    """平均总暴露 `Σ|W|`——它是杠杆，跨方案比较时必须一起看。"""
    gross: float
    """每期毛收益均值。"""
    net: float
    """每期净收益均值。"""
    gross_sharpe: float
    net_sharpe: float
    breakeven_cost: float
    """净收益归零的单边成本。"""

    @property
    def periods(self) -> int:
        return int(self.gross_returns.size)


@dataclass(frozen=True)
class SchemeOptions:
    """方案对比的口径。"""

    horizon: int = 3
    """一期注持有的天数。"""

    cost_one_way: float = DEFAULT_ONE_WAY_COST
    """单边成本。"""

    periods_per_year: float = PERIODS_PER_YEAR
    """年化因子。"""

    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7)
    """要试的开关阈值。"""

    grid: float = 0.1
    """定仓方案的量化网格。"""

    def positions(self) -> PositionOptions:
        return PositionOptions(self.horizon, self.cost_one_way, self.periods_per_year)


def _size_columns(probability: pd.DataFrame, side: pd.DataFrame) -> pd.DataFrame:
    """逐列把概率变成带符号的下注规模。

    `sized_bet` 的正则在 Series 上，逐列调用比 stack/unstack 少一层索引陷坑；
    列数是百量级，代价可以忽略。
    """
    return pd.DataFrame(
        {
            symbol: sized_bet(
                cast(pd.Series, probability[symbol]), cast(pd.Series, side[symbol])
            )
            for symbol in probability.columns
        }
    )


def compare_schemes(
    side: pd.DataFrame,
    returns: pd.DataFrame,
    probability: pd.DataFrame,
    *,
    options: SchemeOptions,
) -> pd.DataFrame:
    """在同一批观测上比较「开关」与「定仓」——这就是第 10 章要回答的问题。

    所有方案共用同一份收益与同一持有期，所以差异只来自下注方案本身。夏普对缩放不变，
    因此不需要对齐杠杆；但 `exposure` 与 `turnover` 必须陪着一起看——仓位小了成本也小，
    单看净收益会把「不敢下注」误读成「下得准」。

    缺失的期望仓位补 0（没有信号就是不想持仓，不能让滚动均值凭空续上一个仓位）。

    `returns` 是**每期一段**的收益（日频即 1 日）：持有 h 期已经由持仓路径表达，
    传 h 期收益会让 P&L 放大 h 倍，并让相邻期共享 h-1 期内容——夏普被抬起来，
    而且各方案抬得不一样（踩过：试跑时用 h 日前向收益，`全下` 的毛收益报到 927bp/期）。
    """
    positions = options.positions()
    sized = _size_columns(probability, side)
    schemes: list[tuple[str, pd.DataFrame]] = [("全下（等权）", side)]
    schemes += [
        (
            f"开关 @{threshold:.2f}",
            cast(pd.DataFrame, side.where(probability >= threshold, 0.0)),
        )
        for threshold in options.thresholds
    ]
    schemes.append(("概率定仓", sized))
    schemes.append(
        (f"概率定仓 + 网格 {options.grid:g}", quantize_bet_size(sized, options.grid))
    )

    rows: list[dict[str, object]] = []
    for name, weights in schemes:
        stats = simulate_positions(weights.fillna(0.0), returns, positions)
        rows.append(
            {
                "scheme": name,
                "gross_bp": stats.gross * 1e4,
                "net_bp": stats.net * 1e4,
                "turnover": float(stats.turnover.mean()),
                "exposure": stats.exposure,
                "gross_sharpe": stats.gross_sharpe,
                "net_sharpe": stats.net_sharpe,
                "breakeven_bp": stats.breakeven_cost * 1e4,
                # 除以平均暴露才读得懂：各方案杠杆差几十倍，原值不可直接比
                "gross_per_unit_bp": stats.gross / stats.exposure * 1e4
                if stats.exposure > 0
                else 0.0,
                "net_per_unit_bp": stats.net / stats.exposure * 1e4
                if stats.exposure > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _breakeven_cost(gross: float, turnover: float) -> float:
    """把净收益吃到零的单边成本（口径见模块文档）。

    毛收益为负时返回 0：无成本也不赚，谈不上平衡点。
    """
    if turnover <= 0.0 or gross <= 0.0:
        return 0.0
    return gross / turnover


def _annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.size < 2:
        return 0.0
    deviation = float(returns.std())
    if deviation == 0.0:
        return 0.0
    return float(returns.mean()) / deviation * float(np.sqrt(periods_per_year))


def simulate_positions(
    weights: pd.DataFrame, returns: pd.DataFrame, options: PositionOptions
) -> BetStats:
    """期望仓位路径 → 每日 P&L、换手与成本。

    `weights[i, t]` 是 t 日收盘决定的期望仓位，`returns[i, t]` 是 t→t+1 的收益。
    收益乘的是**当天**的仓位（`W[t] * r[t→t+1]`），不是下一天的——用下一天的仓位就是
    未来函数。**传 1 期收益，不是持有 h 期的收益**（见 `compare_schemes`）.
    """
    if weights.empty:
        raise ValueError("仓位路径为空")
    aligned = returns.reindex(index=weights.index, columns=weights.columns)
    active = average_active_bets(weights, options.horizon)
    gross = cast(pd.Series, (active * aligned).sum(axis=1, min_count=1))
    # 换手是「今天相对昨天的变化」；第一天没有昨天，不计成本
    turnover = cast(pd.Series, active.diff().abs().sum(axis=1, min_count=1)).fillna(0.0)
    net = gross - turnover * options.cost_one_way

    # pandas-stub 把布尔掩码索引推成联合类型，按仓库惯例用 cast 收窄
    usable = cast(pd.Series, gross.notna())
    used_gross = cast(pd.Series, gross[usable])
    used_net = cast(pd.Series, net[usable])
    used_turnover = cast(pd.Series, turnover[usable])
    gross_exposure = cast(pd.Series, active.abs().sum(axis=1))
    if used_gross.empty:
        raise ValueError("仓位与收益没有可用的交集")
    mean_gross = float(used_gross.mean())
    mean_turnover = float(used_turnover.mean())
    return BetStats(
        gross_returns=used_gross,
        net_returns=used_net,
        turnover=used_turnover,
        exposure=float(gross_exposure[usable].mean()),
        gross=mean_gross,
        net=float(used_net.mean()),
        gross_sharpe=_annualized_sharpe(used_gross, options.periods_per_year),
        net_sharpe=_annualized_sharpe(used_net, options.periods_per_year),
        breakeven_cost=_breakeven_cost(mean_gross, mean_turnover),
    )
