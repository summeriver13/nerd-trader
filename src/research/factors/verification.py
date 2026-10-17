"""平台赢家的收益层验证：把扫描出的窗口送进分位组合，再跑过拟合检验。

扫描器（`sweep.py`）只做信号层 IC，便宜；这里做两件贵的事：把候选跑成分位组合的
逐期多空价差序列，再交给 `overfitting.py` 的 PBO（CSCV）。

PBO 问的是：在样本内挑出的冠军，到样本外还排得上号吗？排不上号的比例就是过拟合
概率。它不需要知道「有多少个真效应」，因此不受 de Prado & Fabozzi (2026) 那条
「搜索-选择过程下 FDR 无法从样本内统计量识别」的限制——这正是扫描器自造的
「有效试验数」门槛做不到的事。

两种候选池：

- `picks_from_plateaus`：每个平台取一个代表窗口。回答「幸存者里最好的那个稳不稳」。
- `picks_from_candidates`：扫描空间里的**全部**候选。回答「这 468 个候选本身是不是
  一片数据挖掘的沼泽」——后者才是忠实的 PBO，因为竞争就是在这 468 个之间发生的。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from src.research.factors.panel import Panel
from src.research.factors.plateau import KIND_PLATEAU
from src.research.factors.portfolio import (
    PortfolioOptions,
    PortfolioStats,
    break_even_cost,
    portfolio_stats,
)
from src.research.factors.sweep import SWEEP_FAMILIES
from src.research.validation.overfitting import build_overfitting_report
from src.research.validation.overlap import DEFAULT_EMBARGO_RATIO
from src.research.validation.trials import ResearchScope, Trial, TrialLedger

DEFAULT_QUANTILES: Final[int] = 5
# 每个候选跑一遍分位组合，是分钟级的活，进度必须能看见
ProgressFn = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PlateauPick:
    """一个候选：族 + 窗口 + 期限，外加交易方向。"""

    family: str
    horizon: int
    window: int
    # 交易方向（+1 做多高分位、−1 做空高分位）。扫描不预设方向，符号本身就是发现；
    # 但 PBO 按夏普排名，候选之间必须同向才有得比，所以统一按 IC 符号定向。
    direction: float

    @property
    def key(self) -> str:
        return f"{self.family}_{self.window}@h{self.horizon}"

    @property
    def label(self) -> str:
        side = "多" if self.direction > 0 else "空"
        return f"{self.family} 窗口 {self.window} 期限 {self.horizon} 做{side}高分位"


def representative_window(start: int, end: int) -> int:
    """取段的中点，不取端点。

    平台的端点是最容易被人事后挑出来的位置（边界处的窗口常因样本长度变化而更
    显著），中点没有可挑的自由度。
    """
    return int((start + end) // 2)


def _direction_of(mean_ic: float) -> float:
    """方向取 IC 的符号；IC 恰为 0 时退化成 +1（无信号，方向无意义）。"""
    return -1.0 if mean_ic < 0 else 1.0


def picks_from_plateaus(plateaus: pd.DataFrame) -> list[PlateauPick]:
    """平台表 → 每个平台一个代表窗口。尖峰不收：判据已经把它们丢了。"""
    if plateaus.empty:
        return []
    real = cast(pd.DataFrame, plateaus[plateaus["kind"] == KIND_PLATEAU])
    picks = [
        PlateauPick(
            family=str(record["family"]),
            horizon=int(cast(int, record["horizon"])),
            window=representative_window(
                int(cast(int, record["start"])), int(cast(int, record["end"]))
            ),
            direction=_direction_of(float(cast(float, record["mean_ic"]))),
        )
        for record in real.to_dict(orient="records")
    ]
    return sorted(picks, key=lambda pick: (pick.family, pick.horizon, pick.window))


def picks_from_candidates(
    ic: pd.DataFrame, horizons: Sequence[int] | None = None
) -> list[PlateauPick]:
    """全候选池：扫描空间里的每一个 (族, 窗口, 期限)，方向取该候选自身 IC 的符号。"""
    if ic.empty:
        return []
    wanted = None if horizons is None else {int(item) for item in horizons}
    picks = [
        PlateauPick(
            family=str(record["family"]),
            horizon=int(cast(int, record["horizon"])),
            window=int(cast(int, record["window"])),
            direction=_direction_of(float(cast(float, record["ic"]))),
        )
        for record in ic.to_dict(orient="records")
        if wanted is None or int(cast(int, record["horizon"])) in wanted
    ]
    return sorted(picks, key=lambda pick: (pick.horizon, pick.family, pick.window))


def _options(pick: PlateauPick, cost_one_way: float | None) -> PortfolioOptions:
    # 锚点固定为 0：所有候选共用同一相位，否则 PBO 的共同时间轴对不上
    base = {"horizon": pick.horizon, "quantiles": DEFAULT_QUANTILES, "anchor": 0}
    if cost_one_way is None:
        return PortfolioOptions(**base)
    return PortfolioOptions(cost_one_way=cost_one_way, **base)


def run_pick(
    pick: PlateauPick,
    panel: Panel,
    mask: pd.DataFrame,
    cost_one_way: float | None = None,
) -> PortfolioStats:
    """把候选跑成分位组合，返回完整统计（含逐期多空价差序列）。"""
    stats = portfolio_stats(
        SWEEP_FAMILIES[pick.family].spec(pick.window),
        panel,
        mask,
        _options(pick, cost_one_way),
    )
    return stats


def oriented_spread(pick: PlateauPick, stats: PortfolioStats) -> pd.Series:
    """按候选自己的方向翻转价差：翻转后「每期价差为正」就是「这个方向对」。"""
    return stats.spread_returns * pick.direction


@dataclass(frozen=True)
class VerificationResult:
    """收益层验证的产物。"""

    # 逐候选：族/期限/窗口/方向/期数/每期价差/年化/换手/平衡成本
    picks: pd.DataFrame
    # 每个候选的逐期价差序列（已定向），键为 PlateauPick.key
    series: dict[str, pd.Series]
    # 逐期限的 PBO 与 DSR：同一期限的候选才是直接竞争关系，跨期限混在一起比是错的
    by_horizon: pd.DataFrame
    # DSR 实际用的试验次数 N，以及本次往账本里新增的条数
    trials: int = 0
    ledger_added: int = 0


def _row(pick: PlateauPick, stats: PortfolioStats) -> dict[str, object]:
    """一个候选的收益层读数。价差按候选方向定向后再取平衡成本。"""
    oriented = float(stats.gross_spread) * pick.direction
    return {
        "family": pick.family,
        "horizon": pick.horizon,
        "window": pick.window,
        "direction": pick.direction,
        "key": pick.key,
        "periods": stats.periods,
        "gross_per_period": oriented,
        "annual_gross": oriented * stats.periods_per_year,
        "turnover": stats.turnover,
        "break_even_cost": break_even_cost(oriented, stats.turnover),
    }


def _candidate_trials(
    picks: Sequence[PlateauPick], scope: ResearchScope
) -> list[Trial]:
    """把候选翻译成账本条目。

    方向不进 params：它是从 IC 符号推出来的，不是我们选择的自由度。算进去会让同一
    配置因为行情变化而变成两次试验，把 N 悄悄放大。
    """
    return [
        Trial(
            study=scope.study,
            market=scope.market,
            timeframe=scope.timeframe,
            kind=pick.family,
            params={"window": pick.window, "horizon": pick.horizon},
        )
        for pick in picks
    ]


def _resolve_trials(
    picks: Sequence[PlateauPick], scope: ResearchScope | None, declared: int | None
) -> tuple[int, int]:
    """定出 DSR 用的 N，并把本批候选记进账本，返回 (N, 新增条数)。

    默认给**全库累计**——DSR 问的是「在你试过的全部东西里，这个还算不算技能」，
    只数本批候选正是 de Prado 第 7 条 pitfall。窄作用域必须由调用方显式声明。
    """
    if scope is None:
        return (declared or len(picks)), 0
    ledger = TrialLedger()
    added = ledger.record(_candidate_trials(picks, scope))
    if declared is not None:
        return declared, added
    return max(ledger.count(), len(picks)), added


def verify(
    picks: Sequence[PlateauPick],
    panel: Panel,
    mask: pd.DataFrame,
    cost_one_way: float | None = None,
    trials: int | None = None,
    progress: ProgressFn | None = None,
    scope: ResearchScope | None = None,
) -> VerificationResult:
    """候选列表 → 收益层读数 + 逐期限 PBO/DSR。

    `trials` 是 DSR 的试验次数 N。**给了 `scope` 就不必声明**：本批候选会被追加进
    `src/research/validation/trials.py` 的账本，N 取账本的累计数。不给 scope 也不声明 trials
    时退化成「本批候选数」——那是低估（de Prado 第 7 条 pitfall），所以会打印警告。
    """
    trial_count, added = _resolve_trials(picks, scope, trials)
    if scope is None and trials is None:
        print(
            "⚠ 未声明 --trials 也未提供 scope：DSR 的 N 退化成「本批候选数」，会低估。\n"
            "  用 scripts/trials_ledger.py 看当前账本累计数。"
        )
    else:
        print(f"试验账本：本次新增 {added} 条，DSR 用的 N={trial_count}")

    series: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for position, pick in enumerate(picks, start=1):
        if progress is not None:
            progress(position, len(picks), pick.label)
        stats = run_pick(pick, panel, mask, cost_one_way)
        if stats.periods == 0:
            continue
        series[pick.key] = oriented_spread(pick, stats)
        rows.append(_row(pick, stats))

    table = pd.DataFrame(rows)
    if table.empty:
        return VerificationResult(table, series, pd.DataFrame(), trial_count, added)
    verdicts = _verdict_table(table, series, trial_count)
    return VerificationResult(table, series, verdicts, trial_count, added)


def _verdict_table(
    picks: pd.DataFrame, series: dict[str, pd.Series], trials: int | None
) -> pd.DataFrame:
    """逐期限各跑一次 PBO 与 DSR。期限 ≥15 日时分不出 4 块，整行跳过。

    purge=0：分位网格是 `index[start::horizon]`，观测非重叠、标签首尾相接，
    没有标签跨进测试块。禁运仍然要加——序列相关与标签重叠是两件事。
    """
    rows: list[dict[str, object]] = []
    for horizon, group in picks.groupby("horizon"):
        subset = {
            str(key): series[str(key)] for key in group["key"] if str(key) in series
        }
        report = build_overfitting_report(subset, trials, embargo=DEFAULT_EMBARGO_RATIO)
        pbo = report.pbo
        if pbo is None:
            continue
        tests = [test for test in report.sharpe_tests if test.sufficient]
        rows.append(
            {
                "horizon": int(cast(int, horizon)),
                "candidates": len(subset),
                "observations": report.observations,
                "blocks": pbo.blocks,
                "splits": pbo.splits,
                "purge": pbo.purge,
                "embargo": pbo.embargo,
                "pbo": pbo.pbo,
                "logit_median": pbo.logit_median,
                "winner_oos_median": pbo.winner_oos_sharpe_median,
                "passed": pbo.passed,
                "best_dsr": max((test.dsr for test in tests), default=float("nan")),
                "dsr_passed": sum(1 for test in tests if test.passed),
            }
        )
    return pd.DataFrame(rows)


def print_verification(result: VerificationResult, top: int = 20) -> None:
    """打印收益层读数（按期价差排序取前若干）与逐期限 PBO 裁决。"""
    if result.picks.empty:
        print("\n收益层验证：没有可跑的候选。")
        return
    print(
        f"\n试验次数 N={result.trials}（账本累计；只数本批候选会低估门槛）"
        f" | 本次新增 {result.ledger_added} 条"
    )
    ordered = cast(pd.DataFrame, result.picks).sort_values(
        "gross_per_period", ascending=False
    )
    print(f"\n=== 收益层验证（{len(result.picks)} 个候选，按每期价差排序前 {top}）===")
    print(
        f"{'族':>5}{'期限':>5}{'窗口':>6}{'方向':>5}{'期数':>6}{'每期价差':>11}"
        f"{'年化':>10}{'换手':>8}{'平衡成本':>10}"
    )
    for row in ordered.head(top).to_dict(orient="records"):
        print(
            f"{row['family']:>5}{row['horizon']:>5}{row['window']:>6}"
            f"{'多' if row['direction'] > 0 else '空':>5}{row['periods']:>6}"
            f"{row['gross_per_period'] * 100:>10.2f}%"
            f"{row['annual_gross'] * 100:>9.0f}%"
            f"{row['turnover']:>8.2f}"
            f"{row['break_even_cost'] * 1e4:>9.1f}bp"
        )
    if result.by_horizon.empty:
        print("\nPBO：不可算（每个期限的非重叠期数不够分出 4 块）。")
        return
    print("\n=== 逐期限过拟合检验 ===")
    print(
        f"{'期限':>5}{'候选':>6}{'期数':>6}{'分块':>5}{'切分':>7}{'PBO':>8}"
        f"{'中位logit':>11}{'冠军样本外中位':>15}{'最佳DSR':>9}{'DSR过':>7}"
    )
    for row in result.by_horizon.to_dict(orient="records"):
        print(
            f"{row['horizon']:>5}{row['candidates']:>6}{row['observations']:>6}"
            f"{row['blocks']:>5}{row['splits']:>7}{row['pbo']:>7.1%}"
            f"{row['logit_median']:>+11.2f}{row['winner_oos_median']:>15.4f}"
            f"{row['best_dsr']:>9.3f}{row['dsr_passed']:>7}"
        )
    print(
        "PBO 问的是：在这批候选里按样本内表现挑冠军，它在样本外跌破中位数的概率。"
        "高 PBO 说明这批候选本身是数据挖掘的沼泽，单个赢家不可信。"
    )
