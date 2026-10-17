"""前向账本的退役判据：什么情况下认定这条策略「不再是候选」。

为什么需要它：`docs/trend/README.md` 把这条线定位为**候选**而非可上线，剩下的唯一
动作是攒前向样本。样本进来之后必须有一个**事前**写死的裁决口径，否则「再观察观察」
会一直拖到回撤发生时用事后叙事解释——判据的价值就在于它不能让任何人临时改。

预注册的阈值（写死在下面的常量里，前向记录不允许反过来改这些数）：

===============  ==============  ====================================================
判据             阈值            理由
===============  ==============  ====================================================
滚动一年夏普     < 0.00 → 退役   有仓位还亏钱。不是等它「跌到负数」才动手，而是符号
                                 反了——研究期 1.11 的符号被前向数据推翻
滚动一年夏普     < 0.30 → 观察   研究期夏普（1.11）的一半以下即半衰，此时不下结论
                                 但必须开始找原因
前向最大回撤     −17.1% → 观察   研究期最大回撤；前向超过它说明风险口径与回测不符
前向最大回撤     −25.7% → 退役   1.5 × 研究期最大回撤：不是「亏得更多」，是风险模型错
平均暴露         ≥ 5% 才判       低夏普若来自空仓则是信号没被触发，不是信号失效
样本长度         ≥ 365 根才判    未满一年全是噪声，此时只报「积累中」
===============  ==============  ====================================================

两条退役判据都**只依赖账本自己的净值与暴露**，不需要任何外部输入，所以能每天自动跑。
第三条判据（实现滑点超过假设的 2 倍）需要真实成交数据，等执行层接入后按同一格式追加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

BARS_PER_YEAR: Final[int] = 365
MIN_BARS: Final[int] = 365  # 未满一年不判定
SHARPE_WINDOW: Final[int] = 365  # 滚动一年
SHARPE_WATCH: Final[float] = 0.30  # 研究期夏普 1.11 的一半
SHARPE_KILL: Final[float] = 0.0
EXPOSURE_FLOOR: Final[float] = 0.05  # 低于此暴露视为「没在市场里」，低夏普不算数
DD_WATCH: Final[float] = -0.171  # 研究期最大回撤
DD_KILL: Final[float] = -0.257  # 1.5 × 研究期最大回撤

VERDICT_ACCUMULATING: Final[str] = "accumulating"
VERDICT_HEALTHY: Final[str] = "healthy"
VERDICT_WATCH: Final[str] = "watch"
VERDICT_KILL: Final[str] = "kill"

_EXIT_CODES: Final[dict[str, int]] = {
    VERDICT_ACCUMULATING: 0,
    VERDICT_HEALTHY: 0,
    VERDICT_WATCH: 1,
    VERDICT_KILL: 2,
}


@dataclass(frozen=True)
class RetireStatus:
    """一次裁决的完整读数：判决 + 支撑它的原始数字 + 触发的判据文本。"""

    bars: int
    verdict: str
    rolling_sharpe: float
    max_dd: float
    exposure: float
    reasons: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        """0 = 积累中/健康，1 = 观察，2 = 退役。给计划任务与 CI 用的机器口径。"""
        return _EXIT_CODES[self.verdict]

    def summary(self) -> str:
        """一行判决，供日志与账本尾部打印。"""
        return (
            f"[{self.verdict}] 期数 {self.bars}"
            f" ｜ 滚动一年夏普 {self.rolling_sharpe:.2f}"
            f" ｜ 前向最大回撤 {self.max_dd:.1%}"
            f" ｜ 平均暴露 {self.exposure:.1%}"
        )


def rolling_sharpe(nav: pd.Series, window: int = SHARPE_WINDOW) -> float:
    """最近 ``window`` 根的夏普（不足窗口就用全部可得，并在判决里标注积累中）。"""
    net = nav.astype(float).pct_change().dropna()
    if len(net) < 2:
        return 0.0
    tail = net.tail(window)
    std = float(tail.std())
    if std == 0.0:
        return 0.0
    return float(tail.mean()) / std * (BARS_PER_YEAR**0.5)


def retire_status(panel: pd.DataFrame) -> RetireStatus:
    """按预注册阈值裁决前向账本（``panel`` 为 ``paper.ledger_panel`` 的输出）。"""
    nav = panel["nav"].astype(float)
    bars = int(len(nav) - 1)  # 首行是起点，没有收益
    if bars <= 0:
        return RetireStatus(
            0, VERDICT_ACCUMULATING, 0.0, 0.0, 0.0, ("账本只有起点，还没有收益行",)
        )

    exposure = float(panel["exposure"].astype(float).mean())
    sharpe = rolling_sharpe(cast(pd.Series, nav))
    drawdown = float((nav / nav.cummax() - 1.0).min())
    reasons: list[str] = []

    if bars < MIN_BARS:
        reasons.append(f"样本不足：{bars}/{MIN_BARS} 根，未满一年不判定")
        return RetireStatus(
            bars, VERDICT_ACCUMULATING, sharpe, drawdown, exposure, tuple(reasons)
        )

    verdict = VERDICT_HEALTHY
    in_market = exposure >= EXPOSURE_FLOOR
    if not in_market:
        reasons.append(
            f"平均暴露 {exposure:.1%} < 门槛 {EXPOSURE_FLOOR:.0%}：低夏普可能只是没被触发，不据此退役"
        )
    if in_market and sharpe < SHARPE_KILL:
        verdict = VERDICT_KILL
        reasons.append(
            f"滚动一年夏普 {sharpe:.2f} < {SHARPE_KILL:.2f} 且暴露 {exposure:.1%}：有仓位还在亏"
        )
    elif in_market and sharpe < SHARPE_WATCH:
        verdict = VERDICT_WATCH
        reasons.append(
            f"滚动一年夏普 {sharpe:.2f} < {SHARPE_WATCH:.2f}：研究期读数半衰，需找原因"
        )

    if drawdown <= DD_KILL:
        verdict = VERDICT_KILL
        reasons.append(
            f"前向最大回撤 {drawdown:.1%} ≤ {DD_KILL:.1%}（研究期 1.5 倍）：风险模型错"
        )
    elif drawdown <= DD_WATCH and verdict != VERDICT_KILL:
        verdict = VERDICT_WATCH
        reasons.append(
            f"前向最大回撤 {drawdown:.1%} ≤ {DD_WATCH:.1%}（研究期读数）：已超出回测口径"
        )

    if not reasons:
        reasons.append("全部判据在阈值内")
    return RetireStatus(bars, verdict, sharpe, drawdown, exposure, tuple(reasons))


def status_report(status: RetireStatus) -> str:
    """多行判决文本（脚本打印用；逻辑仍在 ``retire_status`` 里）。"""
    lines = [status.summary(), f"判据：{'；'.join(status.reasons)}"]
    if status.verdict == VERDICT_ACCUMULATING:
        lines.append(
            f"距首次裁决还差 {MIN_BARS - status.bars} 根（每日一根，约 {MIN_BARS - status.bars} 天）"
        )
    return "\n".join(lines)
