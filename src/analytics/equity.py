"""
src/analytics/equity.py
权益序列采集：按每根 K 线 mark-to-market 记录账户权益与买入持有基准，并落盘。

口径与向量化引擎一致：收益/回撤都取自这条序列，回撤用 metrics 里的唯一定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analytics.metrics import max_drawdown
from src.analytics.reporting import save_equity_series


@dataclass
class EquityTracker:
    """逐根采样权益：现金 + 持仓市值由调用方按最新价算好传进来。"""

    initial_capital: float
    single_symbol: bool = True
    points: list[tuple[datetime, float]] = field(default_factory=list)
    benchmark_points: list[tuple[datetime, float]] = field(default_factory=list)
    _first_close: float = 0.0

    def record(self, stamp: datetime, equity: float, close: float) -> None:
        """记一个采样点；单标的时顺带记录买入持有基准（多标的基准口径未定义）。"""
        self.points.append((stamp, equity))
        if not self.single_symbol:
            return
        if not self.benchmark_points:
            self._first_close = close
        growth = close / self._first_close if self._first_close else 1.0
        self.benchmark_points.append((stamp, self.initial_capital * growth))

    def save(self, path: Path) -> None:
        """落盘净值/基准/回撤三条序列，并打印权益口径的收益与回撤。"""
        if not self.single_symbol or len(self.points) < 2:
            return

        index = pd.DatetimeIndex([stamp for stamp, _ in self.points])
        equity = pd.Series([value for _, value in self.points], index=index)
        benchmark = pd.Series(
            [value for _, value in self.benchmark_points], index=index
        )
        net_ret = equity.pct_change().fillna(0.0)

        save_equity_series(
            {
                "equity_curve": equity,
                "benchmark_curve": benchmark,
                "drawdown_curve": equity / equity.cummax() - 1,
            },
            path,
        )
        total_ret = float(equity.iloc[-1]) / self.initial_capital - 1
        print(f"📈 净值序列已保存: {path}")
        print(
            f"   总收益 {total_ret:+.2%} | 最大回撤 {max_drawdown(net_ret):.2%} | "
            f"采样 {len(equity)} 根"
        )
