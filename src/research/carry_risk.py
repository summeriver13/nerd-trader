"""carry 的风险侧：退出规则回测与爆仓情景。

前几轮把收益侧算清楚了（资金费 − Δbasis − 成本），本模块回答两个「无风险」最该被
检验的问题：

1. **费率转负就退出，能不能改善净收益？** 每次切换要付一次双腿往返成本，规则必须
   赚回它才划算；对长期负费率的标的（BNB 正费率占比只有 35.6%）尤其值得一试。
2. **永续空头会不会被强平？** 1.5x 资本（现货 1.0 + 保证金 0.5）对应永续 2x 杠杆，
   按 0.5% 维持保证金率算，永续涨约 49% 即触发强平。**现货腿的浮盈补不了永续
   保证金**（两个市场、两笔资金），所以被强平后组合就变成裸露的现货多头。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import CRYPTO_ROOT, MARKET_SWAP, crypto_dataset_dir
from src.research.carry import CAPITAL_MULTIPLE, ROUND_TRIP_COST
from src.research.carry_ranking import load_rates
from src.utils.data_sources.funding import daily_rates

EXCHANGE = "okx"
TIMEFRAME = "1d"
FUNDING_EXCHANGE = "binance"
# 永续腿保证金 / 名义：资本占用 1.5x 里现货 1.0、保证金 0.5，故永续是 2x 杠杆
PERP_MARGIN_RATIO = 0.5
# 维持保证金率：OKX 永续低杠杆档位近似值，用于算强平价
MAINTENANCE_MARGIN = 0.005
# 退出规则的滚动窗口（天）
RULE_WINDOWS: tuple[int, ...] = (7, 30, 90)
# 爆仓要看持仓期：持有 1 天 / 1 周 / 1 个月的最大逆向幅度完全不同
LIQUIDATION_HORIZONS: tuple[int, ...] = (1, 7, 30)


@dataclass(frozen=True)
class RuleResult:
    """一条退出规则的净表现（资本口径，已扣双腿成本）。"""

    label: str
    total_return: float
    max_drawdown: float
    days_in_market: int
    days_total: int
    changes: int
    cost: float


@dataclass(frozen=True)
class LiquidationRow:
    """一只标的在不同持仓期下的最坏涨幅与强平判定。"""

    stem: str
    days: int
    worst_1d: float
    worst_7d: float
    worst_30d: float
    threshold: float
    breached: str


def liquidation_gain(margin_ratio: float = PERP_MARGIN_RATIO) -> float:
    """永续空头在给定保证金率下，涨多少会被强平。

    保证金余额 = margin_ratio − x，维持保证金 = MAINTENANCE_MARGIN × (1 + x)，
    强平当 `margin_ratio − x < mm × (1 + x)`，解得 `x = (margin_ratio − mm) / (1 + mm)`。
    """
    return (margin_ratio - MAINTENANCE_MARGIN) / (1 + MAINTENANCE_MARGIN)


def trailing_signal(daily: pd.Series, window: int, threshold: float = 0.0) -> pd.Series:
    """过去 window 天累计资金费高于阈值时在场。

    先 shift 再比较：只依赖已完成的历史（防前视），且 NaN 与阈值比较天然为 False，
    不必 fillna——对布尔序列 fillna 会触发 object dtype 降级的 FutureWarning。
    """
    trailing = cast(pd.Series, daily.rolling(window).sum())
    shifted = cast(pd.Series, trailing.shift(1))
    return cast(pd.Series, shifted > threshold)


def apply_rule(daily: pd.Series, in_market: pd.Series, label: str) -> RuleResult:
    """按给定的在场序列回测：净收益扣掉每次切换的双腿往返成本。

    每次仓位变化（含首日建仓与末日平仓）算半次往返，`always` 基准恰好两次变化
    = 一次完整往返，与 carry.py 的成本口径一致。
    """
    position = cast(pd.Series, in_market.reindex(daily.index).eq(True))
    gross = cast(pd.Series, daily.where(position, 0.0))
    flags = np.concatenate([[False], position.to_numpy(dtype=bool), [False]])
    changes = int((flags[1:] != flags[:-1]).sum())
    cost = changes * ROUND_TRIP_COST / 2
    net = float((1 + gross).prod() - 1) - cost
    equity = (1 + gross).cumprod()
    drawdown = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    return RuleResult(
        label=label,
        total_return=net / CAPITAL_MULTIPLE,
        max_drawdown=drawdown / CAPITAL_MULTIPLE,
        days_in_market=int(position.sum()),
        days_total=len(position),
        changes=changes,
        cost=cost,
    )


def rule_results(daily: pd.Series) -> dict[str, RuleResult]:
    """基准（一直持有）加各滚动窗口的退出规则，按规则名索引。"""
    always = cast(pd.Series, pd.Series(True, index=daily.index))
    rows = {"always": apply_rule(daily, always, "always")}
    for window in RULE_WINDOWS:
        label = f"trail{window}"
        rows[label] = apply_rule(daily, trailing_signal(daily, window), label)
    return rows


def load_daily(stem: str, days: int) -> pd.Series:
    """一只标的的日资金费序列（截断到最近 days 天）。"""
    daily = daily_rates(load_rates(FUNDING_EXCHANGE, stem))
    return cast(pd.Series, daily.iloc[-days:]).dropna()


def load_swap_ohlc(stem: str) -> pd.DataFrame:
    """永续日线的 high / close / volume：爆仓看盘中高点，零成交日要能识别出来。"""
    path = (
        crypto_dataset_dir(EXCHANGE, MARKET_SWAP, TIMEFRAME, root=CRYPTO_ROOT)
        / f"{stem}.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return cast(pd.DataFrame, frame.sort_index()[["high", "close", "volume"]])


def _worst_run(close: pd.Series, high: pd.Series, horizon: int) -> float:
    """任一收盘建仓、其后 horizon 天内盘中高点相对建仓价的最大涨幅。

    先把序列补齐到完整日历（缺日留 NaN），再用 min_periods=horizon 的 rolling：
    窗口内只要有缺日就判为无效，避免「跨过缺口的假涨幅」——ADA 早期有 48 个
    零成交冻结日，剔除后仍会留下日期缺口，不补日历就仍会报出 +116%。
    """
    full = pd.date_range(close.index[0], close.index[-1], freq="1D", tz="UTC")
    close = close.reindex(full)
    high = high.reindex(full)
    future_high = (
        high.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].shift(-1)
    )
    return float((future_high / close - 1).max())


def liquidation_rows(stems: list[str]) -> list[LiquidationRow]:
    """算每只标的在不同持仓期下的最坏涨幅与强平判定。

    零成交日（实测仅 ADA 早期 48 天）是冻结价，会伪造出巨额「涨幅」——
    ADA 2020-04-24 至 04-28 连续 5 天 OHLC 冻在 0.02406、volume 为 0，
    04-29 首笔真实成交直接报出 +113%，先剔除。
    """
    threshold = liquidation_gain()
    rows: list[LiquidationRow] = []
    for stem in stems:
        frame = load_swap_ohlc(stem)
        if frame.empty:
            continue
        frame = frame[frame["volume"] > 0]
        if len(frame) < 8:
            continue
        close = cast(pd.Series, frame["close"])
        high = cast(pd.Series, frame["high"])
        worst = {h: _worst_run(close, high, h) for h in LIQUIDATION_HORIZONS}
        breached = next(
            (f"{h}d" for h in LIQUIDATION_HORIZONS if worst[h] > threshold), ""
        )
        rows.append(
            LiquidationRow(
                stem=stem,
                days=len(frame),
                worst_1d=worst[1],
                worst_7d=worst[7],
                worst_30d=worst[30],
                threshold=threshold,
                breached=breached,
            )
        )
    return rows
