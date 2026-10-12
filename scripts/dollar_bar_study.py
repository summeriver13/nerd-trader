"""
美元棒 vs 时间棒：反转信号活在日历尺度还是信息尺度？

现有全部结论都建立在时间棒上——「3 根反转」是 3 个**日历日**。美元棒按成交额
采样，同一个窗口号是 3 个**信息单位**。实测每根美元棒的日历跨度 p5→p95 差
34.7 倍（0.07 天 → 3.8 天），所以两者是真正不同的尺度，而不是同一件事换个说法。

方法：两边用**同一份前向收益**（日线日历 h 日）与**同一个可交易掩码**，只有因子值
的度量方式不同——日线收盘算的 w 日收益 vs 美元棒收盘算的 w 根收益，后者再按日
采样。采样只取「当日收盘之前已收盘的最后一根美元棒」，无前视。

用法：
    uv run python scripts/dollar_bar_study.py [--market swap] [--horizon 5] [--no-save]

产物落 data/results/bars/dollar_vs_time.parquet。只读本地缓存，不联网。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import cast

import pandas as pd

from src.config.paths import CRYPTO_ROOT, RESULTS_ROOT
from src.research.factors.evaluation import (
    forward_returns,
    non_overlapping_t,
    rank_ic_series,
)
from src.research.factors.panel import Panel, UniverseRule, load_panel, tradable_mask

DEFAULT_WINDOWS = (1, 2, 3, 5, 10, 20, 30)
DEFAULT_HORIZONS = (1, 3, 5, 10, 20, 30)
SCALE_CALENDAR = "日历全池"
SCALE_CALENDAR_SHARED = "日历同池"
SCALE_INFORMATION = "信息同池"
DOLLAR_TIMEFRAME = "dollar"
BAR_COLUMNS = ("close",)


def _dollar_closes(market: str) -> dict[str, pd.Series]:
    """读美元棒收盘价序列（索引是每根棒的收盘时刻）。"""
    root = CRYPTO_ROOT / "okx" / market / DOLLAR_TIMEFRAME
    if not root.is_dir():
        raise FileNotFoundError(f"{root} 不存在：先跑 scripts/build_dollar_bars.py")
    return {
        path.stem: cast(
            pd.Series, pd.read_parquet(path, columns=list(BAR_COLUMNS))["close"]
        )
        for path in sorted(root.glob("*.parquet"))
    }


def _reversal(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """w 期反转：取负号，值越大越看多（与 library 里的方向约定一致）。"""
    return -(close / close.shift(window) - 1.0)


def _calendar_factor(panel: Panel, window: int) -> pd.DataFrame:
    """日历尺度：日线收盘上的 w 日反转。"""
    return _reversal(panel.close, window)


def _subset(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """把面板裁到给定标的集。"""
    return cast(pd.DataFrame, frame[[sym for sym in symbols if sym in frame.columns]])


def _information_factor(
    closes: dict[str, pd.Series], daily_index: pd.DatetimeIndex, window: int
) -> pd.DataFrame:
    """信息尺度：美元棒收盘上的 w 根反转，再按日采样。

    反转必须**逐标的**在它自己的棒序列上算。先拼 DataFrame 再 shift 是错的：
    各标的的美元棒时戳互不相同，拼出来的并集矩阵大部分是 NaN，shift(w) 会变成
    跨标的的乱移（实测 IC 处处恰好 0，很容易被当成「信号不存在」）。

    日线索引是 K 线的**起始**时刻，所以 `close[D]` 是 D 日末的价格；采样目标要
    相应加一天，取「D 日收盘之前已收盘的最后一根美元棒」——不加这一天就只用到
    D 日开盘前的信息，因子会白落后一天。
    """
    target = daily_index + pd.Timedelta(days=1)
    sampled = {
        symbol: (-(close / close.shift(window) - 1.0)).reindex(target, method="ffill")
        for symbol, close in closes.items()
    }
    frame = pd.DataFrame(sampled)
    frame.index = daily_index
    if bool(frame.isna().to_numpy().all()):
        raise ValueError(
            "信息尺度因子全为 NaN：美元棒时戳与面板索引对不上，检查抽样目标日"
        )
    return frame


def _ic_row(
    label: str,
    window: int,
    horizon: int,
    values: pd.DataFrame,
    forward: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[str, object]:
    """一个（尺度 × 窗口 × 期限）的 IC 读数。

    t 值只认非重叠子样本（与 `ic_stats` 同一口径）：重叠样本会把显著性虚高。
    """
    series = rank_ic_series(values, forward, mask)
    if series.empty:
        t_stat, independent = 0.0, 0
    else:
        t_stat, independent = non_overlapping_t(series, horizon)
    return {
        "scale": label,
        "window": window,
        "horizon": horizon,
        "mean_ic": float(series.mean()) if not series.empty else 0.0,
        "t_stat": t_stat,
        "periods": int(series.size),
        "independent": independent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="美元棒 vs 时间棒的反转尺度对比")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--min-notional", type=float, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    windows = [int(part) for part in args.windows.split(",") if part]
    horizons = [int(part) for part in args.horizons.split(",") if part]
    rule = (
        UniverseRule()
        if args.min_notional is None
        else UniverseRule(min_notional_usd=args.min_notional)
    )

    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel, rule)
    closes = _dollar_closes(args.market)
    shared = [sym for sym in panel.symbols if sym in closes]
    print(
        f"美元棒 vs 时间棒 | {args.market} · {args.timeframe} | "
        f"日线 {len(panel.symbols)} 只，美元棒 {len(closes)} 只，交集 {len(shared)} 只"
    )

    started = time.time()
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        forward = forward_returns(panel, horizon)
        for window in windows:
            for scale, values in (
                (SCALE_CALENDAR, _calendar_factor(panel, window)),
                (
                    SCALE_CALENDAR_SHARED,
                    _subset(_calendar_factor(panel, window), shared),
                ),
                (
                    SCALE_INFORMATION,
                    _information_factor(closes, panel.index, window),
                ),
            ):
                rows.append(
                    _ic_row(
                        scale,
                        window,
                        horizon,
                        values,
                        forward,
                        _subset(mask, shared) if scale != SCALE_CALENDAR else mask,
                    )
                )
        print(
            f"  期限 {horizon:>3} 日完成（{time.time() - started:.0f} 秒）", flush=True
        )

    table = pd.DataFrame(rows)
    print(_report(table))
    if not args.no_save:
        target = RESULTS_ROOT / "bars"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "dollar_vs_time.parquet"
        table.to_parquet(path)
        print(f"\n落盘 {path}")
    return 0


def _report(table: pd.DataFrame) -> str:
    """并排打印三个口径：全池日历、同池日历、同池信息。

    信息尺度的池子只有 30 只（1m 缓存所限），所以必须多给一列**同池日历**——
    否则「信息 IC 更弱」可能只是池子换成了大币，与尺度无关。
    """
    lines = ["\n=== 反转因子的 IC（同前向收益、同掩码，只换度量尺度）==="]
    for horizon, group in table.groupby("horizon"):
        pivot = group.pivot_table(
            index="window", columns="scale", values="mean_ic", aggfunc="first"
        )
        counts = group.pivot_table(
            index="window", columns="scale", values="periods", aggfunc="first"
        )
        lines.append(f"\n前向 {cast(int, horizon)} 日")
        lines.append(
            f"{'窗口':>6}{'全池日历':>11}{'同池日历':>11}{'同池信息':>11}"
            f"{'信息−日历':>12}{'期数':>8}"
        )
        for window in pivot.index:
            full = float(pivot.loc[window, SCALE_CALENDAR])
            cal = float(pivot.loc[window, SCALE_CALENDAR_SHARED])
            info = float(pivot.loc[window, SCALE_INFORMATION])
            lines.append(
                f"{int(window):>6}{full:>11.4f}{cal:>11.4f}{info:>11.4f}"
                f"{info - cal:>12.4f}"
                f"{int(counts.loc[window, SCALE_INFORMATION]):>8}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
