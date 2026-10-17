"""1m 尺度方向性信号扫描：过去 N 根收益对未来 M 根收益的预测力。

用法：
    uv run python scripts/signal_scan.py --symbol "BTC/USDT:USDT" --timeframe 1m

要回答的问题：1m 剥头皮有没有可预测性。
判据不是「准不准」，而是「信号强度能否覆盖执行成本」——
taker 往返约 20bp、maker 每笔约 3.4bp（见 docs/execution/README.md）。

口径（只用已收盘 K 线，无前视）：
- 信号：past = close_t / close_{t-N} - 1，用 t 及之前的信息
- 目标：fut = close_{t+M} / close_t - 1，t 之后的价格
- 每个 (N, M) 组合报三件事：全样本相关系数、高低两组的未来收益差、该价差的 t 值
- 高低组按 past 的 20% / 80% 分位切；价差 = 高组未来收益 - 低组未来收益，
  正价差表示动量（追涨有利），负价差表示反转（抄底有利）

显著性按持有期抽稀：只取每 M 根一个样本算 t 值，避免重叠区间把自相关
算成独立样本而虚高自由度。价差与分位阈值仍用全样本估计——抽稀会把 4 小时的
组合缩到一万多个观测，估计噪音过大。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import crypto_bar_path
from src.utils.study_cli import add_timeframe_argument

BP = 1e4
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_NS = "5,15,30,60,120,240"
DEFAULT_MS = "5,15,30,60,120,240"
EXCHANGE = "okx"
BAR_COLUMNS = ("open", "high", "low", "close", "volume")
TAIL = 0.2
COST_TARGETS_BP = (3.4, 20.0)


@dataclass(frozen=True)
class ScanRow:
    """一个 (N, M) 组合的扫描结果。"""

    n: int
    m: int
    samples: int
    ic: float
    spread_bp: float
    t_stat: float


def load_bars(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """读缓存 parquet；直读而不经 data_loader 是为了不触发任何网络请求。"""
    frame = pd.read_parquet(
        crypto_bar_path(EXCHANGE, symbol, timeframe), columns=list(BAR_COLUMNS)
    )
    if bars > 0:
        frame = frame.tail(bars)
    return frame.reset_index()


def welch_t(high: pd.Series, low: pd.Series) -> float:
    """Welch t 值：两组均值差除以标准误，不作等方差假设。"""
    if len(high) < 2 or len(low) < 2:
        return float("nan")
    se = np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))
    if se == 0:
        return float("nan")
    return float((high.mean() - low.mean()) / se)


def scan_one(frame: pd.DataFrame, n: int, m: int) -> ScanRow:
    """扫一个 (N, M) 组合。"""
    close = frame["close"]
    past = close.pct_change(n)
    fut = close.shift(-m) / close - 1
    valid = past.notna() & fut.notna()
    signal = cast(pd.Series, past[valid])
    target = cast(pd.Series, fut[valid])

    # 分位阈值与价差都用全样本估计；只有 t 值放到抽稀样本上算，
    # 避免重叠区间被当成独立样本而虚高显著性
    lo_edge = signal.quantile(TAIL)
    hi_edge = signal.quantile(1 - TAIL)
    high_all = target[signal >= hi_edge]
    low_all = target[signal <= lo_edge]
    thin = slice(None, None, m)
    thin_signal, thin_target = signal.iloc[thin], target.iloc[thin]
    high_thin = thin_target[thin_signal >= hi_edge]
    low_thin = thin_target[thin_signal <= lo_edge]

    return ScanRow(
        n=n,
        m=m,
        samples=int(valid.sum()),
        ic=float(signal.corr(target)),
        spread_bp=float((high_all.mean() - low_all.mean()) * BP),
        t_stat=welch_t(high_thin, low_thin),
    )


def quantile_detail(frame: pd.DataFrame, n: int, m: int, groups: int) -> pd.DataFrame:
    """把信号切成若干档，看未来收益是否单调。"""
    close = frame["close"]
    past = close.pct_change(n)
    fut = close.shift(-m) / close - 1
    valid = past.notna() & fut.notna()
    signal = cast(pd.Series, past[valid])
    target = cast(pd.Series, fut[valid])
    bucket = pd.qcut(signal.rank(method="first"), groups, labels=False)
    rows = [
        {
            "bucket": index + 1,
            "n": int(len(chunk)),
            "signal_bp": float(signal[bucket == key].mean() * BP),
            "fut_bp": float(chunk.mean() * BP),
        }
        for index, (key, chunk) in enumerate(target.groupby(bucket, observed=True))
    ]
    return pd.DataFrame(rows)


def print_tables(rows: list[ScanRow], ms: tuple[int, ...]) -> None:
    """三张矩阵：相关系数、高低组价差、价差 t 值。"""
    by_key = {(row.n, row.m): row for row in rows}
    ns = list(dict.fromkeys(row.n for row in rows))
    for title, attr, fmt in (
        ("相关系数（全样本 Pearson）", "ic", "{:>9.4f}"),
        ("高低组未来收益差 Q5-Q1（bp）", "spread_bp", "{:>9.2f}"),
        ("价差 t 值（按 M 抽稀）", "t_stat", "{:>9.2f}"),
    ):
        header = f"{'N \\ M':>6}" + "".join(f"{m:>9}" for m in ms)
        print(f"\n{title}")
        print(header)
        print("-" * len(header))
        for n in ns:
            cells = "".join(fmt.format(getattr(by_key[(n, m)], attr)) for m in ms)
            print(f"{n:>6}{cells}")


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def run_scan(args: argparse.Namespace) -> None:
    """跑全部组合并打印。"""
    frame = load_bars(args.symbol, args.timeframe, args.bars)
    ns = parse_ints(args.ns)
    ms = parse_ints(args.ms)

    span = f"{frame['datetime'].iloc[0]} → {frame['datetime'].iloc[-1]}"
    print(f"标的 {args.symbol}｜周期 {args.timeframe}｜{len(frame):,} 根｜{span}")
    print(
        f"成本门槛参考：maker 每笔约 {COST_TARGETS_BP[0]:.1f}bp、"
        f"taker 往返约 {COST_TARGETS_BP[1]:.0f}bp（价差要超过门槛才有意义）"
    )

    rows = [scan_one(frame, n, m) for n in ns for m in ms]
    print_tables(rows, ms)

    best = max(rows, key=lambda row: abs(row.spread_bp))
    print(
        f"\n最大｜价差｜组合：N={best.n} M={best.m}"
        f"｜价差 {best.spread_bp:.2f}bp｜t={best.t_stat:.2f}｜样本 {best.samples:,}"
    )
    detail = quantile_detail(frame, best.n, best.m, args.groups)
    print(f"{'档':>4} {'样本':>12} {'信号均值bp':>12} {'未来收益bp':>12}")
    print("-" * 44)
    for _, row in detail.iterrows():
        print(
            f"{int(row['bucket']):>4} {int(row['n']):>12,} "
            f"{float(row['signal_bp']):>12.2f} {float(row['fut_bp']):>12.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="1m 尺度方向性信号扫描（只读本地缓存，不联网）"
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="ccxt 符号")
    add_timeframe_argument(parser, default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--bars", type=int, default=0, help="只用末尾 N 根，0 表示全部（默认）"
    )
    parser.add_argument("--ns", default=DEFAULT_NS, help="回看根数列表")
    parser.add_argument("--ms", default=DEFAULT_MS, help="持有根数列表")
    parser.add_argument("--groups", type=int, default=5, help="明细的分档数")
    args = parser.parse_args()
    run_scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
