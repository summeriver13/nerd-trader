"""池尾成本实测：14 只最不流动标的的 1m 数据能不能测出价差（de Prado Ch 19 的遗留问题）。

Ch 19 的实测单边成本（5.0 ~ 5.4bp）只建在流动性最好的 30 个标的上，而 liq_30 与反转的溢价
恰好发生在池尾——「池尾只会更贵」此前只是推断。本脚本给 14 只池尾标的（180 天 1m，由
`scripts/fetch_crypto_bars.py` 抓取）量三件事，并用同样口径量两组对照：

1. Corwin-Schultz 在**全部分钟**上的可估日占比——Ch 19 用的就是它；
2. 同一估计量只喂**有成交且价格移动**的分钟——检验失效是不是「喂了太多不动的分钟」；
3. Roll 的负自相关估计——池尾唯一还有覆盖率的估计量，但薄市场的价格离散会污染它。

对照组必须能估出来（最流动的 BTC/ETH/SOL、1m 覆盖里最不流动的 1INCH/BAT/MAGIC），否则说明
是估计量实现的问题而不是市场的问题；另用 Ch 19 的 30 只聚合缓存算同一窗口的基线。

读数落 data/results/breaks/pool_tail_costs.parquet。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import CRYPTO_ROOT
from src.core.costs import PERP_FEES
from src.research.microstructure import corwin_schultz_spread_bp, roll_spread_bp

RESULTS_DIR: Path = Path("data/results/breaks")
MICRO_CACHE: Path = RESULTS_DIR / "microstructure_swap_1m_daily.parquet"
READINGS_PATH: Path = RESULTS_DIR / "pool_tail_costs.parquet"
# taker 手续费：价差测不出来时，成本仍有的那个硬底
TAKER_FEE_BP: float = PERP_FEES.taker * 1e4
BP_PER_UNIT = 1e4
DEFAULT_WINDOW_DAYS = 180
# 预登记：180 天窗口下最不流动的 14 只（README 第二十二节第 5 小节遗留的那批）
TAIL_SYMBOLS = (
    "LQTY",
    "IOST",
    "BAND",
    "GMX",
    "GLM",
    "METIS",
    "CVX",
    "ETHW",
    "ZETA",
    "GAS",
    "QTUM",
    "ACH",
    "ONE",
    "AEVO",
)
CONTRAST_SYMBOLS = ("BTC", "ETH", "SOL", "1INCH", "BAT", "MAGIC")
COLUMNS = (
    "组",
    "天数",
    "根/日",
    "CS可估占比",
    "CS中位(bp)",
    "活跃分钟CS占比",
    "Roll可估占比",
    "Roll中位(bp)",
)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="池尾 1m 成本可测性")
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="窗口天数"
    )
    parser.add_argument("--tail", nargs="+", default=list(TAIL_SYMBOLS), help="池尾标的")
    parser.add_argument(
        "--contrast", nargs="+", default=list(CONTRAST_SYMBOLS), help="对照标的"
    )
    parser.add_argument("--no-save", action="store_true", help="只打印不落盘")
    return parser.parse_args()


def load_minutes(name: str, window_days: int) -> pd.DataFrame | None:
    """读一只标的的 1m 棒，截到窗口内的最后 window_days 天。"""
    path = CRYPTO_ROOT / "okx" / "swap" / "1m" / f"{name}-USDT.parquet"
    if not path.exists():
        print(f"{name}-USDT 无 1m 缓存，跳过")
        return None
    frame = pd.read_parquet(path).loc[:, ["high", "low", "close", "volume"]].astype(float)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    stamps = pd.Series(frame.index).dt.normalize().to_numpy()
    cutoff = stamps.max() - np.timedelta64(window_days, "D")
    return frame.loc[stamps >= cutoff]


def daily_estimates(frame: pd.DataFrame, active_only: bool) -> dict[str, float]:
    """逐日估 CS 与 Roll，返回覆盖率与中位读数。

    `active_only=True` 时只喂「有成交且价格移动」的分钟：CS 假设每分钟的高低幅里含价差，
    不动分钟的幅值只是噪声，把它们喂进去会把 α 压到 0 以下（原文口径是钳 0 而不是丢弃）。
    """
    block = frame
    if active_only:
        block = block.loc[(block["high"] > block["low"]) & (block["volume"] > 0.0)]
    if block.empty:
        return {
            "days": 0.0,
            "bars_per_day": 0.0,
            "cs_share": 0.0,
            "cs_median_bp": np.nan,
            "roll_share": 0.0,
            "roll_median_bp": np.nan,
        }
    dates = pd.Series(block.index).dt.normalize()
    codes, uniques = pd.factorize(dates)
    starts = np.searchsorted(codes, np.arange(len(uniques)))
    high = block["high"].to_numpy(dtype=float)
    low = block["low"].to_numpy(dtype=float)
    close = block["close"].to_numpy(dtype=float)
    cs_values: list[float] = []
    cs_days = 0
    roll_values: list[float] = []
    for index in range(len(uniques)):
        lo = int(starts[index])
        hi = int(starts[index + 1]) if index + 1 < len(starts) else codes.size
        cs = corwin_schultz_spread_bp(high[lo:hi], low[lo:hi])
        finite = cs[np.isfinite(cs)]
        if finite.size:
            value = float(np.median(finite))
            if value > 0.0:
                cs_days += 1
                cs_values.append(value)
        if hi - lo > 1:
            roll = roll_spread_bp(np.diff(np.log(close[lo:hi])) * BP_PER_UNIT)
            if np.isfinite(roll):
                roll_values.append(roll)
    days = max(len(uniques), 1)
    return {
        "days": float(len(uniques)),
        "bars_per_day": float(codes.size / days),
        "cs_share": cs_days / days,
        "cs_median_bp": float(np.median(cs_values)) if cs_values else np.nan,
        "roll_share": len(roll_values) / days,
        "roll_median_bp": float(np.median(roll_values)) if roll_values else np.nan,
    }


def symbol_row(name: str, window_days: int) -> dict[str, object] | None:
    """一只标的的一行读数。"""
    frame = load_minutes(name, window_days)
    if frame is None or frame.empty:
        return None
    all_minutes = daily_estimates(frame, active_only=False)
    active = daily_estimates(frame, active_only=True)
    return {
        "symbol": f"{name}-USDT",
        "天数": all_minutes["days"],
        "根/日": all_minutes["bars_per_day"],
        "CS可估占比": all_minutes["cs_share"],
        "CS中位(bp)": all_minutes["cs_median_bp"],
        "活跃分钟CS占比": active["cs_share"],
        "Roll可估占比": all_minutes["roll_share"],
        "Roll中位(bp)": all_minutes["roll_median_bp"],
    }


def covered_baseline(window_days: int) -> pd.Series | None:
    """Ch 19 那 30 只在同一窗口下的 CS 可估占比（每只一个数）。"""
    if not MICRO_CACHE.exists():
        return None
    frame = pd.read_parquet(MICRO_CACHE).loc[:, ["symbol", "date", "cs_spread_bp"]]
    stamps = pd.DatetimeIndex(frame["date"]).to_numpy()
    cutoff = stamps.max() - np.timedelta64(window_days, "D")
    frame = frame.loc[stamps >= cutoff]
    share = frame["cs_spread_bp"].to_numpy(dtype=float) > 0.0
    frame = frame.assign(可估=share.astype(float))
    return frame.groupby("symbol")["可估"].mean()


def main() -> None:
    """主流程。"""
    args = parse_args()
    rows: list[dict[str, object]] = []
    for group, names in (("池尾", args.tail), ("对照", args.contrast)):
        for name in names:
            row = symbol_row(name, args.window_days)
            if row is None:
                continue
            row["组"] = group
            rows.append(row)
    if not rows:
        print("没有可读的 1m 缓存")
        return
    table = pd.DataFrame(rows).set_index("symbol")
    print(f"### 逐标的（窗口 {args.window_days} 天，1m）")
    print(table.loc[:, list(COLUMNS)].round(3).to_string())

    tail = table.loc[table["组"] == "池尾"]
    contrast = table.loc[table["组"] == "对照"]
    print(
        f"\n池尾中位：CS 可估占比 {tail['CS可估占比'].median():.3f}"
        f"、活跃分钟上也只 {tail['活跃分钟CS占比'].median():.3f}"
        f"、Roll 可估占比 {tail['Roll可估占比'].median():.3f}"
    )
    print(
        f"对照中位：CS 可估占比 {contrast['CS可估占比'].median():.3f}"
        f"、活跃分钟 {contrast['活跃分钟CS占比'].median():.3f}"
        f"、Roll 可估占比 {contrast['Roll可估占比'].median():.3f}"
    )
    baseline = covered_baseline(args.window_days)
    if baseline is not None:
        print(
            f"Ch 19 覆盖的 {baseline.size} 只（同窗口）：CS 可估占比中位 {baseline.median():.3f}"
        )
    print(
        f"\n手续费硬底：单边 {TAKER_FEE_BP:.1f}bp（taker）——池尾价差测不出来时，"
        "成本只知下界，Ch 19 的 5.4bp 不能外推到这里"
    )
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_parquet(READINGS_PATH)
        print(f"落盘 {READINGS_PATH}")


if __name__ == "__main__":
    main()
