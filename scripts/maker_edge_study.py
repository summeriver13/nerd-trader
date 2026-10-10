"""被动挂单执行质量研究：成交率与逆向选择。

用法：
    uv run python scripts/maker_edge_study.py --symbol "BTC/USDT:USDT" --timeframe 1m

要回答的问题：做市型策略（被动挂单赚价差）在 1m K 线上有没有立足点。
框架里的回测把「触及即成交」当既定事实，这里量化这个假设值多少钱、代价在哪。

口径（只用已收盘 K 线，无前视）：
- 第 t 根收盘时点挂限价单 limit = close_t × (1 ∓ offset)，buy 挂下方、sell 挂上方
- 成交判定：第 t+1 根 low ≤ limit（buy）或 high ≥ limit（sell），
  与 scripts/measure_maker_fill.py 的 bar 触及口径一致
- 成交价保守取 limit：跳空时真实成交价更好，故本口径偏保守
- 持有到第 t+1+h 根收盘；sell 方向按空头收益计价，两侧的 cond 为负都表示逆向选择

指标（单位 bp）：
- fill：挂单被触及的比例
- cond：成交样本上「按 t 时点市价开仓」到 t+1+h 的持有收益，负值即逆向选择
- maker：成交样本上「按挂单价成交」的同口径收益，含挂得便宜（或卖得贵）的那部分
- base：全样本同口径收益，作无条件基准；base 与 cond 的差就是选择效应
- net：maker 扣掉 maker 手续费
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.config.paths import crypto_bar_path

BP = 1e4
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_OFFSETS = "0.5,1,2,5,10"
DEFAULT_HORIZONS = "1,5,15,60"
DEFAULT_MAKER_FEE_BP = 2.0
EXCHANGE = "okx"
BAR_COLUMNS = ("open", "high", "low", "close", "volume")
SIDES = ("buy", "sell")
SIDE_LABELS = {"buy": "挂买单（买在收盘下方）", "sell": "挂卖单（卖在收盘上方）"}
VOL_LOOKBACK = 60
VOL_BUCKETS = 5
DETAIL_OFFSET_BP = 1.0
DETAIL_HORIZON = 15


@dataclass(frozen=True)
class StudySpec:
    """一次统计的组合参数。"""

    offset_bp: float
    horizon: int
    side: str
    fee_bp: float


@dataclass(frozen=True)
class FillStats:
    """某个组合的统计结果。"""

    spec: StudySpec
    n: int
    fill_rate: float
    base_ret_bp: float
    cond_ret_bp: float
    maker_ret_bp: float

    @property
    def net_ret_bp(self) -> float:
        """每笔成交扣掉 maker 手续费后的净收益。"""
        return self.maker_ret_bp - self.spec.fee_bp

    @property
    def per_order_bp(self) -> float:
        """每挂一笔单的期望净收益（未成交记 0）。"""
        return self.fill_rate * self.net_ret_bp


def load_bars(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """读缓存 parquet；直读而不经 data_loader 是为了不触发任何网络请求。"""
    frame = pd.read_parquet(
        crypto_bar_path(EXCHANGE, symbol, timeframe), columns=list(BAR_COLUMNS)
    )
    if bars > 0:
        frame = frame.tail(bars)
    return frame.reset_index()


def components(frame: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    """逐根算出是否成交与三种口径的收益，供整体与分组统计复用。"""
    close = frame["close"]
    sign = 1.0 if spec.side == "buy" else -1.0
    if spec.side == "buy":
        limit = close * (1 - spec.offset_bp / BP)
        touched = frame["low"].shift(-1) <= limit
    else:
        limit = close * (1 + spec.offset_bp / BP)
        touched = frame["high"].shift(-1) >= limit
    exit_price = close.shift(-(1 + spec.horizon))
    valid = limit.notna() & exit_price.notna()
    hit = valid & touched
    move = exit_price / close - 1
    return pd.DataFrame(
        {
            "valid": valid,
            "hit": hit,
            "base_ret": sign * move.where(valid) * BP,
            "cond_ret": sign * move.where(hit) * BP,
            "maker_ret": (sign * (exit_price / limit - 1)).where(hit) * BP,
        }
    )


def summarize(parts: pd.DataFrame) -> dict[str, float]:
    """把逐根明细聚合成成交率与三种收益。"""
    valid = parts["valid"]
    hit = parts["hit"]
    return {
        "n": int(valid.sum()),
        "fill_rate": float(hit.sum() / valid.sum()),
        "base_ret_bp": float(parts["base_ret"].mean()),
        "cond_ret_bp": float(parts["cond_ret"].mean()),
        "maker_ret_bp": float(parts["maker_ret"].mean()),
    }


def study(frame: pd.DataFrame, spec: StudySpec) -> FillStats:
    """整体统计一个组合。"""
    summary = summarize(components(frame, spec))
    return FillStats(
        spec=spec,
        n=int(summary["n"]),
        fill_rate=summary["fill_rate"],
        base_ret_bp=summary["base_ret_bp"],
        cond_ret_bp=summary["cond_ret_bp"],
        maker_ret_bp=summary["maker_ret_bp"],
    )


def group_study(
    frame: pd.DataFrame, groups: pd.Series, spec: StudySpec
) -> pd.DataFrame:
    """分组统计：每组的成交率与两种收益，base 与 cond 的差即该组的选择效应。"""
    parts = components(frame, spec)
    parts["group"] = groups.to_numpy()
    rows = [
        {"group": key, **summarize(chunk)}
        for key, chunk in parts.groupby("group", sort=True, observed=True)
    ]
    return pd.DataFrame(rows)


def hour_groups(frame: pd.DataFrame) -> pd.Series:
    """UTC 小时，做市流动性随时段变化。"""
    return frame["datetime"].dt.hour


def vol_groups(frame: pd.DataFrame) -> pd.Series:
    """近 VOL_LOOKBACK 根实现波动的分位桶，Q1 最静、Q5 最烈。"""
    vol = frame["close"].pct_change().rolling(VOL_LOOKBACK).std()
    labels = [f"Q{i}" for i in range(1, VOL_BUCKETS + 1)]
    return pd.Series(pd.qcut(vol, VOL_BUCKETS, labels=labels))


def print_stats_table(stats: list[FillStats], title: str) -> None:
    """主表：每档 offset × 每个持有期。"""
    header = (
        f"{'offset':>7} {'hold':>5} {'fill':>7} {'cond':>8} "
        f"{'maker':>8} {'base':>8} {'net':>8} {'per-ord':>8}"
    )
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.spec.offset_bp:>6.1f}b {item.spec.horizon:>5} "
            f"{item.fill_rate:>7.1%} {item.cond_ret_bp:>8.2f} "
            f"{item.maker_ret_bp:>8.2f} {item.base_ret_bp:>8.2f} "
            f"{item.net_ret_bp:>8.2f} {item.per_order_bp:>8.2f}"
        )


def print_group_table(rows: pd.DataFrame, title: str) -> None:
    """分组表。"""
    header = f"{'group':>7} {'n':>10} {'fill':>7} {'cond':>8} {'maker':>8} {'base':>8}"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for _, row in rows.iterrows():
        print(
            f"{str(row['group']):>7} {int(row['n']):>10,} "
            f"{float(row['fill_rate']):>7.1%} {float(row['cond_ret_bp']):>8.2f} "
            f"{float(row['maker_ret_bp']):>8.2f} {float(row['base_ret_bp']):>8.2f}"
        )


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(","))


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def run_study(args: argparse.Namespace) -> None:
    """跑全部统计并打印。"""
    frame = load_bars(args.symbol, args.timeframe, args.bars)
    offsets = parse_floats(args.offsets)
    horizons = parse_ints(args.horizons)
    sides = SIDES if args.side == "both" else (args.side,)

    span = f"{frame['datetime'].iloc[0]} → {frame['datetime'].iloc[-1]}"
    print(f"标的 {args.symbol}｜周期 {args.timeframe}｜{len(frame):,} 根｜{span}")
    print(f"maker 手续费按 {args.maker_fee_bp:.2f}bp/笔计（OKX Regular 永续口径）")

    for side in sides:
        specs = [
            StudySpec(off, hor, side, args.maker_fee_bp)
            for off in offsets
            for hor in horizons
        ]
        stats = [study(frame, spec) for spec in specs]
        print_stats_table(stats, f"主表｜{SIDE_LABELS[side]}（收益单位 bp）")

    detail = StudySpec(
        DETAIL_OFFSET_BP, DETAIL_HORIZON, args.side_first, args.maker_fee_bp
    )
    detail_title = f"分组明细｜offset {DETAIL_OFFSET_BP}bp、持有 {DETAIL_HORIZON} 根"
    hour_rows = group_study(frame, hour_groups(frame), detail)
    print_group_table(
        hour_rows, f"{detail_title}、{SIDE_LABELS[detail.side]}｜按 UTC 小时"
    )
    vol_rows = group_study(frame, vol_groups(frame), detail)
    print_group_table(
        vol_rows, f"{detail_title}、{SIDE_LABELS[detail.side]}｜按实现波动分位"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="被动挂单成交率与逆向选择研究（只读本地缓存，不联网）"
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="ccxt 符号")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="K 线周期")
    parser.add_argument(
        "--bars", type=int, default=0, help="只用末尾 N 根，0 表示全部（默认）"
    )
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS, help="挂单偏移 bp 列表")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="持有根数列表")
    parser.add_argument(
        "--side", default="both", choices=(*SIDES, "both"), help="挂单方向"
    )
    parser.add_argument(
        "--maker-fee-bp",
        type=float,
        default=DEFAULT_MAKER_FEE_BP,
        help="maker 手续费（bp/笔），net 列会扣掉它",
    )
    args = parser.parse_args()
    args.side_first = SIDES[0]
    run_study(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
