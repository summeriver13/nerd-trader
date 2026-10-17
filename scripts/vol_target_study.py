"""
波动率目标仓位统一测试台：全网格 + 等平均暴露对照 + 四段证伪 + 机制检验。

回答的问题：按近期实现波动率反比调仓位，能不能改善 BTC 的风险调整收益？

核心对照是「等平均暴露的固定仓位」——波动率目标天然会降低平均暴露，而回撤随暴露
近似线性缩小；不扣掉这一项，任何降暴露的做法都会在回撤指标上「赢」。表格里的
`gain` 列都是「波动率目标 − 同平均暴露固定仓位」。

用法：
    uv run python scripts/vol_target_study.py
    uv run python scripts/vol_target_study.py --symbol ETH/USDT
    uv run python scripts/vol_target_study.py --no-save

产出：控制台六张表 + data/results/vol_target/ 下五份 parquet。
表头用 ASCII：Windows GBK 控制台下中文列宽会错位。
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import pandas as pd

from src.analytics.metrics import ANNUAL_DAYS
from src.config.paths import crypto_bar_path
from src.research.trend.vol_target import (
    BASELINE_LABEL,
    VolTargetRun,
    build_grid,
    run_baseline,
)
from src.research.trend.vol_target_report import (
    DEFAULT_COST_SPEC,
    VolTargetStudy,
    build_study,
    save_outputs,
)
from src.utils.study_cli import add_no_save, add_timeframe_argument

EXCHANGE = "okx"
TIMEFRAME = "1d"
LABEL_WIDTH = 22
TOP_ROWS = 10
GAIN_COLUMNS = (
    ("sharpe_gain", "夏普增量"),
    ("annual_gain", "复利增量"),
    ("vol_gain", "波动降幅"),
    ("dd_gain", "回撤改善"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="波动率目标仓位统一测试台")
    parser.add_argument("--symbol", default="BTC/USDT", help="标的，默认 BTC/USDT")
    add_timeframe_argument(parser, default=TIMEFRAME)
    add_no_save(parser)
    return parser.parse_args()


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """只读本地 parquet 缓存，不联网。"""
    return pd.read_parquet(crypto_bar_path(EXCHANGE, symbol, timeframe))


def _print_run(label: str, run: VolTargetRun) -> None:
    print(
        f"{label:<{LABEL_WIDTH}}{run.avg_exposure:>10.3f}"
        f"{run.stats.annual_compounded:>12.2%}{run.stats.sharpe:>10.3f}"
        f"{run.stats.volatility:>10.2%}{run.stats.max_drawdown:>11.2%}"
        f"{run.turnover:>10.4f}{run.cost:>10.0f}"
    )


def print_universe(bars: pd.DataFrame, baseline: VolTargetRun, specs: int) -> None:
    """面板口径与满仓基准：任何改进都必须先打败这一行。"""
    print("=== 面板口径 ===")
    first = cast(pd.Timestamp, bars.index[0]).date()
    last = cast(pd.Timestamp, bars.index[-1]).date()
    print(f"标的 {bars.index.size} 根 {first} → {last}")
    print(f"跨度 {bars.index.size / ANNUAL_DAYS:.2f} 年 · 参数组合 {specs} 个")
    print(
        f"\n{'':<{LABEL_WIDTH}}{'平均暴露':>10}{'复利':>12}{'夏普':>10}"
        f"{'波动':>10}{'回撤':>11}{'换手':>10}{'成本':>10}"
    )
    _print_run(BASELINE_LABEL, baseline)


def print_mechanism(study: VolTargetStudy) -> None:
    """两个前提：波动率可预测、且高波动期未来收益更低。前提不成立就不必往下看。"""
    frame = study.mechanism
    print("\n=== 机制检验：波动率目标成立的两个前提 ===")
    autocorr = frame[frame["check"].str.startswith("autocorr")]
    print("-- 前提 1：波动率可预测（已实现波动率的自相关）--")
    for _, row in autocorr.iterrows():
        print(f"{row['check']:<20}{row['value']:>10.4f}")
    forward = frame[frame["check"].str.startswith("fwd")].copy()
    if forward.empty:
        return
    checks = cast(pd.Series, forward["check"])
    parts = checks.str.extract(r"fwd(\d+)_q(\d+)")
    forward["horizon"] = parts[0].astype(int)
    forward["quantile"] = parts[1].astype(int)
    table = forward.pivot(index="quantile", columns="horizon", values="value")
    print("\n-- 前提 2：高波动期未来收益更低（按波动率分档，Q1 最低）--")
    print(f"{'档':<8}" + "".join(f"{f'fwd{h}':>12}" for h in table.columns))
    for position, (_, row) in enumerate(table.iterrows()):
        cells = "".join(f"{row[h]:>12.4f}" for h in table.columns)
        print(f"Q{position + 1:<7}{cells}")


def print_grid(study: VolTargetStudy) -> None:
    """全网格分布：只报最优参数是挑选，分布才说明效应是否系统性。"""
    grid = study.grid
    print("\n=== 全网格：相对「等平均暴露固定仓位」的增量 ===")
    print(
        f"{'指标':<{LABEL_WIDTH}}{'最小':>12}{'中位':>12}{'最大':>12}{'为正占比':>12}"
    )
    for column, title in GAIN_COLUMNS:
        series = grid[column]
        print(
            f"{title:<{LABEL_WIDTH}}{series.min():>12.4f}{series.median():>12.4f}"
            f"{series.max():>12.4f}{float((series > 0).mean()):>12.2%}"
        )
    print(f"\n-- 夏普增量前 {TOP_ROWS} 名 --")
    print(
        f"{'标签':<{LABEL_WIDTH}}{'平均暴露':>10}{'复利':>11}{'夏普':>10}"
        f"{'对照夏普':>10}{'夏普增量':>10}"
    )
    for _, row in grid.nlargest(TOP_ROWS, "sharpe_gain").iterrows():
        print(
            f"{row['label']:<{LABEL_WIDTH}}{row['avg_exposure']:>10.3f}"
            f"{row['annual']:>11.2%}{row['sharpe']:>10.3f}"
            f"{row['ctrl_sharpe']:>10.3f}{row['sharpe_gain']:>10.4f}"
        )
    print("\n-- 四个维度各自的分组均值（看结论依赖哪个参数）--")
    for column in ("target_vol", "lookback", "rebalance_every", "max_exposure"):
        grouped = grid.groupby(column)["sharpe_gain"].mean()
        cells = "".join(f"{key}={value:+.4f}  " for key, value in grouped.items())
        print(f"{column:<{LABEL_WIDTH}}{cells}")


def print_segments(study: VolTargetStudy) -> None:
    """四段一致性：单段为正可能是运气，四段全正才说明是机制。

    按调仓频率分开看：rb30（月调仓）的增量方差是 rb1/rb7 的四倍，混在一起会把
    噪声当信号。
    """
    print("\n=== 分段稳定性（全样本四等分，看夏普增量是否每段都为正）===")
    # stub 把布尔索引的返回标成 Series | DataFrame，运行时必为 DataFrame
    stable = cast(pd.DataFrame, study.segments[study.segments["rebalance_every"] <= 7])
    noisy = cast(pd.DataFrame, study.segments[study.segments["rebalance_every"] == 30])
    for title, subset in (("每根/每周调仓 rb1/rb7", stable), ("月调仓 rb30", noisy)):
        print(f"\n-- {title} --")
        _print_segment_block(subset)


def _print_segment_block(segments: pd.DataFrame) -> None:
    """打印一组分段统计。"""
    edges = segments.groupby("segment").agg(
        start=("start", "first"), stop=("stop", "last")
    )
    summary = segments.groupby("segment")["sharpe_gain"].agg(
        ["min", "median", "max", lambda s: float((s > 0).mean())]
    )
    summary.columns = ["最小", "中位", "最大", "为正占比"]
    print(
        f"{'段':<6}{'起':<12}{'止':<12}" + "".join(f"{c:>12}" for c in summary.columns)
    )
    for position, (segment, row) in enumerate(summary.iterrows()):
        window = edges.loc[segment]
        cells = "".join(f"{row[c]:>12.4f}" for c in summary.columns)
        start = cast(pd.Timestamp, window["start"]).date()
        stop = cast(pd.Timestamp, window["stop"]).date()
        print(f"{position + 1:<6}{start!s:<12}{stop!s:<12}{cells}")


def print_bootstrap(study: VolTargetStudy) -> None:
    """配对块自举：夏普增量的置信区间。区间含 0 就不能当效应报。"""
    print("\n=== 配对块自举（块长 30 天，2000 次）===")
    print(
        f"{'标签':<{LABEL_WIDTH}}{'实测增量':>10}{'自举均值':>10}"
        f"{'95% 下界':>11}{'95% 上界':>11}{'为正占比':>10}"
    )
    for _, row in study.bootstrap.iterrows():
        print(
            f"{row['label']:<{LABEL_WIDTH}}{row['observed']:>10.4f}"
            f"{row['boot_mean']:>10.4f}{row['ci_low']:>11.4f}"
            f"{row['ci_high']:>11.4f}{row['p_positive']:>10.2%}"
        )


def print_cost(study: VolTargetStudy) -> None:
    """换手成本：波动率目标每根都在动仓位，这是它相对满仓多出来的成本项。"""
    print(f"\n=== 成本敏感性（{DEFAULT_COST_SPEC.label()}）===")
    print(
        f"{'单边费率':>10}{'复利':>12}{'夏普':>10}"
        f"{'对照复利':>12}{'对照夏普':>10}{'夏普增量':>10}{'成本':>10}"
    )
    for _, row in study.cost.iterrows():
        print(
            f"{row['fee_one_way']:>10.4%}{row['annual']:>12.2%}"
            f"{row['sharpe']:>10.3f}{row['ctrl_annual']:>12.2%}"
            f"{row['ctrl_sharpe']:>10.3f}{row['sharpe_gain']:>10.4f}"
            f"{row['cost']:>10.0f}"
        )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    bars = load_bars(args.symbol, args.timeframe)
    specs = build_grid()
    study = build_study(bars, specs)

    print_universe(bars, run_baseline(bars), len(specs))
    print_mechanism(study)
    print_grid(study)
    print_segments(study)
    print_bootstrap(study)
    print_cost(study)

    if not args.no_save:
        print("\n落盘：")
        for path in save_outputs(study):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
