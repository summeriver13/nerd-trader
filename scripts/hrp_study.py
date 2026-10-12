"""
scripts/hrp_study.py
第 16 章（HRP）实证：把「配权方法」当成变量——同一批策略，换权重方案，看记录读数。

用法：
    uv run python scripts/hrp_study.py
    uv run python scripts/hrp_study.py --market spot --horizon 1 --no-save

设计（为什么这么设计）：
- **组合的对象是「策略」而不是「资产」**：本仓库资产层的结论已是「分位桶价差多为负」，
  再在资产层换配权救不了；真正没验证过的是「把若干弱策略合成一个组合」。
- **协方差只用训练段、样本外只吃收益**：在测试段上重估协方差再配权就用了未来信息。
- **比较三个方案**：等权（基线）、逆方差（原文 getIVP，HRP 的退化情形）、HRP（第 16 章）。
  原文的主张是 HRP 在**样本外方差**上更稳，故两个方差都打印，不只比夏普。
- 所有策略取同一期限（默认 h=1），保证落在同一日网格；否则各策略的非重叠期次相位
  不同，拼起来的日子对不上。

输出：逐方案一行（夏普 / PSR / MinTRL / 最大回撤 / 训练段方差 / 样本外方差）+ 权重。
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.allocation import hrp_weights, inverse_variance_weights
from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.panel import Panel, UniverseRule, load_panel, tradable_mask
from src.research.factors.portfolio import PortfolioOptions, portfolio_stats
from src.research.track_record import TrackRecord, track_record

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "breaks"
# 参与组合的策略（都是库里已有的因子，期限由 --horizon 统一给）
DEFAULT_STRATEGIES: Final[tuple[str, ...]] = (
    "vol_30",
    "mom_7",
    "mom_30",
    "rev_3",
    "dth_60",
    "ffd_d0p4",
    "liq_30",
)
DEFAULT_SPLIT: Final[float] = 0.5
MIN_SERIES: Final[int] = 400
MIN_TEST_PERIODS: Final[int] = 60


@dataclass(frozen=True)
class SchemeResult:
    """一个配权方案在样本外的读数。"""

    scheme: str
    in_sample_variance: float
    out_of_sample_variance: float
    record: TrackRecord
    weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HRP 配权实证：策略层组合")
    parser.add_argument("--market", default="swap", help="市场层，默认 swap")
    parser.add_argument("--timeframe", default="1d", help="周期，默认 1d")
    parser.add_argument("--horizon", type=int, default=1, help="统一期限，默认 1")
    parser.add_argument("--split", type=float, default=DEFAULT_SPLIT, help="训练段占比")
    parser.add_argument("--strategies", default=None, help="逗号分隔的因子键")
    parser.add_argument("--no-save", action="store_true", help="不落盘，只打印")
    return parser.parse_args()


def build_spreads(
    keys: tuple[str, ...], panel: Panel, mask: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """逐策略跑分位组合，取逐期价差序列，拼成一张宽表。"""
    series: dict[str, pd.Series] = {}
    for key in keys:
        stats = portfolio_stats(
            FACTOR_LIBRARY[key], panel, mask, PortfolioOptions(horizon=horizon)
        )
        spread = stats.spread_returns.dropna()
        if spread.size >= MIN_SERIES:
            series[key] = spread
    return pd.DataFrame(series)


def evaluate(
    scheme: str, weights: np.ndarray, train: pd.DataFrame, test: pd.DataFrame
) -> SchemeResult:
    """一个配权方案：训练段方差、样本外方差，以及样本外记录体检。"""
    covariance = train.cov().to_numpy(dtype=float)
    out_of_sample = test.to_numpy(dtype=float) @ weights
    record = track_record(scheme, pd.Series(out_of_sample, index=test.index))
    return SchemeResult(
        scheme=scheme,
        in_sample_variance=float(weights @ covariance @ weights),
        out_of_sample_variance=float(np.var(out_of_sample, ddof=1)),
        record=record,
        weights=weights,
    )


def print_scheme(result: SchemeResult, columns: list[str]) -> None:
    record = result.record
    track = (
        "inf"
        if math.isinf(record.min_track_record)
        else f"{record.min_track_record:.0f}"
    )
    print(
        f"  {result.scheme:<6}夏普 {record.sharpe:+.3f}｜PSR {record.psr:.3f}｜"
        f"MinTRL {track}｜最大回撤 {record.max_drawdown:.2f}｜"
        f"训练方差 {result.in_sample_variance:.3e}｜样本外方差 {result.out_of_sample_variance:.3e}"
    )
    pairs = ", ".join(
        f"{name} {value:.3f}"
        for name, value in zip(columns, result.weights, strict=True)
    )
    print(f"         权重：{pairs}")


def save_results(results: list[SchemeResult], market: str, timeframe: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f"hrp_study_{market}_{timeframe}.parquet"
    frame = pd.DataFrame(
        [
            {
                "scheme": result.scheme,
                "in_sample_variance": result.in_sample_variance,
                "out_of_sample_variance": result.out_of_sample_variance,
                "observations": result.record.observations,
                "sharpe": result.record.sharpe,
                "psr": result.record.psr,
                "min_track_record": result.record.min_track_record,
                "max_drawdown": result.record.max_drawdown,
            }
            for result in results
        ]
    )
    frame.to_parquet(target)
    return target


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    keys = (
        tuple(item.strip() for item in args.strategies.split(","))
        if args.strategies
        else DEFAULT_STRATEGIES
    )
    rule = UniverseRule()
    panel = load_panel(args.timeframe, rule, market=args.market)
    mask = tradable_mask(panel, rule)
    frame = build_spreads(keys, panel, mask, args.horizon).dropna()
    split = int(frame.shape[0] * args.split)
    train, test = frame.iloc[:split], frame.iloc[split:]
    print(
        f"市场 {args.market} | 周期 {args.timeframe} | 期限 h={args.horizon} | "
        f"策略 {list(frame.columns)}\n训练 {train.shape[0]} 期 / 测试 {test.shape[0]} 期"
    )
    if train.shape[1] < 2 or test.shape[0] < MIN_TEST_PERIODS:
        print("样本不足：要么策略太少，要么训练/测试段太短。")
        return 1

    covariance = train.cov().to_numpy(dtype=float)
    count = train.shape[1]
    results = [
        evaluate("等权", np.full(count, 1.0 / count), train, test),
        evaluate("逆方差", inverse_variance_weights(covariance), train, test),
        evaluate("HRP", hrp_weights(train).to_numpy(dtype=float), train, test),
    ]
    for result in results:
        print_scheme(result, list(frame.columns))

    if not args.no_save:
        print(f"\n落盘：{save_results(results, args.market, args.timeframe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
