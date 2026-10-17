"""池尾成本的日频代理标定：用已覆盖的 30 只拟合「日频流动性 → 1m 实测价差」，再外推到全池。

问题：Ch 19 实测的单边成本 5.4bp 只来自 1m 覆盖的 30 个最流动标的，而 liq_30 与反转的
溢价恰恰发生在池尾。给全池补 1m 要 32 天（实测 335 根/秒），所以先试便宜的路：日频代理。

口径：
- 目标 = 每只标的在窗口内 **CS>0 的那些日子** 的 cs_spread_bp 中位数的一半（半价差，bp），
  手续费是常数 5bp 不加进模型，预测成本 = 预测半价差 + 手续费。
  CS=0 的日子是估计量失效（原文 α<0 钳 0），不是零成本，故用 CS>0 子集拟合、并把
  「CS 可估占比」作为列一起报出：占比低的标的预测值只是下界。
- 特征 = 窗口内 log10 中位日名义成交额、log10 中位 Amihud、日收益波动率（全是日频可得）。
- 诚实性检查 = 留一标的交叉验证（LOSO）：代理模型必须打败「用其余 29 只的均值」这个常数基线，
  否则结论就是「日频代理不含池尾成本信息」，不能拿它当答案。
- 预登记 = 池尾 12 只（正在补 1m 的那批）的预测值先落盘，等真值到手对账，而不是事后挑口径。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.panel import Panel, load_panel
from src.utils.study_cli import add_market_argument, add_timeframe_argument

TAKER_FEE_BP: float = PERP_FEES.taker * 1e4
DEFAULT_WINDOW_DAYS = 180
MIN_CS_DAYS = 30
DECILES = 10
PER_MILLION = 1e6
BP_PER_UNIT = 1e4
TAIL_COUNT = 12
# 三根反转（swap、h=1）的平衡成本：README 第二十一节，低于它才有净收益
BREAK_EVEN_BP = 4.69
# Ch 19 在 30 个流动标的上实测的单边成本，作为对照
ASSUMED_COST_BP = 5.4
# 预登记：正在补 1m 的池尾 12 只（按近一年中位名义成交额排最不流动的 12 个未覆盖标的）
TAIL_SYMBOLS = (
    "LQTY-USDT",
    "IOST-USDT",
    "BAND-USDT",
    "GMX-USDT",
    "GLM-USDT",
    "METIS-USDT",
    "CVX-USDT",
    "ETHW-USDT",
    "ZETA-USDT",
    "GAS-USDT",
    "QTUM-USDT",
    "ACH-USDT",
)
FEATURE_COLUMNS: tuple[str, ...] = ("log_notional", "log_amihud", "vol")
SIMPLE_COLUMNS: tuple[str, ...] = ("log_notional",)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="池尾成本的日频代理标定")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser, help_text="日频面板周期，默认 1d")
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="窗口天数"
    )
    parser.add_argument(
        "--micro-cache",
        default="data/results/breaks/microstructure_swap_1m_daily.parquet",
        help="1m 聚合缓存（含 cs_spread_bp）",
    )
    return parser.parse_args()


def _column_reduce(
    values: np.ndarray, minimum: int, reducer: Callable[[np.ndarray], float]
) -> np.ndarray:
    """按列统计：只喂有限值；有效值不足 minimum 个的列返回 NaN（不用 nanmedian，避开全 NaN 警告）。"""
    out = np.full(values.shape[1], np.nan, dtype=float)
    for index in range(values.shape[1]):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        if finite.size >= minimum:
            out[index] = float(reducer(finite))
    return out


def measured_costs(micro: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """每只标的的半价差（bp）：窗口内 CS>0 日子的中位数折半，附 CS 可估占比。"""
    stamps = pd.DatetimeIndex(micro["date"]).to_numpy()
    cutoff = stamps.max() - np.timedelta64(window_days, "D")
    recent = micro.loc[stamps >= cutoff]
    rows: list[dict[str, float | str]] = []
    for symbol, group in recent.groupby("symbol"):
        spread = group["cs_spread_bp"].to_numpy(dtype=float)
        positive = spread[spread > 0.0]
        estimable = positive.size >= MIN_CS_DAYS
        rows.append(
            {
                "symbol": str(symbol),
                "cs_share": float(positive.size / max(spread.size, 1)),
                "days": float(spread.size),
                "half_spread_bp": (
                    float(np.median(positive)) / 2.0 if estimable else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("symbol")
    frame["one_way_bp"] = frame["half_spread_bp"] + TAKER_FEE_BP
    return frame


def daily_features(panel: Panel, window_days: int) -> pd.DataFrame:
    """每只标的的日频代理：log10 中位名义成交额、log10 中位 Amihud（bp/$1M）、波动率。"""
    close = panel.close.tail(window_days)
    notional = panel.notional.tail(window_days)
    returns = close.pct_change()
    amihud = (returns.abs() / notional.replace(0.0, np.nan)) * PER_MILLION * BP_PER_UNIT
    amihud = amihud.replace([np.inf, -np.inf], np.nan)
    median_notional = _column_reduce(notional.to_numpy(dtype=float), 1, np.median)
    median_amihud = _column_reduce(amihud.to_numpy(dtype=float), 1, np.median)
    volatility = _column_reduce(
        returns.to_numpy(dtype=float), 2, lambda column: float(np.std(column, ddof=1))
    )
    return pd.DataFrame(
        {
            "log_notional": np.log10(np.clip(median_notional, 1.0, None)),
            "log_amihud": np.log10(np.clip(median_amihud, 1e-9, None)),
            "vol": volatility,
            "traded_days": notional.notna().to_numpy().sum(axis=0).astype(float),
        },
        index=pd.Index(close.columns, name="symbol"),
    )


def design(features: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """回归设计矩阵（首列为截距）。"""
    matrix = features[list(columns)].to_numpy(dtype=float)
    return np.column_stack([np.ones(matrix.shape[0]), matrix])


def fit_coefficients(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """最小二乘系数（对数空间）。"""
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return np.asarray(solution, dtype=float)


def loso_predictions(
    features: pd.DataFrame, target: pd.Series, columns: tuple[str, ...]
) -> np.ndarray:
    """留一标的交叉验证：每次用其余标的拟合，预测被留的那只（对数空间）。"""
    matrix = design(features, columns)
    values = target.to_numpy(dtype=float)
    out = np.empty(values.size, dtype=float)
    for index in range(values.size):
        keep = np.arange(values.size) != index
        coefficients = fit_coefficients(matrix[keep], values[keep])
        out[index] = float(matrix[index] @ coefficients)
    return out


def loso_constant(target: pd.Series) -> np.ndarray:
    """常数基线：预测「其余标的的均值」（同样留一，保证可比）。"""
    values = target.to_numpy(dtype=float)
    total = float(values.sum())
    return (total - values) / (values.size - 1)


def loso_table(joined: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """三种口径的留一误差对照（bp，对数预测先取指数）。"""
    rows: list[dict[str, object]] = []
    candidates: list[tuple[str, np.ndarray]] = [
        ("常数基线", loso_constant(target)),
        ("仅成交额", loso_predictions(joined, target, SIMPLE_COLUMNS)),
        ("成交额+Amihud+波动", loso_predictions(joined, target, FEATURE_COLUMNS)),
    ]
    actual = np.exp(target.to_numpy(dtype=float))
    for name, log_prediction in candidates:
        error = np.exp(log_prediction) - actual
        rows.append(
            {
                "口径": name,
                "中位绝对误差(bp)": float(np.median(np.abs(error))),
                "平均绝对误差(bp)": float(np.mean(np.abs(error))),
                "最大绝对误差(bp)": float(np.max(np.abs(error))),
            }
        )
    return pd.DataFrame(rows).set_index("口径")


def pool_predictions(
    features: pd.DataFrame, joined: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    """在覆盖标的上拟合、给全池出预测（含实测列，覆盖标的才有）。"""
    coefficients = fit_coefficients(
        design(joined, FEATURE_COLUMNS), target.to_numpy(dtype=float)
    )
    predicted_half = np.exp(design(features, FEATURE_COLUMNS) @ coefficients)
    out = features.copy()
    out["pred_half_spread_bp"] = predicted_half
    out["pred_one_way_bp"] = predicted_half + TAKER_FEE_BP
    return out.join(joined[["half_spread_bp", "one_way_bp", "cs_share"]], how="left")


def main() -> None:
    """跑标定并落盘预登记预测。"""
    args = parse_args()
    micro = pd.read_parquet(args.micro_cache)
    panel = load_panel(args.timeframe, market=args.market)
    measured = measured_costs(micro, args.window_days)
    features = daily_features(panel, args.window_days)

    covered = features.join(measured, how="inner").dropna(
        subset=[*FEATURE_COLUMNS, "half_spread_bp"]
    )
    target = pd.Series(
        np.log(covered["half_spread_bp"].to_numpy(dtype=float)), index=covered.index
    )
    print(f"覆盖标的可用 {covered.shape[0]} 只（1m 缓存里价差可估的）")
    print(
        f"窗口 {args.window_days} 天；实测单边成本 中位 {covered['one_way_bp'].median():.2f}bp、"
        f"区间 [{covered['one_way_bp'].min():.2f}, {covered['one_way_bp'].max():.2f}]；"
        f"CS 可估占比 中位 {covered['cs_share'].median():.3f}"
    )

    table = loso_table(covered, target)
    print("\n### 留一标的交叉验证（bp，越小越好）")
    print(table.round(3).to_string())

    predictions = pool_predictions(features, covered, target)
    print("\n### 全池预测（按中位成交额分档，10 档）")
    buckets = pd.qcut(
        pd.Series(predictions["log_notional"].to_numpy(), index=predictions.index),
        DECILES,
        labels=False,
    )
    summary = (
        predictions.assign(档=buckets)
        .groupby("档")
        .agg(
            标的数=("pred_one_way_bp", "size"),
            中位成交额=("log_notional", "median"),
            预测单边成本=("pred_one_way_bp", "median"),
            实测单边成本=("one_way_bp", "median"),
            实测只数=("one_way_bp", "count"),
        )
    )
    print(summary.round(3).to_string())

    tail = predictions.loc[[s for s in TAIL_SYMBOLS if s in predictions.index]]
    print(f"\n### 预登记池尾 {tail.shape[0]} 只的预测（1m 抓完对账）")
    print(
        tail[["log_notional", "pred_one_way_bp", "cs_share"]]
        .rename(
            columns={"log_notional": "log10成交额", "pred_one_way_bp": "预测单边(bp)"}
        )
        .round(3)
        .to_string()
    )
    below = float((predictions["pred_one_way_bp"] <= BREAK_EVEN_BP).mean())
    below_assumed = float((predictions["pred_one_way_bp"] <= ASSUMED_COST_BP).mean())
    print(
        f"\n池内预测成本 ≤ 反转平衡成本 {BREAK_EVEN_BP}bp 的占比 {below:.1%}；"
        f"≤ Ch 19 实测 {ASSUMED_COST_BP}bp 的占比 {below_assumed:.1%}"
    )
    ranked = features.sort_values("log_notional").index[:TAIL_COUNT]
    print(f"按成交额排最不流动的 {TAIL_COUNT} 只 = {[str(s) for s in ranked]}")
    print(f"预登记名单是否一致：{set(map(str, ranked)) == set(TAIL_SYMBOLS)}")

    predictions.to_parquet("data/results/breaks/cost_calibration_pool.parquet")
    print("落盘 data/results/breaks/cost_calibration_pool.parquet")


if __name__ == "__main__":
    main()
