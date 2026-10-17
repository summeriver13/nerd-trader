"""
scripts/microstructure_study.py
第 19 章实证：把「成本」从假设的数字换成实测的数字。

本仓所有策略的裁决都被成本卡住，而成本一直是常数（5bp 手续费 + 5bp **假设**的滑点）。
本节用 1 分钟棒实测价差与冲击，回答：实测单边成本 vs 假设的 10bp（分流动性档，并报
估计量失效比例）、这些量与 liq_30 是不是同一件事、流动性溢价收不收得上、VPIN 到底
预测波动还是收益。口径与限制见 README 第十九节。

"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.library import DIRECTION_SHORT, FACTOR_LIBRARY, FactorSpec
from src.research.factors.panel import Panel, load_panel, tradable_mask
from src.research.factors.portfolio import (
    PortfolioOptions,
    break_even_cost,
    forward_returns,
    portfolio_stats,
)
from src.research.liquidity.microstructure import load_microstructure
from src.utils.study_cli import add_no_save

RESULTS_DIR: Path = Path("data/results/breaks")
TRAILING_WINDOW: int = 30
QUANTILES: int = 5
DEFAULT_COST_BP: float = PERP_FEES.taker_net() * 1e4
TAKER_FEE_BP: float = PERP_FEES.taker * 1e4
LEVELS: tuple[str, ...] = ("小", "中", "大")
MEASURES: tuple[tuple[str, str, str], ...] = (
    ("micro_cs", "价差（CS）", "cs_spread_bp"),
    ("micro_kyle", "价格冲击 λ", "kyle_lambda_bp"),
    ("micro_amihud", "Amihud 非流动性", "amihud_bp"),
    ("micro_vpin", "VPIN 订单流毒性", "vpin"),
)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Ch 19 微观结构特征实证")
    parser.add_argument(
        "--market", default="swap", help="市场（1m 只有 swap 有 30 个标的）"
    )
    parser.add_argument("--horizon", type=int, default=1, help="组合调仓间隔（根）")
    parser.add_argument("--refresh", action="store_true", help="重建 1m 聚合缓存")
    add_no_save(parser)
    return parser.parse_args()


def micro_table(market: str, *, refresh: bool) -> pd.DataFrame:
    """读或重建日频微观结构长表（重建约 2~4 分钟：30 个标的 × 约 3000 天）。"""
    path = RESULTS_DIR / f"microstructure_{market}_1m_daily.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    table = load_microstructure(market=market)
    if not table.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    return table


def wide(table: pd.DataFrame, column: str, *, window: int = 0) -> pd.DataFrame:
    """长表 → 宽表（时间 × 标的）；window>0 时取拖后中位。

    单日估计噪声太大，直接当因子不可用，故因子一律走 30 日中位（只吃历史）。
    """
    frame = cast(
        pd.DataFrame, table.pivot(index="date", columns="symbol", values=column)
    ).sort_index()
    if window <= 0:
        return frame
    return cast(
        pd.DataFrame, frame.rolling(window, min_periods=TRAILING_WINDOW // 2).median()
    )


def rank_corr(left: pd.Series, right: pd.Series) -> float:
    """两序列的秩相关（先在公共标的上对齐；样本不足返回 nan）。"""
    joined = pd.concat([left.rename("x"), right.rename("y")], axis=1).dropna()
    if joined.shape[0] < 10:
        return np.nan
    x = cast(pd.Series, joined["x"]).rank()
    return float(x.corr(cast(pd.Series, joined["y"]).rank(), method="spearman"))


def report_distributions(table: pd.DataFrame) -> None:
    """各估计量的分布与失效比例（失效必须报，否则「零成本」会被当真）。"""
    print("\n### 1. 各估计量的分布（日频，全部标的）")
    stats = pd.DataFrame(
        {
            "中位": table.median(numeric_only=True),
            "缺失": table.isna().mean(numeric_only=True),
            "零值占比": (table == 0.0).mean(numeric_only=True),
        }
    )
    print(
        stats.loc[[column for _, _, column in MEASURES]].to_string(
            float_format=lambda v: f"{v:,.4f}"
        )
    )
    print(
        "读法：CS 是 1 分钟尺度的往返价差（bp），α<0 取 0；Roll 无解时缺失，两者都 ≠ 零成本"
    )


def cost_table(table: pd.DataFrame, liquidity: pd.Series) -> pd.DataFrame:
    """按成交额分档：单边成本 = 半价差 + 手续费，另报 CS 失效占比与 λ 的规模含义。"""
    levels = pd.Series(
        pd.qcut(liquidity.rank(method="first"), 3, labels=list(LEVELS)),
        index=liquidity.index,
    )
    rows: list[dict[str, float | str]] = []
    for level in LEVELS:
        symbols = [name for name in liquidity.index if levels.loc[name] == level]
        block = table.loc[table["symbol"].isin(symbols)]
        spread = float(block["cs_spread_bp"].median())
        lam = float(block["kyle_lambda_bp"].median())
        row = {
            "档": level,
            "标的数": float(len(symbols)),
            "日成交额($M)": float(liquidity.loc[symbols].median() / 1e6),
        }
        row["往返价差(bp)"] = spread
        row["CS 失效占比"] = float((block["cs_spread_bp"] == 0.0).mean())
        row["单边实测(bp)"] = spread / 2.0 + TAKER_FEE_BP
        row["假设单边(bp)"] = DEFAULT_COST_BP
        row["λ冲击1M(bp)"] = lam
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("档")
    print("\n### 2. 按流动性分档的实测单边成本 vs 假设的 10bp")
    print(frame.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(
        "读法：单边实测 = 半价差 + 5bp 手续费；CS 失效占比高 → 该档价差是估计量给的 0，不可读成零成本"
    )
    return frame


def _from_frame(frame: pd.DataFrame) -> Callable[[Panel], pd.DataFrame]:
    """把算好的宽表包成 FactorSpec 要的 compute（与面板无关：值已就绪）。"""

    def compute(_panel: Panel) -> pd.DataFrame:
        return frame

    return compute


def measure_specs(table: pd.DataFrame) -> list[FactorSpec]:
    """四个微观结构因子（越小越看多：便宜 / 低冲击 / 低毒性）。"""
    return [
        FactorSpec(
            key=key,
            label=label,
            family="micro",
            direction=DIRECTION_SHORT,
            warmup=TRAILING_WINDOW,
            compute=_from_frame(wide(table, column, window=TRAILING_WINDOW)),
        )
        for key, label, column in MEASURES
    ]


def leg_costs(
    spec: FactorSpec,
    panel: Panel,
    mask: pd.DataFrame,
    options: PortfolioOptions,
    cost: pd.Series,
) -> tuple[float, float]:
    """多腿与空腿各自的平均单边成本（bp）：每期按当期成员取均值，再对期取均值。"""
    values = spec.values(panel).where(mask)
    grid = panel.index[spec.warmup :: options.horizon]
    long_costs: list[float] = []
    short_costs: list[float] = []
    for date in grid[grid >= panel.index[spec.warmup]]:
        row = cast(pd.Series, values.loc[date]).dropna()
        if row.size < options.quantiles * 3:
            continue
        bucket = pd.qcut(row.rank(method="first"), options.quantiles, labels=False)
        names = row.index
        long_costs.append(
            float(cost.reindex(names[bucket == options.quantiles - 1]).mean())
        )
        short_costs.append(float(cost.reindex(names[bucket == 0]).mean()))
    if not long_costs:
        return np.nan, np.nan
    return float(np.mean(long_costs)), float(np.mean(short_costs))


def report_portfolios(
    table: pd.DataFrame, panel: Panel, mask: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """分位组合：毛收益、10bp 下的净收益、按分腿实测成本的净收益。"""
    print(f"\n### 3. 分位组合（{QUANTILES} 分位，h={horizon}）：流动性溢价收得上吗")
    options = PortfolioOptions(horizon=horizon, quantiles=QUANTILES)
    spread = table.groupby("symbol")["cs_spread_bp"].median()
    cost = cast(pd.Series, spread / 2.0 + TAKER_FEE_BP)
    rows: list[dict[str, float | str]] = []
    for spec in [*measure_specs(table), FACTOR_LIBRARY["liq_30"]]:
        stats = portfolio_stats(spec, panel, mask, options)
        if stats.periods == 0:
            continue
        long_cost, short_cost = leg_costs(spec, panel, mask, options, cost)
        # 成本已是 bp、换手是每期被替换的比例：两项相乘即每期成本，不要再乘 1e4
        measured = 2.0 * (
            stats.long_turnover * long_cost + stats.short_turnover * short_cost
        )
        rows.append(
            {
                "因子": spec.label,
                "期数": float(stats.periods),
                "毛(bp/期)": stats.gross_spread * 1e4,
                "净@10bp": stats.net_spread * 1e4,
                "平衡成本(bp)": break_even_cost(stats.gross_spread, stats.turnover)
                * 1e4,
                "实测成本合计": measured,
                "净@实测": stats.gross_spread * 1e4 - measured,
            }
        )
    frame = pd.DataFrame(rows).set_index("因子")
    print(frame.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("读法：成本列单位 bp/边；「实测成本合计」已折算到每期，只能行内比较")
    return frame


def report_vpin(table: pd.DataFrame, panel: Panel, mask: pd.DataFrame) -> None:
    """VPIN 的用途：预测次日波动（风险），而不是收益（alpha）。"""
    vpin_frame = wide(table, "vpin", window=TRAILING_WINDOW)
    forward = forward_returns(panel, 1).where(mask)
    size_ic: list[float] = []
    signed_ic: list[float] = []
    for date in vpin_frame.index:
        if date not in forward.index:
            continue
        row = cast(pd.Series, vpin_frame.loc[date]).dropna()
        nxt = cast(pd.Series, forward.loc[date]).reindex(row.index).dropna()
        if nxt.size < 10:
            continue
        size_ic.append(rank_corr(row, nxt.abs()))
        signed_ic.append(rank_corr(row, nxt))
    print("\n### 4. VPIN 预测的是波动，不是收益")
    print(
        f"对次日 |收益| 的截面秩相关：均值 {np.nanmean(size_ic):+.4f}（{len(size_ic)} 天）"
    )
    print(
        f"对次日**带符号**收益的截面秩相关：均值 {np.nanmean(signed_ic):+.4f}（≈0 = 没有 alpha）"
    )


def main() -> None:
    """入口：实测成本 → 与既有流动性的关系 → 分位组合 → VPIN 用途。"""
    args = parse_args()
    table = micro_table(args.market, refresh=args.refresh)
    if table.empty:
        raise SystemExit("没有 1m 聚合结果：确认 data/storage/crypto/okx/swap/1m 存在")
    panel = load_panel("1d", market=args.market)
    mask = tradable_mask(panel)
    symbols = sorted(table["symbol"].unique().tolist())
    print(f"1m 覆盖 {len(symbols)} 个标的，日频池 {len(panel.symbols)} 个")
    print(
        f"区间：{table['date'].min()} → {table['date'].max()}（{table.shape[0]} 标的日）"
    )
    report_distributions(table)

    liquidity = cast(
        pd.Series, panel.notional.reindex(columns=symbols).median()
    ).dropna()
    cost_frame = cost_table(table, liquidity)

    print("\n### 2b. 与既有流动性的关系（标的层面中位，Spearman）")
    log_liquidity = pd.Series(np.log(liquidity.to_numpy()), index=liquidity.index)
    spread_med = cast(pd.Series, table.groupby("symbol")["cs_spread_bp"].median())
    lambda_med = cast(pd.Series, table.groupby("symbol")["kyle_lambda_bp"].median())
    print(
        f"价差 vs 成交额：{rank_corr(spread_med, log_liquidity):+.3f}｜"
        f"λ vs 成交额：{rank_corr(lambda_med, log_liquidity):+.3f}"
        "（负 = 越不流动越贵；绝对值接近 1 说明只是 liq_30 的代理）"
    )
    portfolio_frame = report_portfolios(table, panel, mask, args.horizon)
    report_vpin(table, panel, mask)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"microstructure_study_{args.market}.parquet"
        frames = [
            cost_frame.assign(表="成本分档"),
            portfolio_frame.assign(表="分位组合").reset_index(),
        ]
        pd.concat(frames, ignore_index=True).to_parquet(path)
        print(f"\n读数落盘：{path}")


if __name__ == "__main__":
    main()
