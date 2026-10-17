"""
scripts/entropy_study.py
第 18 章实证：熵特征是不是只是 vol_30 的代理？

熵是「可预测性」的信息论度量：低熵 = 序列有结构，高熵 = 接近白噪声。本节的证伪问题是——
若熵因子与既有因子（尤其 vol_30）的截面秩相关很高，说明熵族没带来新信息、本仓的量价面板
已被现有 7 因子吃完；相关低才值得看 IC 与分位组合。

三组读数的分工：

- `shannon_bits` 是**分散度对照组**（固定宽度分箱，随波动单调上升）：它与 vol_30 高度相关是
  **预期的管道自检**，不是发现
- `ordinal_bits`（排序模式，对单调变换不变）与 `lz/konto`（二值化的方向序列）测的是**时序结构**，
  与波动无关才是它们唯一的价值
- 序列类估计量一律扣同窗置换基线（`*_excess`）：二者在小样本下有已知偏置（LZ 短语数是整数，
  逐个短语的步长是 log2(窗口)/窗口）

口径：窗口默认 250 根（n=100 时 LZ 熵率只能按 0.066 跳变，粒度太粗）；只用历史窗口
[t-window+1, t]；成本同时报 10bp（仓库假设）与 5.4bp（Ch 19 实测）。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.core.costs import PERP_FEES
from src.research.factors.evaluation import MIN_NAMES_PER_PERIOD, ic_stats
from src.research.factors.library import DIRECTION_LONG, FACTOR_LIBRARY, FactorSpec
from src.research.factors.panel import Panel, load_panel, tradable_mask
from src.research.factors.portfolio import (
    PortfolioOptions,
    break_even_cost,
    portfolio_stats,
)
from src.research.sampling.entropy import (
    EntropyOptions,
    entropy_panel,
    entropy_panel_name,
    load_panel_returns,
)
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

RESULTS_DIR: Path = Path("data/results/breaks")
DEFAULT_WINDOW: int = 250
DEFAULT_SURROGATES: int = 2
QUANTILES: int = 5
DEFAULT_COST_BP: float = PERP_FEES.taker_net() * 1e4
MEASURED_COST_BP: float = 5.4
MIN_PAIRS: int = 100
# 秩相关读法阈值：|ρ| ≥ 代理上限即「与既有因子是同一件事」，≤ 新信息下限才算另一族
PROXY_LEVEL: float = 0.7
NOVEL_LEVEL: float = 0.3

CONTROLS: tuple[tuple[str, str], ...] = (
    ("vol_30", "vol_30"),
    ("liq_30", "liq_30"),
    ("mom_30", "mom_30"),
    ("rev_3", "rev_3"),
)
# (因子键, 显示名, 长表列名)：全部按「值大 = 更随机」摆，价差为正表示高熵跑赢
ENTROPY_FACTORS: tuple[tuple[str, str, str], ...] = (
    ("ent_shannon", "香农熵（分散度对照）", "shannon_bits"),
    ("ent_ordinal", "顺序熵", "ordinal_bits"),
    ("ent_lz", "Lempel-Ziv 熵率", "lz_bits"),
    ("ent_lz_excess", "LZ 熵率−置换基线", "lz_excess"),
    ("ent_konto", "Kontoyiannis 熵率", "konto_bits"),
    ("ent_konto_excess", "Konto 熵率−置换基线", "konto_excess"),
)
COLUMNS: tuple[str, ...] = tuple(entry[2] for entry in ENTROPY_FACTORS)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Ch 18 熵特征实证")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="滚窗根数")
    parser.add_argument(
        "--surrogates", type=int, default=DEFAULT_SURROGATES, help="置换对照次数"
    )
    parser.add_argument("--horizon", type=int, default=1, help="组合调仓间隔（根）")
    parser.add_argument(
        "--limit-symbols", type=int, default=0, help="只跑前 N 只（冒烟）"
    )
    parser.add_argument("--refresh", action="store_true", help="重建熵面板缓存")
    add_no_save(parser)
    return parser.parse_args()


def entropy_table(
    market: str, timeframe: str, options: EntropyOptions, *, limit: int, refresh: bool
) -> pd.DataFrame:
    """读或重建熵长表（全池约数分钟：逐标的滚窗，逐窗算四个估计量加置换对照）。"""
    path = RESULTS_DIR / entropy_panel_name(market, timeframe, options)
    if path.exists() and not refresh and limit <= 0:
        return pd.read_parquet(path)
    panel = load_panel(timeframe, market=market)
    returns = load_panel_returns(panel)
    if limit > 0:
        returns = returns.iloc[:, :limit]
    table = entropy_panel(returns, options)
    if not table.empty and limit <= 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    return table


def wide(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """长表 → 宽表（时间 × 标的）。"""
    return cast(
        pd.DataFrame, table.pivot(index="date", columns="symbol", values=column)
    ).sort_index()


def pooled_rank_corr(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """池化秩相关（按 date/symbol 对齐；样本不足返回 nan）。"""
    joined = pd.DataFrame(
        {
            "x": left.stack(future_stack=True),
            "y": right.stack(future_stack=True),
        }
    ).dropna()
    if joined.shape[0] < MIN_PAIRS:
        return np.nan
    x = cast(pd.Series, joined["x"]).rank()
    return float(x.corr(cast(pd.Series, joined["y"]).rank(), method="spearman"))


def report_distributions(
    table: pd.DataFrame, panel: Panel, mask: pd.DataFrame, window: int
) -> None:
    """分布、缺失与**每期不同取值数**：截面取值太少就无法排序，IC 会被整片跳过。"""
    print(f"\n### 1. 各估计量的分布（单位 bit/根；窗口 {window} 根）")
    distinct = {
        column: float(
            wide(table, column)
            .reindex(panel.index)
            .where(mask)
            .nunique(axis=1)
            .median()
        )
        for _, _, column in ENTROPY_FACTORS
    }
    frame = pd.DataFrame(
        {
            "中位": table[list(COLUMNS)].median(numeric_only=True),
            "5%": table[list(COLUMNS)].quantile(0.05),
            "95%": table[list(COLUMNS)].quantile(0.95),
            "缺失": table[list(COLUMNS)].isna().mean(),
            "每期不同值": pd.Series(distinct),
        }
    )
    print(frame.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(
        f"标的日 {len(table):,}｜LZ 的步长 = log2({window})/{window} = {np.log2(window) / window:.4f}"
        f"｜每期不同值 < {MIN_NAMES_PER_PERIOD} 的因子无法排序"
    )


def report_proxy(table: pd.DataFrame, panel: Panel, mask: pd.DataFrame) -> pd.DataFrame:
    """证伪主问题：熵因子与既有因子的池化秩相关。"""
    print("\n### 2. 与既有因子的秩相关（证伪主问题）")
    rows: list[dict[str, float | str]] = []
    for _, label, column in ENTROPY_FACTORS:
        values = wide(table, column).reindex(panel.index).where(mask)
        row: dict[str, float | str] = {"因子": label}
        for control, control_label in CONTROLS:
            row[control_label] = pooled_rank_corr(
                values, FACTOR_LIBRARY[control].values(panel).where(mask)
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("因子")
    print(frame.to_string(float_format=lambda v: f"{v:+,.3f}"))
    print(
        f"读法：|ρ| ≥ {PROXY_LEVEL} = 与既有因子同一件事；≤ {NOVEL_LEVEL} = 另一族信息"
        f"（香农熵与 vol_30 高相关是预期的管道自检）"
    )
    return frame


def frame_compute(frame: pd.DataFrame, warmup: int) -> Callable[[Panel], pd.DataFrame]:
    """把算好的宽表包成 FactorSpec 要的 compute：对齐面板全索引（暖机期为 NaN）。

    熵表从第 warmup 根起才有值（要 window 根历史），而 portfolio_stats 的网格是
    panel.index[warmup::h]、逐点 .loc[date]——不给全索引就直接 KeyError。
    """

    def compute(panel: Panel) -> pd.DataFrame:
        return frame.reindex(panel.index)

    return compute


def entropy_specs(table: pd.DataFrame, warmup: int) -> list[FactorSpec]:
    """把熵长表包成因子（值大 = 更随机，价差为正即「高熵跑赢」）。"""
    return [
        FactorSpec(
            key=key,
            label=label,
            family="entropy",
            direction=DIRECTION_LONG,
            warmup=warmup,
            compute=frame_compute(wide(table, column), warmup),
        )
        for key, label, column in ENTROPY_FACTORS
    ]


def report_information(
    panel: Panel, mask: pd.DataFrame, table: pd.DataFrame, *, warmup: int
) -> None:
    """IC：熵能不能预测收益（h=1 与 h=7；显著性只看非重叠子样本）。"""
    print("\n### 3. 秩 IC（点估计用重叠样本，t 值只认非重叠子样本）")
    rows: list[dict[str, float | str]] = []
    for spec in entropy_specs(table, warmup):
        row: dict[str, float | str] = {"因子": spec.label}
        for horizon in (1, 7):
            stats = ic_stats(spec, panel, mask, horizon)
            row[f"IC h={horizon}"] = stats.mean
            row[f"t h={horizon}"] = stats.t_stat
            row[f"n h={horizon}"] = float(stats.independent)
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("因子")
    print(frame.to_string(float_format=lambda v: f"{v:+,.4f}"))


def report_portfolios(
    panel: Panel, mask: pd.DataFrame, table: pd.DataFrame, horizon: int, *, warmup: int
) -> pd.DataFrame:
    """分位组合：毛收益、10bp 与实测 5.4bp 下的净收益、平衡成本。"""
    print(f"\n### 4. 分位组合（{QUANTILES} 分位，h={horizon}）")
    rows: list[dict[str, float | str]] = []
    # 对照组的网格起点与熵因子不一定一致（它自带 warmup=30，实测期数多 70~320 期），
    # 故这一行只当方向参考，不作为「与 vol 比大小」的定量依据
    for spec in [*entropy_specs(table, warmup), FACTOR_LIBRARY["vol_30"]]:
        stats = portfolio_stats(
            spec, panel, mask, PortfolioOptions(horizon=horizon, quantiles=QUANTILES)
        )
        if stats.periods == 0:
            continue
        gross = stats.gross_spread
        rows.append(
            {
                "因子": spec.label,
                "期数": float(stats.periods),
                "毛(bp/期)": gross * 1e4,
                "净@10bp": (gross - stats.turnover * 4 * DEFAULT_COST_BP * 1e-4) * 1e4,
                "净@5.4bp": (gross - stats.turnover * 4 * MEASURED_COST_BP * 1e-4)
                * 1e4,
                "换手": stats.turnover,
                "平衡成本(bp)": break_even_cost(gross, stats.turnover) * 1e4,
            }
        )
    frame = pd.DataFrame(rows).set_index("因子")
    print(frame.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("读法：价差为正 = 高熵跑赢（低熵并不更可赚）；反向下注取负号但成本照付")
    return frame


def main() -> None:
    """入口：分布 → 与既有因子的秩相关 → IC → 分位组合。"""
    args = parse_args()
    options = EntropyOptions(window=args.window, surrogates=args.surrogates)
    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel)
    table = entropy_table(
        args.market,
        args.timeframe,
        options,
        limit=args.limit_symbols,
        refresh=args.refresh,
    )
    print(
        f"\n=== Ch 18 熵特征（{args.market} {args.timeframe}，窗口 {args.window} 根，"
        f"置换对照 {args.surrogates} 次）==="
    )
    print(f"标的日 {len(table):,}｜可交易掩码 {int(mask.to_numpy().sum()):,}")
    report_distributions(table, panel, mask, args.window)
    report_proxy(table, panel, mask)
    report_information(panel, mask, table, warmup=args.window)
    report_portfolios(panel, mask, table, args.horizon, warmup=args.window)
    if not args.no_save and args.limit_symbols <= 0:
        path = RESULTS_DIR / entropy_panel_name(
            args.market, args.timeframe, options, prefix="entropy_study"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        print(f"\n读数落盘 {path}")


if __name__ == "__main__":
    main()
