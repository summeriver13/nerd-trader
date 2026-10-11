"""协整配对交易研究：主口径 + 随机对照 + 参数变体 + 多重检验校正。

配对是这套研究里第一个**真正市场中性**的结构（beta ≈ 0），所以它的结论不是
「能不能跑赢 BTC」，而是「扣掉 beta 之后还剩多少边际、这些边际扛不扛得住成本」。

用法：
    uv run python scripts/pairs_study.py                      # 现货面板（主口径）
    uv run python scripts/pairs_study.py --market swap         # 永续面板（含资金费）
    uv run python scripts/pairs_study.py --controls 10 --no-save

输出五张表（组合 vs 基准、单笔分布、成本敏感性、分段稳定性、随机对照）+ 变体表 +
DSR 校正，全部落盘到 `data/results/pairs/<market>/`（现货与永续是两套结论，分开存）。

只读本地缓存，不联网；面板口径与截面因子研究共用 factors.panel（池子规则只有一处定义）。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import cast

import pandas as pd

from src.analytics.metrics import bars_per_year, sharpe_ratio
from src.config.paths import RESULTS_ROOT
from src.research.carry_ranking import load_daily
from src.research.factors.panel import Panel, UniverseRule, load_panel, tradable_mask
from src.research.factors.portfolio import leg_stats
from src.research.overfitting import (
    deflated_sharpe,
    expected_max_sharpe,
    sharpe_estimator_variance,
)
from src.research.pairs.backtest import PairsOptions, PairsRun, run_pairs
from src.research.pairs.output import print_report, save_outputs
from src.research.pairs.report import build_report
from src.utils.console import pad

# 资金费代理源：OKX 只保留约 103 天，长历史只能用 Binance（跨源代理假设已由
# docs/carry/ 的同期校验背书）
FUNDING_EXCHANGE: str = "binance"
# 参数变体：多重检验的试验次数就来自这张表，漏记一个变体 = 门槛算低了
VARIANTS: dict[str, PairsOptions] = {
    "主口径": PairsOptions(),
    "β∈[0.5,2]": PairsOptions(
        selection=replace(PairsOptions().selection, min_beta=0.5, max_beta=2.0)
    ),
    "形成120/交易30": PairsOptions(
        selection=replace(PairsOptions().selection, formation_bars=120),
        trading_bars=30,
        step_bars=30,
    ),
    "形成240/交易90": PairsOptions(
        selection=replace(PairsOptions().selection, formation_bars=240),
        trading_bars=90,
        step_bars=90,
    ),
    "开1.5σ/平0.25σ": PairsOptions(entry_z=1.5, exit_z=0.25),
    "开2.5σ/平1.0σ": PairsOptions(entry_z=2.5, exit_z=1.0),
    "只取前3对": PairsOptions(selection=replace(PairsOptions().selection, top_pairs=3)),
    "半衰期≤15天": PairsOptions(
        selection=replace(PairsOptions().selection, max_half_life=15.0)
    ),
    "相关≥0.8": PairsOptions(selection=replace(PairsOptions().selection, min_corr=0.8)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="协整配对交易研究")
    parser.add_argument(
        "--market", default="spot", choices=("spot", "swap"), help="现货或永续面板"
    )
    parser.add_argument(
        "--controls", type=int, default=10, help="随机对照的重复次数（0 关闭）"
    )
    parser.add_argument("--no-save", action="store_true", help="只打印，不落盘")
    return parser.parse_args()


def load_funding(
    panel_symbols: pd.Index, market: str
) -> tuple[pd.DataFrame | None, str]:
    """永续面板才加载资金费；返回 (费率宽表, 报告用的口径说明)。"""
    if market != "swap":
        return None, "未计资金费（现货没有这一项；永续腿按研究假设处理）"
    directory = (
        RESULTS_ROOT.parent / "storage" / "crypto" / FUNDING_EXCHANGE / "funding" / "8h"
    )
    stems = [path.stem for path in sorted(directory.glob("*.parquet"))]
    available = [stem for stem in stems if stem in set(panel_symbols)]
    if not available:
        return None, f"未计资金费（{FUNDING_EXCHANGE} 缓存里没有池内标的）"
    table = load_daily(FUNDING_EXCHANGE, available)
    coverage = len(available) / max(len(panel_symbols), 1)
    return (
        table,
        f"{FUNDING_EXCHANGE} 日费率求和（覆盖池内 {coverage:.0%} 标的，缺失日按 0）",
    )


def stats_of(daily: pd.Series) -> dict[str, float]:
    """一条日收益序列的展示口径。"""
    clean = daily.dropna()
    if clean.size < 2:
        return {"年化": 0.0, "波动": 0.0, "夏普": 0.0, "回撤": 0.0}
    stats = leg_stats(clean, bars_per_year(clean.index))
    return {
        "年化": stats.annual_compounded,
        "波动": stats.volatility,
        "夏普": stats.sharpe,
        "回撤": stats.max_drawdown,
    }


def run_variants(
    panel: Panel, mask: pd.DataFrame, funding: pd.DataFrame | None, rule: UniverseRule
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """跑全部变体，返回展示表与逐日收益（DSR 要用原始序列）。"""
    rows: dict[str, dict[str, float]] = {}
    series: dict[str, pd.Series] = {}
    for label, options in VARIANTS.items():
        started = time.time()
        run = run_pairs(panel, options, funding, rule)
        series[label] = run.daily
        rows[label] = {
            **stats_of(run.daily),
            "对数": run.pairs_per_window,
            "暴露": float(run.exposure.mean()),
            "秒": time.time() - started,
        }
    return pd.DataFrame(rows).T, series


def deflated_row(series: dict[str, pd.Series], controls: int) -> pd.Series:
    """多重检验校正：变体 + 随机对照都算试验，取夏普最高的变体做 DSR。"""
    trials = len(series) + controls
    observations = int(min(item.dropna().size for item in series.values()))
    sharpes = [sharpe_ratio(item.dropna()) for item in series.values()]
    expected_max = expected_max_sharpe(
        sharpe_estimator_variance(sharpes, observations), trials
    )
    best = max(series, key=lambda key: sharpe_ratio(series[key].dropna()))
    test = deflated_sharpe(best, series[best], expected_max)
    return pd.Series(
        {
            "试验次数": float(trials),
            "观测数": float(observations),
            "最好变体": best,
            "观测夏普": test.sharpe,
            "期望最大夏普": test.expected_max,
            "DSR": test.dsr,
            "判定": "通过" if test.passed else "未通过",
        }
    )


def _format_cell(column: str, value: object) -> str:
    """变体表的单元格：百分比 / 夏普 / 其余数字三种格式（不出现科学计数法）。"""
    number = float(cast(float, value))
    if column in ("年化", "波动", "回撤"):
        return f"{number:+.1%}"
    if column in ("夏普", "暴露"):
        return f"{number:+.3f}"
    return f"{number:.1f}"


def print_variants(table: pd.DataFrame, deflated: pd.Series) -> None:
    """变体表 + DSR 一行：变体越多，门槛越高。"""
    print("\n-- 6. 参数变体（每行都是一次试验） --")
    header = "".join(pad(column, 13, align="right") for column in table.columns)
    print(f"{pad('', 16)}  {header}")
    for label, row in table.iterrows():
        cells = "".join(
            pad(_format_cell(str(column), value), 13, align="right")
            for column, value in row.items()
        )
        print(f"{pad(str(label), 16)}  {cells}")
    print("\n-- 7. 多重检验校正（DSR） --")
    print(
        f"  试验 {int(deflated['试验次数'])} 次 · 观测 {int(deflated['观测数'])} 根 · "
        f"最好变体「{deflated['最好变体']}」"
    )
    print(
        f"  观测夏普 {deflated['观测夏普']:+.3f} · 零假设期望最大夏普 "
        f"{deflated['期望最大夏普']:+.3f} · DSR {deflated['DSR']:.3f} → {deflated['判定']}"
    )


def main() -> int:
    args = parse_args()
    rule = UniverseRule()
    panel = load_panel("1d", rule, market=args.market)
    mask = tradable_mask(panel, rule)
    funding, funding_label = load_funding(panel.symbols, args.market)
    options = VARIANTS["主口径"]

    print(
        f"面板：{args.market} · {panel.close.shape[1]} 只标的 × {panel.close.shape[0]} 根日线"
        f" · 可交易数峰值 {int(mask.sum(axis=1).max())}"
    )
    run: PairsRun = run_pairs(panel, options, funding, rule)
    report = build_report(
        panel, run, mask, options, funding_label, controls=args.controls
    )
    print_report(report, run)

    variants, series = run_variants(panel, mask, funding, rule)
    deflated = deflated_row(series, args.controls)
    print_variants(variants, deflated)

    if not args.no_save:
        # 按市场分目录：现货与永续是两套结论，写同一目录会互相覆盖
        out_dir = RESULTS_ROOT / "pairs" / args.market
        paths = save_outputs(report, run, out_dir)
        variants.to_parquet(out_dir / "pairs_variants.parquet")
        # DSR 行里既有数字也有文字（最好变体、判定），parquet 列必须同型 → 统一成字符串
        deflated.rename("值").rename_axis("指标").reset_index().astype(str).to_parquet(
            out_dir / "pairs_deflated.parquet"
        )
        print(f"\n已落盘 {len(paths) + 2} 份 parquet → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
