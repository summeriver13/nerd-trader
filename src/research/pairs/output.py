"""
src/research/pairs/output.py
配对研究的输出层：控制台打印与落盘。

与 `report.py` 分开的理由只有一个：表的**构造**（数据）与表的**呈现**（格式、路径）
是两件事，混在一个文件里两边都会越写越长。落盘按市场分目录——现货与永续是两套
结论，写同一目录会互相覆盖。
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.pairs.backtest import PairsRun
from src.research.pairs.report import PairsReport
from src.utils.console import pad

OUTPUT_DIR: Path = RESULTS_ROOT / "pairs"


def print_report(report: PairsReport, run: PairsRun) -> None:
    """按「结论 → 证据」的顺序打印五张表。"""
    print("\n=== 口径 ===")
    print(f"选对与交易：{report.options_label}")
    print(f"资金费：{report.funding_label}")
    _print_frame(
        "1. 组合 vs 基准",
        report.headline,
        percent_columns=("复利年化", "年化波动", "最大回撤", "累计收益"),
    )
    print("\n-- 2. 单笔分布与集中度 --")
    for key, value in report.trade_stats.items():
        print(f"  {pad(str(key), 14)} {_fmt(value)}")
    _print_frame(
        "3. 成本敏感性",
        report.cost,
        percent_columns=("复利年化", "最大回撤", "累计成本"),
    )
    _print_frame(
        "4. 分段稳定性", report.segments, percent_columns=("复利年化", "最大回撤")
    )
    if not report.control.empty:
        _print_frame(
            "5. 随机对照（同池随机选对）",
            report.control,
            percent_columns=("复利年化", "最大回撤"),
        )
    print(
        f"\n配对笔数 {len(run.trades)} · 成交 {run.rebalances} 次 · "
        f"组合平均 gross 暴露 {run.exposure.mean():.2f}"
    )


def save_outputs(
    report: PairsReport, run: PairsRun, out_dir: Path = OUTPUT_DIR
) -> list[Path]:
    """落盘：五张表 + 逐日贡献（后续研究可直接复用，不必重跑）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "pairs_headline": report.headline,
        "pairs_cost": report.cost,
        "pairs_segments": report.segments,
        "pairs_trades": run.trades,
        "pairs_contributions": run.contributions,
    }
    if not report.control.empty:
        frames["pairs_control"] = report.control
    paths: list[Path] = []
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.reset_index().to_parquet(path)
        paths.append(path)
    return paths


def _print_frame(
    title: str, frame: pd.DataFrame, percent_columns: tuple[str, ...] = ()
) -> None:
    """对齐打印一张表（中文列名按显示宽度算，见 console.pad）。"""
    print(f"\n-- {title} --")
    if frame.empty:
        print("  （空）")
        return
    index_width = max(len(str(name)) for name in frame.index)
    header = "".join(pad(column, 14, align="right") for column in frame.columns)
    print(f"{pad('', index_width)}  {header}")
    for name, row in frame.iterrows():
        cells = "".join(
            pad(_fmt(value, column in percent_columns), 14, align="right")
            for column, value in row.items()
        )
        print(f"{pad(str(name), index_width)}  {cells}")


def _fmt(value: object, percent: bool = False) -> str:
    """百分比/数值/文本三种格式：表格里不能出现科学计数法。"""
    if isinstance(value, str):
        return value
    number = float(cast(float, value))
    if not np.isfinite(number):
        return "n/a"
    if percent:
        return f"{number:+.1%}"
    if abs(number) >= 100:
        return f"{number:+.1f}"
    return f"{number:+.3f}"
