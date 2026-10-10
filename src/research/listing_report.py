"""上新币效应的报告层：把七张表组装、打印并落盘。

读表规则（与 factors / vol_target 一致）：

1. **中位与均值必须一起看**——新币收益分布极度右偏（少数百倍币把均值抬得远高于中位），
   只报均值会读成「有机会」，只报中位会漏掉「极少数暴富」这个事实。
2. **为正值占比比收益本身更可信**——它不受极端值影响，206 只里有多少只赚钱一眼可读。
3. **中位深负不等于可做空**——空头亏损不封顶，必须看 `short_risk` 表的 wipeout 列。
4. **事件研究与组合回测必须并列**——每笔中位损益回答不了「能不能累积」：上市事件在
   时间上重叠、每日再均衡引入波动拖累，两者结论可以完全相反。
"""

from pathlib import Path

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.listing import (
    HORIZONS,
    ListingEvent,
    artifact_summary,
    cohort_table,
    entry_table,
    horizon_table,
    lifecycle_table,
    short_risk_table,
)
from src.research.listing_portfolio import (
    DEFAULT_FEE_ONE_WAY,
    backtest_grid,
    build_price_panel,
)

OUTPUT_DIR = RESULTS_ROOT / "listing"
DEFAULT_HORIZON = 90
TABLE_NAMES: tuple[str, ...] = (
    "listing_artifact",
    "listing_horizon",
    "listing_lifecycle",
    "listing_entry",
    "listing_cohort",
    "listing_short_risk",
    "listing_short_backtest",
)


def build_report(
    events: list[ListingEvent],
    prices: pd.DataFrame | None = None,
    *,
    horizon: int = DEFAULT_HORIZON,
    fee_one_way: float = DEFAULT_FEE_ONE_WAY,
) -> dict[str, pd.DataFrame]:
    """组装七张表；horizon 影响入场扫描与年份分组，prices 不传则从缓存现拼。"""
    artifact = artifact_summary(events)
    panel = build_price_panel(events) if prices is None else prices
    return {
        "listing_artifact": pd.DataFrame([artifact]),
        "listing_horizon": horizon_table(events, HORIZONS),
        "listing_lifecycle": lifecycle_table(events),
        "listing_entry": entry_table(events, horizon=horizon),
        "listing_cohort": cohort_table(events, horizon=horizon),
        "listing_short_risk": short_risk_table(events, horizons=HORIZONS),
        "listing_short_backtest": backtest_grid(panel, events, fee_one_way=fee_one_way),
    }


def save_outputs(
    tables: dict[str, pd.DataFrame], directory: Path = OUTPUT_DIR
) -> list[str]:
    """把六张表写成 parquet，返回写出的路径（字符串）。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in TABLE_NAMES:
        path = target / f"{name}.parquet"
        tables[name].to_parquet(path)
        written.append(str(path))
    return written


def _fmt(value: float, kind: str) -> str:
    """按列类型格式化：pct 是比例，int 是计数，ratio 是两位小数。"""
    if kind == "int":
        return f"{int(value)}"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value:+.2%}"


def _index_label(index: object) -> str:
    """索引转可读字符串；二维网格的索引是元组，用斜杠拼。"""
    if isinstance(index, tuple):
        return "/".join(str(part) for part in index)
    return str(index)


def _print_table(
    title: str, frame: pd.DataFrame, columns: dict[str, tuple[str, str]]
) -> None:
    """打印一张表；columns 是 列名 → (表头, 格式)，格式取 pct / int / ratio。"""
    print(f"\n-- {title} --")
    header = "  ".join(f"{label:>12}" for label, _ in columns.values())
    print(f"{'':>12}  {header}")
    for index, row in frame.iterrows():
        cells = "  ".join(
            f"{_fmt(float(row[key]), kind):>12}" for key, (_, kind) in columns.items()
        )
        print(f"{_index_label(index):>12}  {cells}")


def print_report(
    tables: dict[str, pd.DataFrame], *, horizon: int = DEFAULT_HORIZON
) -> None:
    """打印全部六张表。"""
    artifact = tables["listing_artifact"].iloc[0]
    print("\n=== 首日假象 ===")
    print(f"事件数 {int(artifact['count'])}")
    print(
        f"首根 bar 的 open 等于当日最低价的占比 {float(artifact['open_at_low_share']):.1%}"
        "  ← open 是参考价不是成交价，首日收益不可用"
    )

    _print_table(
        "持有期收益（从上市首日收盘买入）",
        tables["listing_horizon"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "mean": ("均值", "pct"),
            "positive": ("为正", "pct"),
            "p5": ("P5", "pct"),
            "p95": ("P95", "pct"),
        },
    )
    _print_table(
        "生命周期分段（每段独立，非累计）",
        tables["listing_lifecycle"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "mean": ("均值", "pct"),
            "positive": ("为正", "pct"),
        },
    )
    _print_table(
        f"入场时点扫描（第 d 根收盘买入，持有 {horizon} 天）",
        tables["listing_entry"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "mean": ("均值", "pct"),
            "positive": ("为正", "pct"),
        },
    )
    _print_table(
        f"按上市年份（持有 {horizon} 天）",
        tables["listing_cohort"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "positive": ("为正", "pct"),
        },
    )
    _print_table(
        "做空尾部风险（1x 空头在标的翻倍时归零）",
        tables["listing_short_risk"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "wipeout": ("爆仓率", "pct"),
            "worst": ("最差标的", "pct"),
        },
    )
    _print_table(
        "做空组合回测（行=入场时点/持有期，每日等权再均衡，成本按换手）",
        tables["listing_short_backtest"],
        {
            "avg_positions": ("平均持仓", "ratio"),
            "annual_compounded": ("复利年化", "pct"),
            "total_return": ("累计", "pct"),
            "sharpe": ("夏普", "ratio"),
            "annual_vol": ("年化波动", "pct"),
            "max_drawdown": ("回撤", "pct"),
        },
    )
