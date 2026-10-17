"""meme 行情体检的报告层：把四张表组装、打印并落盘。

读表规则（与 factors / listing / vol_target 一致）：

1. **中位与均值必须一起看**——meme 是彩票分布，少数翻倍币把均值抬到远高于中位；
   只报均值会读成「有机会」，只报中位会漏掉「极少数暴富」这个事实。
2. **为正值占比比收益本身更可信**——它不受极端值影响，12 只里几只赚钱一眼可读。
3. **beta 必须拆上涨日/下跌日**——单一 beta 会把「跌得更多」平均掉，而 meme 的
   风险恰恰在下跌段放大。
4. **暴涨后的收益必须与无条件基准比**——meme 的日收益中位本来就是负的，
   不减去 `base_t1` 会把「常态漂移」读成「暴涨后回落」。
"""

from pathlib import Path

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research.meme.meme_bars import CoinBars
from src.research.meme.meme_profile import (
    benchmark_table,
    burst_table,
    lifecycle_table,
    profile_table,
)

OUTPUT_DIR = RESULTS_ROOT / "meme"
TABLE_NAMES: tuple[str, ...] = (
    "meme_profile",
    "meme_lifecycle",
    "meme_benchmark",
    "meme_burst",
)


def build_report(
    bars: dict[str, CoinBars], benchmark: CoinBars
) -> dict[str, pd.DataFrame]:
    """组装四张表；bars 只放 meme 池，benchmark 只进表三。"""
    return {
        "meme_profile": profile_table(bars),
        "meme_lifecycle": lifecycle_table(bars),
        "meme_benchmark": benchmark_table(bars, benchmark),
        "meme_burst": burst_table(bars),
    }


def save_outputs(
    tables: dict[str, pd.DataFrame], directory: Path = OUTPUT_DIR
) -> list[str]:
    """把四张表写成 parquet，返回写出的路径（字符串）。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in TABLE_NAMES:
        path = target / f"{name}.parquet"
        tables[name].to_parquet(path)
        written.append(str(path))
    return written


def _fmt(value: float, kind: str) -> str:
    """按列类型格式化：pct 是比例，int 是计数，ratio 两位小数，num 三位小数。"""
    if kind == "int":
        return f"{int(value)}"
    if kind == "ratio":
        return f"{value:>12.2f}"
    if kind == "num":
        return f"{value:>12.3f}"
    return f"{value:+.2%}"


def _print_table(
    title: str, frame: pd.DataFrame, columns: dict[str, tuple[str, str]]
) -> None:
    """打印一张表；columns 是 列名 → (表头, 格式)，格式取 pct / int / ratio / num。"""
    print(f"\n-- {title} --")
    header = "  ".join(f"{label:>12}" for label, _ in columns.values())
    print(f"{'':>12}  {header}")
    for index, row in frame.iterrows():
        cells = "  ".join(
            f"{_fmt(float(row[key]), kind):>12}" for key, (_, kind) in columns.items()
        )
        print(f"{str(index):>12}  {cells}")


def _aggregate(frame: pd.DataFrame, keys: tuple[str, ...]) -> str:
    """表下的小结：中位数一行，省得每张表都自己算。"""
    return "  ".join(f"{key}中位={float(frame[key].median()):+.2%}" for key in keys)


def print_report(tables: dict[str, pd.DataFrame], benchmark: CoinBars) -> None:
    """打印四张表，外加基准对照与每张表的中位小结。"""
    profile = tables["meme_profile"]
    print(f"\n=== meme 池：{len(profile)} 只 ===")
    print(
        f"总收益中位 {float(profile['total_return'].median()):+.2%} · "
        f"年化波动中位 {float(profile['annual_vol'].median()):.1%} · "
        f"回撤中位 {float(profile['max_drawdown'].median()):.1%} · "
        f"为正 {(profile['total_return'] > 0).sum()}/{len(profile)} 只"
    )
    reference = profile_table({benchmark.symbol: benchmark}).iloc[0]
    print(
        f"对照 {benchmark.symbol}：总收益 {float(reference['total_return']):+.1%} · "
        f"年化波动 {float(reference['annual_vol']):.1%} · "
        f"回撤 {float(reference['max_drawdown']):.1%} · "
        f"年数 {float(reference['years']):.1f}"
    )

    _print_table(
        "1a. 行情体检：收益与风险（现货日线，首根日线起）",
        profile,
        {
            "bars": ("根数", "int"),
            "years": ("年数", "ratio"),
            "total_return": ("总收益", "pct"),
            "annual_return": ("年化", "pct"),
            "annual_vol": ("年化波动", "pct"),
            "max_drawdown": ("最大回撤", "pct"),
            "from_high": ("距高点", "pct"),
        },
    )
    _print_table(
        "1b. 行情体检：分布形状",
        profile,
        {
            "skew": ("偏度", "ratio"),
            "kurt": ("超额峰度", "ratio"),
            "up_share": ("上涨日", "pct"),
            "best_day": ("最好一天", "pct"),
            "worst_day": ("最差一天", "pct"),
        },
    )

    _print_table(
        "2. 生命周期（首根日线后的第 N 天，每段独立不累计）",
        tables["meme_lifecycle"],
        {
            "count": ("样本", "int"),
            "median": ("中位", "pct"),
            "mean": ("均值", "pct"),
            "positive": ("为正", "pct"),
        },
    )

    _print_table(
        "3. 与 BTC：重叠日收益的 beta（上涨日/下跌日分开）",
        tables["meme_benchmark"],
        {
            "days": ("重叠天", "int"),
            "corr": ("相关", "ratio"),
            "beta": ("beta", "ratio"),
            "beta_up": ("beta↑", "ratio"),
            "beta_down": ("beta↓", "ratio"),
            "alpha_annual": ("年化alpha", "pct"),
            "r2": ("R²", "ratio"),
        },
    )
    print(
        "\n  "
        + _aggregate(tables["meme_benchmark"], ("corr", "beta", "beta_up", "beta_down"))
    )

    _print_table(
        "4. 暴涨暴跌之后：动量还是反转（单日 ±10% 为事件）",
        tables["meme_burst"],
        {
            "acf1": ("自相关1", "num"),
            "abs_acf1": ("|收益|自相关1", "num"),
            "base_t1": ("基准T+1", "pct"),
            "up_n": ("暴涨数", "int"),
            "up_t1": ("暴涨T+1", "pct"),
            "up_t5": ("暴涨T+5", "pct"),
            "down_n": ("暴跌数", "int"),
            "down_t1": ("暴跌T+1", "pct"),
            "down_t5": ("暴跌T+5", "pct"),
        },
    )
    burst = tables["meme_burst"]
    print(
        "\n  "
        + _aggregate(burst, ("acf1", "abs_acf1", "base_t1", "up_t1", "down_t1"))
        + f"  暴涨事件合计 {int(burst['up_n'].sum())} · 暴跌事件合计 {int(burst['down_n'].sum())}"
    )
    print(
        "  暴涨后次日低于全样本基准的只数 "
        f"{int((burst['up_t1'] < burst['base_t1']).sum())}/{len(burst)}"
        " · 暴跌后次日低于基准的只数 "
        f"{int((burst['down_t1'] < burst['base_t1']).sum())}/{len(burst)}"
    )
