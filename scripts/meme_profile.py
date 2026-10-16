"""meme 币行情体检：四张表（体检 / 生命周期 / 与 BTC / 暴涨暴跌）。

用法：
    uv run python scripts/meme_profile.py             # 全量并落盘
    uv run python scripts/meme_profile.py --no-save   # 只打印

数据来源：本地 OKX 现货日线缓存（12 只 meme + BTC 基准），只读，不触发采集。
口径与读表规则见 src/research/meme_profile.py 的模块 docstring。
"""

import argparse

from src.research.meme_bars import load_universe
from src.research.meme_report import (
    OUTPUT_DIR,
    build_report,
    print_report,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="meme 币行情体检")
    parser.add_argument("--no-save", action="store_true", help="只打印，不落盘")
    return parser.parse_args()


def main() -> int:
    """读缓存 → 四张表 → 打印（可选落盘）。"""
    args = parse_args()
    bars, benchmark = load_universe()

    print("=== 面板口径 ===")
    print(f"meme 池 {len(bars)} 只 · 基准 {benchmark.symbol}")
    print("OKX 现货日线；起点取首根日线，已剔除零成交的冻结日")

    tables = build_report(bars, benchmark)
    print_report(tables, benchmark)

    if not args.no_save:
        written = save_outputs(tables)
        print(f"\n已落盘 {len(written)} 份 parquet → {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
