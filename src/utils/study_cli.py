"""研究脚本（`scripts/*_study.py`）的 CLI 骨架：公共参数与逗号列表解析。

存在的理由只有一条：同一套参数在 25~30 个脚本里各写一遍，默认值与帮助文案已经漂成
四五种说法（`--no-save` 有「不落盘」「不落盘，只打印」「只打印不落盘」「只统计不落盘」
四种），读文档的人无法判断这些参数语义是否相同。收口在这里之后：

- `--market` / `--timeframe` / `--no-save` 的语义、默认值来源、帮助文案只有一处；
- 默认值仍由各脚本决定（趋势线默认 `swap`、因子线默认 `spot`），helper 只负责**不覆盖**它。

刻意不做的事：不抽取「载面板 → 跑 → 打印 → 落盘」的主流程。各线的装载口径
（`factors.panel.load_panel`、`universe`、`bars.load_bars`）与产物命名各不相同，
把主流程抽象成回调只会把三行样板换成一层间接。

用法：

    parser = argparse.ArgumentParser(description="截面因子统一测试台")
    add_market_argument(parser, default="spot")
    add_timeframe_argument(parser)
    add_no_save(parser)
"""

from __future__ import annotations

import argparse

MARKET_DEFAULT = "spot"
TIMEFRAME_DEFAULT = "1d"
NO_SAVE_HELP = "只打印，不落盘"


def add_market_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = MARKET_DEFAULT,
    help_text: str | None = None,
) -> None:
    """加 `--market`；默认值由调用方给，帮助文案默认带上它。"""
    parser.add_argument(
        "--market",
        default=default,
        help=help_text or f"市场层：spot（现货）/ swap（永续），默认 {default}",
    )


def add_timeframe_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = TIMEFRAME_DEFAULT,
    help_text: str | None = None,
) -> None:
    """加 `--timeframe`（K 线周期）。"""
    parser.add_argument(
        "--timeframe",
        default=default,
        help=help_text or f"K 线周期，默认 {default}",
    )


def add_no_save(parser: argparse.ArgumentParser) -> None:
    """加 `--no-save`：给了就只打印、不落盘。"""
    parser.add_argument("--no-save", action="store_true", help=NO_SAVE_HELP)
