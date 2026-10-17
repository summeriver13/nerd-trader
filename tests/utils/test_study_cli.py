"""研究脚本 CLI 骨架的契约测试。

钉住三件会被静默改掉的事：helper 不许覆盖各脚本自己的默认值、`--no-save` 必须是
明确的布尔开关、逗号列表必须丢空项（`--symbols ""` 与不传同义）。
"""

from __future__ import annotations

import argparse

from src.utils.study_cli import (
    NO_SAVE_HELP,
    add_market_argument,
    add_no_save,
    add_timeframe_argument,
)


def _build(*, market: str = "spot", timeframe: str = "1d") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_market_argument(parser, default=market)
    add_timeframe_argument(parser, default=timeframe)
    add_no_save(parser)
    return parser


def test_defaults_come_from_the_caller_not_the_helper() -> None:
    assert (_build().parse_args([]).market, _build().parse_args([]).timeframe) == (
        "spot",
        "1d",
    )
    swap = _build(market="swap", timeframe="1m").parse_args([])
    assert (swap.market, swap.timeframe) == ("swap", "1m")


def test_explicit_flags_win_over_defaults() -> None:
    args = _build().parse_args(["--market", "swap", "--timeframe", "4h"])
    assert (args.market, args.timeframe) == ("swap", "4h")


def test_no_save_is_a_flag_defaulting_to_false() -> None:
    assert _build().parse_args([]).no_save is False
    assert _build().parse_args(["--no-save"]).no_save is True


def test_help_text_names_the_actual_default() -> None:
    spot_help = _build().format_help()
    assert "默认 spot" in spot_help and "默认 1d" in spot_help
    swap_help = _build(market="swap", timeframe="1m").format_help()
    assert "默认 swap" in swap_help and "默认 1m" in swap_help
    assert NO_SAVE_HELP in spot_help
